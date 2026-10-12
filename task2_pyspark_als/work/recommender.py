from pathlib import Path

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession, Window, functions as F


def als_candidates(model, users, history, k, item_count):
    pending = users.select('user_id').distinct().localCheckpoint(eager=True)
    pending.count()
    result = None
    size = min(max(50, k), item_count)
    while True:
        candidates = (model.recommendForUserSubset(pending, size)
            .select('user_id', F.explode('recommendations').alias('r'))
            .select('user_id', F.col('r.item_id').alias('item_id'), F.col('r.rating').alias('score'))
            .join(history.select('user_id', 'item_id'), ['user_id', 'item_id'], 'left_anti')
            .withColumn('rank', F.row_number().over(Window.partitionBy('user_id').orderBy(F.desc('score'), 'item_id')))
            .filter(F.col('rank') <= k).localCheckpoint(eager=True))
        candidates.count()
        if result is None:
            result = candidates
        else:
            replacement = result.join(pending, 'user_id', 'left_anti').unionByName(candidates).localCheckpoint(eager=True)
            replacement.count()
            result.unpersist()
            candidates.unpersist()
            result = replacement
        missing = (pending.join(result.groupBy('user_id').count(), 'user_id', 'left')
                   .filter(F.coalesce(F.col('count'), F.lit(0)) < k).select('user_id').localCheckpoint(eager=True))
        count = missing.count()
        pending.unpersist()
        pending = missing
        if count == 0 or size == item_count:
            break
        size = min(size * 2, item_count)
    pending.unpersist()
    return result


class BookRecommender:
    def __init__(self, spark, model, catalog, history, popularity):
        self.spark, self.model = spark, model
        self.catalog, self.history, self.popularity = catalog, history, popularity
        self.item_count = model.itemFactors.count()

    @classmethod
    def load(cls, directory, spark=None):
        spark = spark or SparkSession.builder.appName('book-recommendations-inference').master('local[4]').getOrCreate()
        root = Path(directory)
        return cls(spark, ALSModel.load(str(root / 'model')),
                   spark.read.parquet(str(root / 'catalog')),
                   spark.read.parquet(str(root / 'history')),
                   spark.read.parquet(str(root / 'popularity')))

    def recommend(self, user_id, k=3, liked_isbns=None, cold_items_only=False):
        if not isinstance(user_id, int) or not -(2**31) <= user_id < 2**31:
            raise ValueError('user_id должен быть целым числом int32')
        if not isinstance(k, int) or not 1 <= k <= 100:
            raise ValueError('k должен быть от 1 до 100')
        liked = sorted(set(str(value).strip().upper() for value in (liked_isbns or [])))
        if len(liked) > 100:
            raise ValueError('передайте не более 100 любимых книг')
        selected = self.catalog.filter(F.col('isbn').isin(liked)).collect() if liked else []
        known = {row.isbn for row in selected}
        unknown = sorted(set(liked) - known)
        if unknown:
            print('isbn отсутствуют в каталоге и пропущены:', unknown)
        seen = self.history.filter(F.col('user_id') == user_id).select('item_id')
        pool = self.catalog.join(seen, 'item_id', 'left_anti').filter(~F.col('isbn').isin(liked))
        pool = pool.join(self.popularity, 'item_id', 'left').fillna({'readers': 0, 'smoothed_mean': 0.0})
        warm = self.model.userFactors.filter(F.col('id') == user_id).limit(1).count() > 0
        primary = None
        if warm and not cold_items_only:
            users = self.spark.createDataFrame([(user_id,)], 'user_id int')
            extra_seen = self.catalog.filter(F.col('isbn').isin(liked)).select(F.lit(user_id).alias('user_id'), 'item_id')
            history = self.history.unionByName(extra_seen)
            predictions = als_candidates(self.model, users, history, k, self.item_count)
            primary = pool.join(predictions.select('item_id', 'score'), 'item_id').withColumn('source', F.lit('als'))
            primary = primary.withColumn('reason', F.lit('факторы пользователя и книги, изученные на обучении'))
        elif selected:
            authors = [r.author for r in selected if r.author]
            publishers = [r.publisher for r in selected if r.publisher]
            if cold_items_only:
                pool = pool.join(self.model.itemFactors.select(F.col('id').alias('item_id')), 'item_id', 'left_anti')
            primary = pool.withColumn('score', 2.0 * F.col('author').isin(authors).cast('int') + F.col('publisher').isin(publishers).cast('int'))
            primary = primary.filter(F.col('score') > 0).withColumn('source', F.lit('content'))
            primary = primary.withColumn('reason', F.lit('совпадение автора (+2) и/или издателя (+1)'))
        columns = ['item_id', 'isbn', 'title', 'author', 'score', 'source', 'reason', 'readers', 'smoothed_mean']
        records = [] if primary is None else primary.orderBy(F.desc('score'), F.desc('readers'), 'isbn').select(columns).limit(k).collect()
        if len(records) < k and not cold_items_only:
            ids = [r.item_id for r in records]
            fallback = (pool.filter(~F.col('item_id').isin(ids)).orderBy(F.desc('readers'), F.desc('smoothed_mean'), 'isbn')
                        .withColumn('score', F.col('readers').cast('double')).withColumn('source', F.lit('popular'))
                        .withColumn('reason', F.lit('популярность по уникальным читателям обучающей выборки')))
            records += fallback.select(columns).limit(k - len(records)).collect()
        import pandas as pd
        return pd.DataFrame([dict(rank=i + 1, **r.asDict()) for i, r in enumerate(records)])


_context = None


def recommend_books(user_id, k=3, liked_isbns=None):
    global _context
    if _context is None:
        _context = BookRecommender.load(Path(__file__).parent / 'artifacts/full/inference')
    return _context.recommend(user_id, k, liked_isbns)
