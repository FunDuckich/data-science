import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from recommender import BookRecommender

root = Path(sys.argv[1])
manifest = json.loads((root / 'inference/manifest.json').read_text(encoding='utf-8'))
spark = (SparkSession.builder.master('local[4]').appName('verify-book-recommender')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
context = BookRecommender.load(root / 'inference', spark)
expected = pd.read_json(root / 'reload_sample.json').sort_values(['user_id', 'item_id'])
pairs = spark.createDataFrame([(int(r.user_id), int(r.item_id)) for r in expected.itertuples()], 'user_id int, item_id int')
actual = context.model.transform(pairs).toPandas().sort_values(['user_id', 'item_id'])
delta = np.abs(actual.prediction.to_numpy() - expected.prediction.to_numpy())
assert np.isfinite(delta).all() and delta.max() < 1e-6
routes = {}
for name, user_id, liked in [('warm', manifest['target_id'], None), ('cold', manifest['cold_id'], None), ('onboarding', manifest['cold_id'], manifest['liked_isbns'])]:
    result = context.recommend(user_id, 3, liked)
    assert len(result) == 3 and result.isbn.nunique() == 3
    seen = {row.item_id for row in context.history.filter(F.col('user_id') == user_id).collect()}
    assert not seen.intersection(result.item_id)
    assert not set(result.isbn).intersection(liked or [])
    if name == 'warm':
        queries = spark.createDataFrame([(user_id, int(i)) for i in result.item_id], 'user_id int, item_id int')
        rescored = context.model.transform(queries).toPandas().set_index('item_id').prediction
        assert max(abs(float(r.score) - float(rescored.loc[r.item_id])) for r in result.itertuples()) < 1e-4
    routes[name] = sorted(set(result.source))
assert routes['warm'] == ['als'] and routes['cold'] == ['popular'] and 'content' in routes['onboarding']
unknown = context.recommend(manifest['cold_id'], 3, ['NONEXISTENT-ISBN'])
assert set(unknown.source) == {'popular'}
truth_table = pd.read_csv(root / 'ranking_truth.csv')
truth_sets = truth_table.groupby('user_id').item_id.apply(set).to_dict()
ranking_check = {}
for metric in manifest['ranking_metrics']:
    assert len(truth_sets) == metric['users']
    name = metric['model']
    table = pd.read_csv(root / f'ranking_{name}_top10.csv')
    assert not table.duplicated(['user_id', 'item_id']).any()
    assert table['rank'].between(1, 10).all()
    assert set(table.user_id) == set(truth_sets)
    recalls, ndcgs = [], []
    for user_id, relevant in truth_sets.items():
        ordered = table.loc[table.user_id == user_id].sort_values('rank')
        hits = [int(row.item_id in relevant) for row in ordered.itertuples()]
        dcg = sum(hit / math.log2(int(rank) + 1) for hit, rank in zip(hits, ordered['rank']))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
        recalls.append(sum(hits) / len(relevant))
        ndcgs.append(dcg / ideal)
    actual_ranking = {'recall_at_10': float(np.mean(recalls)), 'ndcg_at_10': float(np.mean(ndcgs)),
                      'catalog_coverage': table.item_id.nunique() / manifest['factor_items']}
    for key, value in actual_ranking.items():
        assert abs(value - metric[key]) < 1e-12, (name, key, value, metric[key])
    ranking_check[name] = actual_ranking
report = {'status': 'passed', 'compared_predictions': len(expected), 'max_absolute_error': float(delta.max()),
          'routes': routes, 'independent_ranking_check': ranking_check}
(root / 'reload_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
spark.stop()
