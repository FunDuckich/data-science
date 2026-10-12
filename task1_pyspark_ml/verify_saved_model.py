from pathlib import Path
import json
import os
import sys

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

root = Path(sys.argv[1]).resolve()
spark = (SparkSession.builder.appName('FraudModelIndependentReload')
         .master(f'local[{min(2, os.cpu_count() or 1)}]')
         .config('spark.sql.shuffle.partitions', '4').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
try:
    model = PipelineModel.load(str(root / 'models' / 'best_pipeline'))
    data = spark.read.parquet(str(root / 'reload_input'))
    assert 'Class' not in data.columns and 'weight_10' not in data.columns
    expected = spark.read.parquet(str(root / 'reload_expected'))
    actual = model.transform(data).select('group_id', 'prediction', vector_to_array('probability')[1].alias('score'))
    keys = ['group_id', 'prediction']
    expected_groups = expected.groupBy(*keys).agg(F.count('*').alias('n'), F.min('score').alias('score'))
    actual_groups = actual.groupBy(*keys).agg(F.count('*').alias('n'), F.min('score').alias('score'))
    assert expected_groups.select(*keys, 'n').exceptAll(actual_groups.select(*keys, 'n')).count() == 0
    assert actual_groups.select(*keys, 'n').exceptAll(expected_groups.select(*keys, 'n')).count() == 0
    difference = expected_groups.alias('a').join(actual_groups.alias('b'), keys).select(F.max(F.abs(F.col('a.score') - F.col('b.score')))).first()[0]
    assert difference is not None and difference <= 1e-10
    rows = data.count()
    assert actual.count() == expected.count() == rows
    missing = data.limit(1).withColumn('V1', F.lit(None).cast('double'))
    missing_score = model.transform(missing).select(vector_to_array('probability')[1].alias('score')).first()['score']
    assert missing_score is not None and 0 <= missing_score <= 1
    required = [f'V{i}' for i in range(1, 29)] + ['Amount']
    minimal_score = model.transform(data.select(*required).limit(1)).select(vector_to_array('probability')[1].alias('score')).first()['score']
    assert minimal_score is not None and 0 <= minimal_score <= 1
    result = {'passed': True, 'rows': rows, 'max_score_difference': difference,
              'labels_and_multiplicity_equal': True, 'inference_without_label_and_weights': True,
              'missing_value_inference': True, 'minimal_input_inference': True, 'independent_process': True}
    (root / 'reload_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
finally:
    spark.stop()
