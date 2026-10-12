import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

root = Path(__file__).resolve().parent
notebook = json.loads((root / 'work' / 'fraud_detection.ipynb').read_text(encoding='utf-8'))
namespace = dict(np=np, pd=pd, average_precision_score=average_precision_score,
                 precision_recall_curve=precision_recall_curve, roc_auc_score=roc_auc_score, auc=auc)
selected = {'threshold_metrics', 'ranking_metrics', 'choose_threshold', 'log_amount_array'}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    module = ast.parse(''.join(cell['source']))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in selected:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), namespace)

rng = np.random.default_rng(17)
cases = 0
for n in [10, 100, 1000]:
    for rounded in [False, True]:
        y = np.r_[np.ones(max(2, n // 20), dtype=int), np.zeros(n - max(2, n // 20), dtype=int)]
        scores = rng.random(n)
        if rounded:
            scores = np.round(scores, 1)
        for threshold in [0.0, 0.2, 0.5, 0.9]:
            metrics = namespace['threshold_metrics'](y, scores, threshold)
            predicted = scores > threshold
            assert np.isclose(metrics['precision'], precision_score(y, predicted, zero_division=0))
            assert np.isclose(metrics['recall'], recall_score(y, predicted, zero_division=0))
            assert np.isclose(metrics['f1'], f1_score(y, predicted, zero_division=0))
            assert sum(metrics[key] for key in ['tp', 'tn', 'fp', 'fn']) == n
            cases += 1
        chosen, _ = namespace['choose_threshold'](pd.DataFrame({'Class': y, 'score': scores}))
        unique = np.unique(scores)
        candidates = np.r_[0, (unique[1:] + unique[:-1]) / 2]
        expected = max(f1_score(y, scores > t, zero_division=0) for t in candidates)
        assert np.isclose(f1_score(y, scores > chosen, zero_division=0), expected)
        cases += 1

array = np.array([[1.0, 0.0], [2.0, 999.0]])
transformed = namespace['log_amount_array'](array)
assert np.array_equal(array, [[1.0, 0.0], [2.0, 999.0]])
assert np.allclose(transformed[:, -1], np.log1p(array[:, -1]))
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        assert cell['execution_count'] is None or isinstance(cell['execution_count'], int)
print(f'Проверены синтаксис всех ячеек и {cases} сценариев метрик/порогов; преобразование не меняет исходный массив.')
