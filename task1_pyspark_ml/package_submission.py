from pathlib import Path
import csv
import json
import zipfile

root = Path(__file__).resolve().parent
result = root / 'work' / 'artifacts' / 'full'
completion = json.loads((result / 'execution_progress.json').read_text(encoding='utf-8'))
assert completion['status'] == 'completed'
reload_result = json.loads((result / 'reload_verification.json').read_text(encoding='utf-8'))
assert reload_result['passed'] and reload_result['independent_process']
experiment = json.loads((result / 'completion.json').read_text(encoding='utf-8'))
assert experiment['profile'] == 'full' and not experiment['test_was_used_for_selection']
notebook = json.loads((root / 'work' / 'fraud_detection.ipynb').read_text(encoding='utf-8'))
code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
assert code_cells and all(cell['execution_count'] is not None for cell in code_cells)
assert not any(output.get('output_type') == 'error' for cell in code_cells for output in cell['outputs'])
with (result / 'tables' / 'cross_validation.csv').open(encoding='utf-8', newline='') as stream:
    searches = list(csv.DictReader(stream))
assert len(searches) == 28 and {row['model'] for row in searches} == {'lr', 'rf', 'gbt'}
with (result / 'tables' / 'test_models.csv').open(encoding='utf-8', newline='') as stream:
    scores = list(csv.DictReader(stream))
assert len(scores) == 6
assert all(sum(int(row[key]) for key in ['tp', 'tn', 'fp', 'fn']) == reload_result['rows'] for row in scores)
assert (result / 'models' / 'best_pipeline' / 'metadata' / '_SUCCESS').exists()
assert (result / 'fraud_detection.html').stat().st_size > 10000
destination = root.parent / 'task1_pyspark_ml_solution.zip'
files = [path for path in root.iterdir() if path.is_file() and path.suffix in {'.py', '.ps1', '.yaml', '.md'}]
files += [root / '.gitignore', root / 'work' / 'fraud_detection.ipynb', root / 'work' / 'pyspark_ml_demo.ipynb',
          root / 'work' / 'image-digest.json', root / 'work' / 'data' / 'source.json',
          root / 'work' / 'data' / 'Churn_Modelling.csv']
for path in result.rglob('*'):
    if not path.is_file():
        continue
    relative = path.relative_to(result)
    if relative.parts[0] in {'reload_input', 'reload_expected'} or path.name in {'running.ipynb', 'failed.ipynb'}:
        continue
    files.append(path)
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(set(files)):
        if path.exists():
            archive.write(path, Path(root.name) / path.relative_to(root))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(json.dumps({'archive': str(destination), 'files': len(files), 'bytes': destination.stat().st_size,
                  'creditcard_csv_included': False}, ensure_ascii=False))
