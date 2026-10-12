import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

work = Path('/home/jovyan/work')
profile = os.environ.get('FRAUD_PROFILE', 'full')
source = work / 'fraud_detection.ipynb'
output = work / 'artifacts' / profile
output.mkdir(parents=True, exist_ok=True)
for filename in ['completion.json', 'failed.ipynb']:
    previous = output / filename
    if previous.exists():
        previous.unlink()
notebook = nbformat.read(source, as_version=4)
notebook.cells = [cell for cell in notebook.cells if not cell.metadata.get('post_execution_review')]
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.execution_count = None
        cell.outputs = []
started = time.monotonic()


def progress(cell_index=None, **kwargs):
    state = {'profile': profile, 'cell_index': cell_index, 'cells': len(notebook.cells),
             'elapsed_seconds': round(time.monotonic() - started, 1), 'status': 'running'}
    (output / 'execution_progress.json').write_text(json.dumps(state), encoding='utf-8')
    nbformat.write(notebook, output / 'running.ipynb')
    print(json.dumps(state), flush=True)


class ReportingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream' and 'конфигураций' in msg['content'].get('text', ''):
            print(msg['content']['text'].strip(), flush=True)
        if msg['msg_type'] == 'stream' and 'PR-AUC CV=' in msg['content'].get('text', ''):
            print(msg['content']['text'].strip(), flush=True)
        return result


client = ReportingClient(notebook, timeout=14400, kernel_name='python3', allow_errors=False,
                        resources={'metadata': {'path': str(work)}}, on_cell_start=progress,
                        on_cell_executed=progress)
try:
    client.execute()
except BaseException as error:
    nbformat.write(notebook, output / 'failed.ipynb')
    state = {'profile': profile, 'elapsed_seconds': round(time.monotonic() - started, 1),
             'status': 'failed', 'error_type': type(error).__name__}
    (output / 'execution_progress.json').write_text(json.dumps(state), encoding='utf-8')
    raise
nbformat.write(notebook, output / 'fraud_detection.ipynb')
if profile == 'full':
    nbformat.write(notebook, source)
html, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
(output / 'fraud_detection.html').write_text(html, encoding='utf-8')
state = {'profile': profile, 'elapsed_seconds': round(time.monotonic() - started, 1), 'status': 'completed'}
(output / 'execution_progress.json').write_text(json.dumps(state), encoding='utf-8')
print(json.dumps(state), flush=True)
