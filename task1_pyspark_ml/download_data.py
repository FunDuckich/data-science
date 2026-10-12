from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
data = root / 'work' / 'data'
data.mkdir(parents=True, exist_ok=True)
target = data / 'creditcard.csv'
url = 'https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud'
if not target.exists():
    archive = data / 'creditcardfraud.zip'
    if not archive.exists() or not zipfile.is_zipfile(archive):
        temporary = archive.with_suffix('.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'pyspark-ml-coursework/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError('Kaggle вернул не ZIP. Скачайте creditcard.csv с официальной страницы датасета.')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as package:
        matches = [name for name in package.namelist() if Path(name).name == 'creditcard.csv']
        if len(matches) != 1:
            raise RuntimeError('В архиве должен находиться один creditcard.csv.')
        extracted = target.with_suffix('.csv.part')
        with package.open(matches[0]) as source, extracted.open('wb') as destination:
            shutil.copyfileobj(source, destination)
        extracted.replace(target)
with target.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = {'source': url, 'filename': target.name, 'bytes': target.stat().st_size, 'sha256': digest}
(data / 'source.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False), flush=True)
