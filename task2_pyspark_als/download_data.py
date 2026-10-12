from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
data = root / 'work/data'
data.mkdir(parents=True, exist_ok=True)
names = ['Books.csv', 'Ratings.csv', 'Users.csv']
url = 'https://www.kaggle.com/api/v1/datasets/download/arashnic/book-recommendation-dataset'
if not all((data / name).exists() for name in names):
    archive = data / 'book_recommendation.zip'
    if not archive.exists() or not zipfile.is_zipfile(archive):
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError('Kaggle вернул не ZIP. Скачайте три CSV вручную со страницы датасета.')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as package:
        for name in names:
            matches = [member for member in package.namelist() if Path(member).name == name]
            if len(matches) != 1:
                raise RuntimeError(f'В архиве должен быть один файл {name}.')
            target = data / name
            temporary = target.with_suffix('.csv.part')
            with package.open(matches[0]) as source, temporary.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            temporary.replace(target)
manifest = {'url': url, 'files': {}}
for name in names:
    path = data / name
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': digest}
(data / 'source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest), flush=True)
