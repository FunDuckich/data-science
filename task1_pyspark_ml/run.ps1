param([ValidateSet('smoke', 'full')][string]$Profile = 'full')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'work/data/creditcard.csv'))) {
        python download_data.py
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось получить creditcard.csv с Kaggle.' }
    }
    docker compose up -d --wait --wait-timeout 360
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить контейнер. Проверьте Docker Desktop.' }
    $sparkImage = docker compose images -q jupyter_spark
    if ($LASTEXITCODE -ne 0 -or -not $sparkImage) { throw 'Не удалось определить образ контейнера.' }
    docker image inspect $sparkImage --format '{{json .RepoDigests}}' | Set-Content -LiteralPath 'work/image-digest.json' -Encoding utf8
    docker compose exec -T -e "FRAUD_PROFILE=$Profile" jupyter_spark python /home/jovyan/execute_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Выполнение прервано. Проверьте журнал и execution_progress.json; при обработанной ошибке сохраняется failed.ipynb.' }
} finally {
    docker compose down
}
