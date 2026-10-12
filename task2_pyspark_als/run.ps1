param([ValidateSet('smoke', 'full')][string]$Profile = 'full')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    python download_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить данные.' }
    docker compose up -d --wait --wait-timeout 360
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить контейнер. Проверьте Docker Desktop и порт 8888.' }
    docker compose exec -T -e "BOOKS_PROFILE=$Profile" jupyter_spark python /home/jovyan/execute_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Выполнение прервано. Проверьте журнал и execution_progress.json.' }
} finally {
    docker compose down
}
