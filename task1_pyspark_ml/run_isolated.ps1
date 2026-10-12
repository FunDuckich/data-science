param([ValidateSet('smoke', 'full')][string]$Profile = 'full')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$composeArgs = @('compose', '-f', 'docker-compose.yaml', '-f', 'docker-compose.isolated.yaml')
$started = $false
try {
    if (-not (Test-Path -LiteralPath 'work/data/creditcard.csv')) {
        python download_data.py
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось скачать данные.' }
    }
    & docker @composeArgs up -d --wait --wait-timeout 360
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить контейнер.' }
    $started = $true
    $sparkImage = & docker @composeArgs images -q jupyter_spark
    docker image inspect $sparkImage --format '{{json .RepoDigests}}' | Set-Content -LiteralPath 'work/image-digest.json' -Encoding utf8
    foreach ($file in @('execute_notebook.py', 'verify_saved_model.py')) {
        docker cp $file "task1_pyspark_ml_container:/home/jovyan/$file"
        if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $file" }
    }
    foreach ($file in @('fraud_detection.ipynb', 'image-digest.json')) {
        docker cp "work/$file" "task1_pyspark_ml_container:/home/jovyan/work/$file"
        if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $file" }
    }
    & docker @composeArgs exec -T jupyter_spark mkdir -p /home/jovyan/work/data
    foreach ($file in @('creditcard.csv', 'source.json')) {
        docker cp "work/data/$file" "task1_pyspark_ml_container:/home/jovyan/work/data/$file"
        if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $file" }
    }
    & docker @composeArgs exec -T -u root jupyter_spark chown -R 1000:100 /home/jovyan/work
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось подготовить рабочий том.' }
    & docker @composeArgs exec -T -e "FRAUD_PROFILE=$Profile" jupyter_spark python /home/jovyan/execute_notebook.py
    if ($LASTEXITCODE -ne 0) { throw 'Выполнение ноутбука завершилось ошибкой.' }
} finally {
    if ($started) {
        $exportRoot = Join-Path $PSScriptRoot ('work/export_' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $exportRoot | Out-Null
        docker cp "task1_pyspark_ml_container:/home/jovyan/work/artifacts/$Profile" $exportRoot
        $exportStatus = $LASTEXITCODE
        docker cp 'task1_pyspark_ml_container:/home/jovyan/work/fraud_detection.ipynb' (Join-Path $exportRoot 'fraud_detection.ipynb')
        $notebookStatus = $LASTEXITCODE
        & docker @composeArgs down
        if ($exportStatus -ne 0 -or $notebookStatus -ne 0) {
            throw 'Экспорт не завершён; результаты сохранены в томе fraud_work. Не удаляйте том.'
        }
        $destination = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "work/artifacts/$Profile"))
        $incoming = (Resolve-Path -LiteralPath (Join-Path $exportRoot $Profile)).Path
        $workspacePrefix = [IO.Path]::GetFullPath($PSScriptRoot) + [IO.Path]::DirectorySeparatorChar
        if (-not $destination.StartsWith($workspacePrefix) -or -not $incoming.StartsWith($workspacePrefix)) {
            throw 'Путь экспорта находится вне проекта.'
        }
        if (Test-Path -LiteralPath $destination) {
            Move-Item -LiteralPath $destination -Destination (Join-Path $exportRoot "previous_$Profile")
        }
        Move-Item -LiteralPath $incoming -Destination $destination
        Copy-Item -LiteralPath (Join-Path $exportRoot 'fraud_detection.ipynb') -Destination 'work/fraud_detection.ipynb' -Force
    }
}
