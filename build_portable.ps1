# build_portable.ps1
$PYTHON_VERSION = "3.10.11"
$DIST_DIR = "MangaOCR_Portable_GPU"
$PYTHON_ZIP = "python-$PYTHON_VERSION-embed-amd64.zip"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/$PYTHON_ZIP"

if (Test-Path $DIST_DIR) { Remove-Item -Recurse -Force $DIST_DIR }
New-Item -ItemType Directory -Path $DIST_DIR

Write-Host "1. 下载 Python 核心..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $PYTHON_URL -OutFile $PYTHON_ZIP
Expand-Archive -Path $PYTHON_ZIP -DestinationPath $DIST_DIR
Remove-Item $PYTHON_ZIP

Write-Host "2. 配置环境与 Pip..." -ForegroundColor Cyan
$pth_file = Get-ChildItem -Path $DIST_DIR -Filter "*._pth" | Select-Object -First 1
(Get-Content $pth_file.FullName) -replace '#import site', 'import site' | Set-Content $pth_file.FullName
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$DIST_DIR\get-pip.py"
& "$DIST_DIR\python.exe" "$DIST_DIR\get-pip.py" --no-warn-script-location
Remove-Item "$DIST_DIR\get-pip.py"

Write-Host "3. 安装依赖 (读取 requirements.txt)..." -ForegroundColor Magenta
& "$DIST_DIR\python.exe" -m pip install -r requirements.txt --no-warn-script-location

Write-Host "4. 同步源码并进行安全清理..." -ForegroundColor Cyan
# 复制源码文件夹
Copy-Item -Path "suwayomigo_service" -Destination "$DIST_DIR\" -Recurse -Force
# 物理删除整合包内的私人 .env
$privateEnv = Join-Path $DIST_DIR "suwayomigo_service\.env"
if (Test-Path $privateEnv) { Remove-Item $privateEnv -Force }

# 处理配置文件模板
$exampleEnv = "suwayomigo_service\.env.example"
if (Test-Path $exampleEnv) {
    Copy-Item -Path $exampleEnv -Destination "$DIST_DIR\.env" -Force
    Write-Host "已根据模板生成根目录 .env" -ForegroundColor Green
}

Write-Host "5. 生成启动脚本..." -ForegroundColor Cyan
$bat = "@echo off`ntitle Manga-OCR Server`nset HF_HOME=%~dp0huggingface`n.\python.exe .\suwayomigo_service\server.py`npause"
$bat | Out-File -FilePath "$DIST_DIR\开始运行.bat" -Encoding ascii

Write-Host "`n[完成] 请手动将 huggingface 文件夹放入 $DIST_DIR" -ForegroundColor Green