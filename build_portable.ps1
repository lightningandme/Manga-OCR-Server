# build_portable.ps1
# 本文件用于一键生成整合包，进入包含 build_portable.ps1 和 suwayomigo_service 文件夹的根目录
# 在文件夹空白处按住 Shift 键，同时点击 鼠标右键，选择 “在此处打开 PowerShell 窗口”，依次运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\build_portable.ps1

# 以下为脚本代码
# 强制指定当前窗口的编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
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

# 5. 生成启动脚本 (使用英文文件名避免乱码)
Write-Host "5. 生成启动脚本..." -ForegroundColor Cyan
$bat = "@echo off`ntitle Manga-OCR Server`nset HF_HOME=%~dp0huggingface`n.\python.exe .\suwayomigo_service\server.py`npause"

# 将文件名改为 Run_Server.bat
$bat | Out-File -FilePath "$DIST_DIR\[Run_Server].bat" -Encoding ascii

Write-Host "`n[Done] Please manually copy the 'huggingface' folder into $DIST_DIR" -ForegroundColor Green