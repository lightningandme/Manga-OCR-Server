# --- 配置参数 ---
$PYTHON_VERSION = "3.10.11"
$DIST_DIR = "MangaOCR_Portable_GPU"
$PYTHON_ZIP = "python-$PYTHON_VERSION-embed-amd64.zip"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/$PYTHON_ZIP"

# 1. 创建并清理打包目录
if (Test-Path $DIST_DIR) {
    Write-Host "正在清理旧目录..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $DIST_DIR
}
New-Item -ItemType Directory -Path $DIST_DIR

# 2. 下载并解压嵌入式 Python
Write-Host "正在从官网获取 Python $PYTHON_VERSION..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $PYTHON_URL -OutFile $PYTHON_ZIP
Expand-Archive -Path $PYTHON_ZIP -DestinationPath $DIST_DIR
Remove-Item $PYTHON_ZIP

# 3. 开启 site-packages 支持 (允许 Python 加载第三方库)
Write-Host "配置 Python 运行环境..." -ForegroundColor Cyan
$pth_file = Get-ChildItem -Path $DIST_DIR -Filter "*._pth" | Select-Object -First 1
(Get-Content $pth_file.FullName) -replace '#import site', 'import site' | Set-Content $pth_file.FullName

# 4. 安装 Pip 工具
Write-Host "正在集成 Pip..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$DIST_DIR\get-pip.py"
& "$DIST_DIR\python.exe" "$DIST_DIR\get-pip.py" --no-warn-script-location
Remove-Item "$DIST_DIR\get-pip.py"

# 5. 一键安装 requirements.txt 中的所有内容 (含 GPU 版 Torch)
if (Test-Path "requirements.txt") {
    Write-Host "正在安装依赖清单 (这涉及数 GB 的数据下载，请保持网络稳定)..." -ForegroundColor Magenta
    # 使用 --no-cache-dir 可以减小临时空间占用，但如果下载中断，下次需重新下载
    & "$DIST_DIR\python.exe" -m pip install -r requirements.txt --no-warn-script-location
} else {
    Write-Host "错误：未能在当前目录找到 requirements.txt！" -ForegroundColor Red
    exit
}

# 6. 复制源码文件夹与配置模板
Write-Host "正在同步源码文件夹..." -ForegroundColor Cyan

# A. 复制整个源码目录到整合包
if (Test-Path "suwayomigo_service") {
    Write-Host "正在同步源码文件夹: suwayomigo_service" -ForegroundColor Gray
    Copy-Item -Path "suwayomigo_service" -Destination "$DIST_DIR\" -Recurse -Force

    # --- 安全清理：删除整合包内可能存在的私人 .env 文件 ---
    $targetEnv = Join-Path $DIST_DIR "suwayomigo_service\.env"
    if (Test-Path $targetEnv) {
        Remove-Item $targetEnv -Force
        Write-Host "已从整合包源码中安全移除私人 .env 文件。" -ForegroundColor Yellow
    }
}

# B. 处理配置文件模板
$envExamplePath = Join-Path "suwayomigo_service" ".env.example"
$targetEnvFile = Join-Path $DIST_DIR ".env"

if (Test-Path $envExamplePath) {
    Write-Host "正在从模板生成用户配置文件: .env" -ForegroundColor Green
    # 复制 .env.example 到根目录并重命名为 .env
    Copy-Item -Path $envExamplePath -Destination $targetEnvFile -Force
} else {
    Write-Host "警告：未能在源码中找到 .env.example 模板！" -ForegroundColor Red
}

# 7. 生成 Windows 启动脚本
$bat_content = @"
@echo off
title Manga-OCR Server
set HF_HOME=%~dp0huggingface
echo ==================================================
echo      Manga-OCR GPU 离线服务端正在启动
echo ==================================================
.\python.exe server.py
pause
"@
$bat_content | Out-File -FilePath "$DIST_DIR\开始运行.bat" -Encoding ascii

Write-Host "`n[成功] 离线整合包已准备就绪！" -ForegroundColor Green
Write-Host "路径: $PSScriptRoot\$DIST_DIR" -ForegroundColor Gray
Write-Host "注意：请确保将 'huggingface' 缓存文件夹手动复制到该目录下。" -ForegroundColor Yellow