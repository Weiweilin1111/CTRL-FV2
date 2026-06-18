# CtrlF 一鍵啟動：先開後端 API（新視窗），再開 Streamlit 前端
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; python run_api.py"
Set-Location $root
streamlit run ui/app.py
