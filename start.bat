@echo off
rem CtrlF launcher (ASCII only: cmd mis-parses UTF-8 Chinese in .bat files,
rem which silently breaks the "start" line and the backend never launches)
cd /d "%~dp0"
echo ============================================
echo   CtrlF starting...
echo   Backend opens in a new window.
echo   The UI opens in your browser shortly.
echo ============================================
start "CtrlF Backend" cmd /k "chcp 65001 >nul && python -X utf8 run_api.py"
timeout /t 3 >nul
streamlit run ui/app.py
