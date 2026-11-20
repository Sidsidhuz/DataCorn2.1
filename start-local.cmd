@echo off
setlocal
cd /d "%~dp0"
set "AUTOINSIGHT_MODE=local"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "APP_PYTHON=%~1"
if not defined APP_PYTHON if exist "%~dp0.venv-v2\Scripts\python.exe" set "APP_PYTHON=%~dp0.venv-v2\Scripts\python.exe"
if not defined APP_PYTHON if exist "G:\Backend\XAMPP\Anaconda\python.exe" set "APP_PYTHON=G:\Backend\XAMPP\Anaconda\python.exe"
if not defined APP_PYTHON set "APP_PYTHON=python"
echo Starting DataCorn locally at http://127.0.0.1:8503
"%APP_PYTHON%" -c "import streamlit, pandas, scipy, sklearn, shap, reportlab, openpyxl, skops.io, pyarrow, defusedxml"
if errorlevel 1 (
  echo The selected Python environment is incomplete. Run setup-local.cmd first.
  exit /b 1
)
"%APP_PYTHON%" -m v2.launch --mode local --port 8503
if errorlevel 1 (
  echo DataCorn could not start. Read the error above.
  pause
)
endlocal
