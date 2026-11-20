@echo off
setlocal
cd /d "%~dp0"
set "BASE_PYTHON=%~1"
if not defined BASE_PYTHON if exist "G:\Backend\XAMPP\Anaconda\python.exe" set "BASE_PYTHON=G:\Backend\XAMPP\Anaconda\python.exe"
if not defined BASE_PYTHON set "BASE_PYTHON=python"
if not exist "%~dp0.venv-v2\Scripts\python.exe" (
  "%BASE_PYTHON%" -m venv "%~dp0.venv-v2"
  if errorlevel 1 goto failed
)
set "LOCAL_PYTHON=%~dp0.venv-v2\Scripts\python.exe"
echo Installing the pinned local runtime. Package download requires internet once.
"%LOCAL_PYTHON%" -m pip install -r v2/requirements.txt
if errorlevel 1 goto failed
"%LOCAL_PYTHON%" -m pip check
if errorlevel 1 goto failed
"%LOCAL_PYTHON%" -c "import streamlit, pandas, scipy, sklearn, shap, reportlab, openpyxl, skops.io, pyarrow, defusedxml; print('Local runtime dependencies are ready.')"
if errorlevel 1 goto failed
echo Setup complete. Launch with start-local.cmd.
exit /b 0
:failed
echo Setup failed. The application was not started. Check the error above and retry.
exit /b 1
