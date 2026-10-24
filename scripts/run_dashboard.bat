@echo off
rem Start the dashboard with one command (Windows): scripts\run_dashboard.bat
rem Creates .venv and installs the pinned dashboard requirements on the first run (D-033).
rem No database or saved model is needed: the dashboard reads committed files and refits the model if necessary (D-034).
cd /d "%~dp0.."

if not exist .venv (
    echo Creating .venv and installing dashboard requirements ^(first run only^)...
    python -m venv .venv || exit /b 1
    .venv\Scripts\python -m pip install --quiet --upgrade pip
    .venv\Scripts\python -m pip install --quiet -r dashboard\requirements.txt || exit /b 1
)

.venv\Scripts\python -c "import streamlit, plotly" 2>nul || (
    echo Installing dashboard requirements into the existing .venv...
    .venv\Scripts\python -m pip install --quiet -r dashboard\requirements.txt || exit /b 1
)

.venv\Scripts\python -m streamlit run dashboard\app.py %*
