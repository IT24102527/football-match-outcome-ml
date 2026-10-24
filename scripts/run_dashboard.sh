#!/usr/bin/env bash
# Start the dashboard with one command (macOS / Linux): ./scripts/run_dashboard.sh
# Creates .venv and installs the pinned dashboard requirements on the first run (D-033).
# No database or saved model is needed: the dashboard reads committed files and refits the model if necessary (D-034).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "Creating .venv and installing dashboard requirements (first run only)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r dashboard/requirements.txt
fi

if ! .venv/bin/python -c "import streamlit, plotly" 2>/dev/null; then
    echo "Installing dashboard requirements into the existing .venv..."
    .venv/bin/python -m pip install --quiet -r dashboard/requirements.txt
fi

exec .venv/bin/python -m streamlit run dashboard/app.py "$@"
