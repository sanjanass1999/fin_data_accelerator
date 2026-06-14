#!/usr/bin/env bash
# One-command setup for Linux / WSL / macOS.
#
#   bash scripts/setup.sh
#
# Creates a virtual environment, installs dependencies, seeds the ChromaDB
# knowledge base, and prints how to launch the dashboard.
#
# If you are behind a corporate SSL-intercepting proxy and pip fails with
# "CERTIFICATE_VERIFY_FAILED", re-run with:
#
#   PIP_TRUSTED=1 bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project root: ${PROJECT_ROOT}"

PY="${PYTHON:-python3}"
echo "Using interpreter: $(${PY} --version)"

if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  ${PY} -m venv venv
fi

PIP_ARGS=""
if [ "${PIP_TRUSTED:-0}" = "1" ]; then
  PIP_ARGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"
  echo "(SSL trust workaround enabled)"
fi

echo "Upgrading pip..."
./venv/bin/python -m pip install --upgrade pip ${PIP_ARGS}

echo "Installing requirements (this can take several minutes the first time)..."
./venv/bin/python -m pip install -r requirements.txt ${PIP_ARGS}

echo "Seeding the knowledge base..."
./venv/bin/python scripts/seed_data.py --reset

echo ""
echo "Setup complete. Launch the platform with:"
echo "    ./venv/bin/python run.py"
echo "Then open http://127.0.0.1:8000/dashboard"
