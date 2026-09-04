#!/usr/bin/env bash
set -euo pipefail

python3 -m venv ~/nhtsa-venv
source ~/nhtsa-venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install -U \
  jinja2 \
  transformers \
  accelerate \
  peft \
  bitsandbytes \
  pandas \
  pyarrow \
  scikit-learn \
  sentencepiece \
  protobuf \
  python-dotenv

python - <<'PY'
import accelerate
import bitsandbytes
import jinja2
import pandas
import peft
import pyarrow
import sklearn
import transformers

print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
print("sklearn", sklearn.__version__)
print("jinja2", jinja2.__version__)
print("accelerate", accelerate.__version__)
PY
