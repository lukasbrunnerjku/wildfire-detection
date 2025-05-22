Running IR code on linux (WSL) for official efficient Mamba (selective SSM) implementation.

# Environment installation

python3 -m venv .venv --upgrade-deps

source .venv/bin/activate

pip install -r ir_requirements.txt

pip install mamba-ssm[causal-conv1d] --no-build-isolation
