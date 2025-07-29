<!-- Running IR code on linux (WSL) for official efficient Mamba (selective SSM) implementation.

# Environment installation

python3 -m venv .venv --upgrade-deps

source .venv/bin/activate

pip install -r ir_requirements.txt

pip install mamba-ssm[causal-conv1d] --no-build-isolation

# Open directories from within WSL in Windows explorer

/mnt/c/Windows/explorer.exe . -->


# Relevant commands

pip install -r ir_requirements.txt

python -m src.ir.train

python -m src.ir.export "C:\IR\runs\mambaout\2025-07-11_16-47-41\checkpoints\best.pth"


# Run demo web app

..in the src/ir/app/config.yaml file set checkpoint location of the exported model accordingly:

checkpoint: "C:/IR/runs/mambaout/2025-07-11_16-47-41/checkpoints/scripted_model.pt"

cd \src\ir\app

streamlit run homepage.py
