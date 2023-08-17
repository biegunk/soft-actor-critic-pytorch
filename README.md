# soft-actor-critic-pytorch
A simple PyTorch implementation with mypy typing of the Soft Actor-Critic algorithm with learnable temperature from https://arxiv.org/abs/1812.05905

## Set-up
Create Python environment:
```
python -m venv venv
```

Activate environment and path variables on zsh/bash:
```
source ./init_zsh
```
or
```
source ./init_bash
```

If not on Pytohn 3.9 run the following to recompile the requirements:
```
pip-compile requirements.in
```

Install requirements:
```
pip install --no-cache-dir -r requirements.txt
```
