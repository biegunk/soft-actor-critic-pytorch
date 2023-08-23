# soft-actor-critic-pytorch
A simple PyTorch implementation with mypy typing of the Soft Actor-Critic (SAC) algorithm with learnable temperature from [Soft Actor-Critic: Algorithms and Applications](https://arxiv.org/abs/1812.05905).

This repo runs SAC on the [DeepMind Control Suite](https://www.softwareimpacts.com/article/S2665-9638(20)30009-9/fulltext) with wrappers to make it more similar to OpenAI Gym style environments.

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

## Usage
### Configuration:
Set configurable hyperparameters in `sac/config.py`

### Training:
Training is handled by `run/train.py`. The best scoring policy weights throughout training are saved to `{out_dir}/{domain}/{task}/{timestamp}`, along with the configs and train/test curves.

```
usage: Train SAC agent on specified environment [-h] --domain DOMAIN --task TASK [--n-train N_TRAIN] [--n-test N_TEST]
                                                [--test-every TEST_EVERY] [--out-dir OUT_DIR]
                                                [--device {cpu,gpu,cuda,mps}] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --domain DOMAIN       Domain name of dm-control-suite environment
  --task TASK           Task name of dm-control-suite environment
  --n-train N_TRAIN     Number of train episodes (default: 500)
  --n-test N_TEST       Number of test episodes per evaluation (default: 10)
  --test-every TEST_EVERY
                        Number of train episodes between evaluations (default: 10)
  --out-dir OUT_DIR     Path to directory to save weights, configs and train/test curves (default: out)
  --device {cpu,gpu,cuda,mps}
                        Which device to run on, defaults to GPU if available else CPU
  --seed SEED           Random seed (default: 42)
```

Example:
```
python run/train.py --domain cartpole --task swingup
```

### Evaluating:
Evaluation is handled by `run/eval.py`. Optionally, videos of the evluation episodes are saved to `{out_dir}/{domain}/{task}/{timestamp}`.

```
usage: Evaluate SAC agent on specified environment [-h] --domain DOMAIN --task TASK [--n-test N_TEST] [--out-dir OUT_DIR]
                                                   [--device {cpu,gpu,cuda,mps}] [--weight-path WEIGHT_PATH] [--render]
                                                   [--cam-ids CAM_IDS [CAM_IDS ...]] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --domain DOMAIN       Domain name of dm-control-suite environment
  --task TASK           Task name of dm-control-suite environment
  --n-test N_TEST       Number of test episodes (default: 1)
  --out-dir OUT_DIR     Path to directory to save configs and videos (default: out)
  --device {cpu,gpu,cuda,mps}
                        Which device to run on, defaults to GPU if available else CPU
  --weight-path WEIGHT_PATH
                        Path to saved policy weights
  --render              Whether to save video of test episodes (default: false)
  --cam-ids CAM_IDS [CAM_IDS ...]
                        Camera IDs to render (default: 0)
  --seed SEED           Random seed (default: 42)
```

Example:
```
python run/train.py --domain cartpole --task swingup --weight-path out/cartpole/swingup/1692744472.3513181/actor_weights.pt --render --cam-ids 0 1
```
