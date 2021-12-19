# Offline Reinforcement Learning Codebased on the [Unstable Baselines](https://github.com/x35f/unstable_baselines)

---
## Stable Algorithms (Runnable and have good performance):

## Unstable Algorithms (Runnable but does not have SOTA performance)

* [Batch-Constrained Deep Q-Learning](https://arxiv.org/abs/1812.02900)

## Under Development Algorithms

---
## Quick Start
### Install
Please install [unstable_baselines](https://github.com/x35f/unstable_baselines) firstly. Then it needs to install [D4RL](https://github.com/rail-berkeley/d4rl) under the `unstable_baslines/offline_rl`.

``` shell
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

### To run an algorithm
``` shell
python3 /path/to/algorithm/main.py /path/to/algorithm/configs/some-config.json args(optional)
```