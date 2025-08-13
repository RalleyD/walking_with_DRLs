# walking_with_DRLs
Evaluating the performance of modern TD3 against REINFORCE baseline.

# Project Objectives

A comparative study of how existing deep reinforcement learning (DRL) models can solve the
problem of learning a policy that enables simulated robots to walk.

These simulations represent complex continuous environments, posing the following challenges:

*TBD*

The project serves as an A/B testing framework. Comparing a simple, on-policy trial by reward
method i.e REINFORCE against a more modern and complex off-policy method i.e Twin Delayed Deep
Deterministic Policy Gradient (TD3). The performance metrics shall identify the learning ability, as well as the rate of learning (how quickly the model achieves a good performance).

The project shall also research future enhacments, through research i.e Decision Transformers.

# Project Stages

## A/B Framework:

- Model (policy and agent) implementation
- Training loop
- Parameter tuning
- Evaluation

# Theoretical Comparison:
- Research data in the same simulated Gymnasium environments
- Researched model: - Decision Transformers



# REINFORCE

## Limitations

- high variance gradients (due to no baseline), leading to gradient explosion during training updates
  - this causes infinitely large values which could lead to NaN tensors.
- high variance gradients, leading to large actions with large rewards, leading to large gradient updates.
- this requires constrained gradients (TODO cite) to prevent destructive policy updates.
- due to the high variance, normalise the rewards so that gradient magnitudes remain in reasonable range, preventing extreme updates.
- due to the randomness of the network, standard deviations (predictions) could degenerate (s.d -> 0) or become unrealistically wide.
  - clamp the S.Ds to provide numerical stability.
  - S.D -> 0, confident predictions
  - S.D -> inf, random predictions (uncertain)
- training time
  - 10 trials of 1M time steps (recommended from TD3 literature) takes too long. At least, for an M1 macbook pro.
  - reducing to 5 trials. Which, still showed decent perceivable walking performance from TD3 i.e ~4200 mean reward.

# Walker2D environment

## issues

**in macos, if the display goes to sleep, GLFW can cause a segmentation fault due to
nullptr access:**

This ocurrs when trying to render frames in 'human' mode when the display is asleep.

```
Fatal Python error: Segmentation fault

Thread 0x000000016fe47000 (most recent call first):
  <no Python frame>

Current thread 0x00000001f2580840 (most recent call first):
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/glfw/__init__.py", line 1177 in get_video_mode
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py", line 362 in __init__
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py", line 776 in _get_viewer
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py", line 761 in render
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_env.py", line 158 in render
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/mujoco_env.py", line 183 in reset
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/utils/passive_env_checker.py", line 185 in env_reset_passive_checker
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py", line 293 in reset
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/core.py", line 333 in reset
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py", line 400 in reset
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/core.py", line 333 in reset
  File "/Users/danralley/projects/walking_with_DRLs/venv/lib/python3.12/site-packages/gymnasium/wrappers/common.py", line 146 in reset
  File "/Users/danralley/projects/walking_with_DRLs/src/reinforce/reinforce_trainer.py", line 101 in show_policy
  File "/Users/danralley/projects/walking_with_DRLs/src/reinforce/reinforce_trainer.py", line 74 in train
  File "/Users/danralley/projects/walking_with_DRLs/main.py", line 46 in train_reinforce
  File "/Users/danralley/projects/walking_with_DRLs/main.py", line 56 in <module>

Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._mt19937, numpy.random._generator, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand, torch._C, torch._C._dynamo.autograd_compiler, torch._C._dynamo.eval_frame, torch._C._dynamo.guards, torch._C._dynamo.utils, torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._nn, torch._C._sparse, torch._C._special, PIL._imaging, kiwisolver._cext, PIL._imagingft, PIL._imagingmath, PIL._avif, PIL._webp (total: 28)
zsh: segmentation fault  PYTHONFAULTHANDLER=1  /Users/danralley/projects/walking_with_DRLs/main.py
```
