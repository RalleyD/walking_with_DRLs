# Walking with DRLs 🚶‍♂️

> A comprehensive comparative study of deep reinforcement learning algorithms for simulated robot locomotion

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Gym](https://img.shields.io/badge/Gymnasium-0.28+-orange.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3+-purple.svg)](https://github.com/deepmind/mujoco)

## Table of Contents

- [Walking with DRLs 🚶‍♂️](#walking-with-drls-️)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Quickstart](#quickstart)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Project Structure](#project-structure)
  - [Results](#results)
    - [Performance Comparison](#performance-comparison)
    - [Locomotion Demonstrations](#locomotion-demonstrations)
  - [Coming Soon](#coming-soon)
  - [Citation](#citation)
    - [Decision Transformer Data Source](#decision-transformer-data-source)
    - [This Project](#this-project)
  - [License](#license)

## Overview

This project evaluates the performance of modern TD3 (Twin Delayed Deep Deterministic Policy Gradient) against REINFORCE baseline algorithms in simulated robot locomotion tasks. The study demonstrates quantifiable and perceivable improvements of advanced deep reinforcement learning methods over basic policy gradient approaches.

**Key Features:**
- Comparative A/B testing framework for DRL algorithms
- Implementation of REINFORCE and TD3 algorithms
- Evaluation across multiple MuJoCo environments (Walker2d, HalfCheetah, InvertedPendulum)
- Extensible object-oriented design for future algorithm integration
- Performance visualization and metrics tracking

**Research Findings:**
- TD3 achieved 10-15x higher returns than enhanced REINFORCE
- TD3 converged to stable walking within 200,000 timesteps
- Only TD3 produced policies suitable for deployment across all tested environments

## Quickstart

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Gymnasium 0.28+
- MuJoCo 2.3+

### Installation

```bash
# Clone the repository
git clone https://github.com/RalleyD/walking_with_DRLs.git
cd walking_with_DRLs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train TD3 agent
python main.py --algorithm td3 --env Walker2d-v4

# Train REINFORCE agent
python main.py --algorithm reinforce --env Walker2d-v4

# Compare both algorithms
python main.py --compare
```

### Evaluation

```bash
# Evaluate trained model
python main.py --evaluate --model models/td3_walker2d.pth
```

## Project Structure

```
walking_with_DRLs/
├── src/
│   ├── TD3/
│   │   ├── actor_network.py       # Actor policy network
│   │   ├── critic_network.py      # Twin critic networks
│   │   ├── actor_critic_agent.py  # TD3 agent implementation
│   │   └── td3_trainer.py         # Training loop for TD3
│   ├── reinforce/
│   │   ├── policy_network.py      # REINFORCE policy network
│   │   ├── reinforce_agent.py     # REINFORCE agent
│   │   └── reinforce_trainer.py   # Training loop for REINFORCE
│   ├── evaluate/
│   │   └── performance_metrics.py # Performance tracking
│   └── util/
│       └── plotter.py             # Visualization utilities
├── doc/
│   └── rl-dissertation-report.md  # Comprehensive research report
├── plots/                         # Generated learning curves
├── recordings/                    # GIF recordings of trained agents
├── models/                        # Saved model checkpoints
└── main.py                        # Main entry point
```

## Results

### Performance Comparison

| Environment | Algorithm | Mean Returns | Std Deviation | Convergence Time | Stable Solution |
|-------------|-----------|--------------|---------------|------------------|-----------------|
| **Walker2D-v4** | TD3 | ~2800 | ~600 | ~200k steps | ✅ Yes |
| | REINFORCE | ~300 | ~100 | Not achieved | ❌ No |
| **HalfCheetah-v4** | TD3 | ~5000 | ~500 | ~200k steps | ✅ Yes |
| | REINFORCE | ~(-700) | - | Not achieved | ❌ No |
| **InvertedPendulum-v4** | TD3 | ~980 | ~300 | ~200k steps | ✅ Yes |
| | REINFORCE | ~30 | - | Not achieved | ❌ No |

### Locomotion Demonstrations

| Environment | TD3 Results | Demo GIF | Best Performance |
|-------------|-------------|----------|------------------|
| **Walker2D-v4** | Mean: ~2800 ± 600 | *[GIF Placeholder - Walker2D]* | 4682.82 ± 539.64¹ |
| **HalfCheetah-v4** | Mean: ~5000 ± 500 | *[GIF Placeholder - HalfCheetah]* | Research baseline¹ |

¹ *Best results from TD3 research (Fujimoto et al., 2018)*

**Key Achievements:**
- ✅ Stable bipedal walking achieved with TD3
- ✅ 50x faster convergence compared to theoretical Decision Transformer baseline
- ✅ Robust performance across multiple continuous control environments
- ✅ GPU optimization reducing training time by 72%

## Coming Soon

🚧 **Planned Enhancements & Refactoring:**

- **🏗️ Code Architecture Improvements**
  - Refactor code duplication with base classes for agents and trainers
  - Implement unified interface for algorithm comparison and extension

- **📊 Advanced Monitoring & Visualization**
  - Migrate learning curve visualizations and logging to TensorBoard
  - Real-time training progress monitoring with interactive dashboards
  - Enhanced experiment tracking and hyperparameter comparison

- **🤖 Next-Generation Algorithms**
  - Implement own version of minimal Decision Transformer
  - Evaluate transformer-based approaches for locomotion tasks
  - Compare autoregressive sequence modeling vs. traditional RL methods

- **🔬 Research Extensions**
  - Isaac gym for higher-fidelity simulation environments
  

## Citation

### Decision Transformer Data Source

The Walker2D Decision Transformer analysis referenced in this study uses data from:

```bibtex
@misc{minimal_decision_transformer,
    author = {Barhate, Nikhil},
    title = {Minimal Implementation of Decision Transformer},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/nikhilbarhate99/min-decision-transformer}},
}
```

### This Project

```bibtex
@misc{walking_with_drls,
    author = {Ralley, Daniel},
    title = {Walking with DRLs: A Comparative Study of REINFORCE and TD3 Algorithms},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/RalleyD/walking_with_DRLs}},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
