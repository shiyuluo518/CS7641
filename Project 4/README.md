# Project 4, Spring '25 - DeepRacer

![deepracer](https://github.gatech.edu/rldm/P4_deepracer/assets/78388/86684160-fe6f-4a03-972c-078cd9a9afde)

## Clone this repository
```bash
git clone https://github.gatech.edu/rldm/P4_deepracer.git
cd P4_deepracer
```

## Setup and Install Dependencies
This project requires the following to work.
- Docker or Apptainer.
- Conda (or Python 3.10 or higher).
- Linux or Windows machine with an **Intel CPU**.

Please see the detailed setup instructions in [`SETUP.md`](https://github.gatech.edu/rldm/P4_deepracer/blob/main/SETUP.md).

## Usage

Launch the DeepRacer simulation.
```bash
source scripts/start_deepracer.sh \
    [-C=MAX_CPU; default="3"] \
    [-M=MAX_MEMORY; default="6g"]

# example:
# source scripts/start_deepracer.sh -C "3" -M "6g"
```

Interact with the environment via `gymnasium`.
```python
import gymnasium as gym
import deepracer_gym

env = gym.make('deepracer-v0')

observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
```
See the [packages directory](https://github.gatech.edu/rldm/P4_deepracer/tree/main/packages) and the [`usage.ipynb`](https://github.gatech.edu/rldm/P4_deepracer/tree/main/usage.ipynb) notebook for details.

## Project Structure

The project is organized into logical directories:
- `scripts/training/` - Training scripts for all iterations and tasks
- `scripts/utils/` - Utility scripts (comparison, GPU check, video generation)
- `configs/rewards/` - Reward function implementations
- `configs/hyperparams/` - Optimized hyperparameter configurations
- `docs/optimization/` - Optimization guides and plans
- `docs/training/` - Training analysis and documentation
- `results/analysis/` - Analysis scripts for training results

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete details.

## Demo Videos

The following videos demonstrate the trained agent's performance on different race types. All videos are generated using the `src.utils.demo` function.

### 1. Time-Trial

**Video**: [Time-Trial Demo Video](results/demos/reInvent2019_wide-time_trial-time_trial_agent-rl-video-episode-0.mp4)

**Track**: reInvent2019_wide (A to Z Speedway)  
**Race Type**: Time-Trial (no obstacles, no bot cars)  
**Agent**: Trained PPO agent for Time-Trial racing

### 2. Object-Avoidance

**Video**: [Object-Avoidance Demo Video](results/demos/reInvent2019_wide-obstacle_avoidance-obstacle_avoidance_agent-rl-video-episode-0.mp4)

**Track**: reInvent2019_wide (A to Z Speedway)  
**Race Type**: Object-Avoidance (6 obstacles)  
**Agent**: Time-Trial agent adapted for obstacle avoidance

### 3. Head-to-Bot

**Video**: [Head-to-Bot Demo Video](results/demos/reInvent2019_wide-head_to_bot-head_to_bot_agent-rl-video-episode-0.mp4)

**Track**: reInvent2019_wide (A to Z Speedway)  
**Race Type**: Head-to-Bot (3 bot cars)  
**Agent**: Time-Trial agent adapted for head-to-head racing

### Generating Videos

To generate or regenerate these videos, run:
```bash
python scripts/utils/generate_demo_videos.py
```

This script will:
1. Load the latest trained model
2. Generate a Time-Trial video (no obstacles, no bots)
3. Generate an Object-Avoidance video (6 obstacles)
4. Generate a Head-to-Bot video (3 bot cars)

**Note**: For Object-Avoidance and Head-to-Bot videos, you may need to restart the simulation container after updating the configuration to ensure obstacles/bot cars are properly loaded.
