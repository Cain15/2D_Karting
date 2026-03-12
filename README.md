# Reinforcement Learning Racing Game

A 2D top-down racing game built with Pygame, used as an environment for experimenting with Reinforcement Learning (RL)
algorithms. This project was developed as a student side project experiment to compare the performance and learning
behavior of several RL approaches, ranging from simple approximate Q-learning to modern deep RL agents.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [How to Use](#how-to-use)
4. [State & Action Space](#state--action-space)
5. [Reward Function](#reward-function)
6. [Design Choices](#design-choices)
7. [Model Findings](#model-findings)
    - [Approximate Q-Learning](#approximate-q-learning)
    - [DQN](#dqn-deep-q-network)
    - [PPO](#ppo-proximal-policy-optimization)
    - [DDQN](#ddqn-double-deep-q-network)
    - [Rainbow DQN](#rainbow-dqn)
8. [Hyperparameter Sensitivity](#hyperparameter-sensitivity)
9. [Lessons Learned](#lessons-learned)
10. [Future Work](#future-work)

---

## Project Overview

The environment is a fixed 2D racing track rendered using Pygame. One or more car agents must learn to navigate the
track as fast as possible without going off-road. The game supports both a **manual player mode** and an **AI training
mode**, where multiple agents can be trained simultaneously in the same environment.

The project explores the progression from simple RL methods to state-of-the-art deep RL, with each algorithm building on
the shortcomings of the previous one.

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install pygame torch numpy
```

Place the following files in the same directory:

```
main.py
AIModel.py
Player.py
track_gen.py
track1.tr
assets/
  player.png
  corner.png
  straight.png
  finish.png
```

---

## How to Use

### Playing Manually

In `main.py`, set the AI mode flag to `False` and comment out the model to avoid loading delay:

```python
AI_mode = False
# model = DDQNAgent()
# model.load()
```

Then run:

```bash
python main.py
```

Controls:

- **Arrow Up** — Accelerate
- **Arrow Down** — Brake
- **Arrow Left** — Steer left
- **Arrow Right** — Steer right

### Training an AI Agent

Set `AI_mode = True` in `main.py`. The agent will begin training immediately. Multiple agents (default: 6) run
simultaneously to speed up experience collection.

```python
AI_mode = True
```

The model auto-saves every 5 minutes to `rainbow_model.pth`. The best lap time model is saved separately as
`Rainbow_best.pth`.

To load a previously saved model before training continues:

```python
model = DDQNAgent()
model.load()  # loads rainbow_model.pth by default
```

To start fresh (no loading):

```python
model = DDQNAgent()
# do not call model.load()
```

To load a specific checkpoint:

```python
model.load("Rainbow_best.pth")
```

### Watching a Trained Agent (No Training)

To watch a trained agent without further training, comment out the `model.update(...)` call in `main.py`. This will
prevent the update delay.

### Racing your AI model

Load your specific checkpoint in race.py

```python
model.load("Rainbow_best.pth")
```

Simply run race.py instead of main.py. This will set up the same environment. Except you are playing and the AI is shown
as a ghost driver.

---

## State & Action Space

### State (18-dimensional vector)

| Feature            | Description                                                                                                |
|--------------------|------------------------------------------------------------------------------------------------------------|
| `v_norm`           | Velocity normalized by max velocity                                                                        |
| `sin(θ)`, `cos(θ)` | Player heading encoded as unit vector components                                                           |
| `angle_diff`       | Signed angle between heading and next waypoint, normalized to [-1, 1]                                      |
| `progress`         | Current tile index / total tiles (track progress)                                                          |
| 13× ray distances  | Distance to track boundary at angles: -90°, -70°, -50°, -35°, -20°, -10°, 0°, 10°, 20°, 35°, 50°, 70°, 90° |

All ray distances are clipped and normalized to [0, 1] (max ray length: 300px).

### Action Space (9 discrete actions)

The action space is the Cartesian product of steering × throttle:

|             | Accelerate | Decelerate | Coast |
|-------------|------------|------------|-------|
| **Left**    | ✓          | ✓          | ✓     |
| **Right**   | ✓          | ✓          | ✓     |
| **Neutral** | ✓          | ✓          | ✓     |

---

## Reward Function

| Condition                    | Reward                                                 |
|------------------------------|--------------------------------------------------------|
| Off track (grass)            | −50                                                    |
| Moving forward 1 tile        | +5                                                     |
| Moving backward              | −5                                                     |
| Skipping tiles (teleporting) | −50                                                    |
| Standing still (< 5 units/s) | −2 per step                                            |
| Waypoint alignment bonus     | `corner_reward()` reward based on distance to waypoint |

The waypoint reward guides the car through corners by rewarding progress toward the next geometric waypoint on the
track. This was added after finding that tile-only rewards were insufficient to teach cornering behavior. It speeds up
training by giving more regular rewards. Once the agent reaches a corner they get rewards for moving towards the next
corner, which provides faster feedback.

Lap completion triggers a reset. A lap is only valid if the car visits at least 68 of the track tiles for a real player.
For an AI agent, the rules are stricter: resets are triggered when cutting any part of the track or going off-track.

---

## Design Choices

### Multiple Parallel Agents

Six agents are trained simultaneously within the same game loop, all sharing a single neural network. This increases
experience diversity and speeds up training without requiring true parallelism. Each agent maintains its own N-step
buffer and episode state but contributes to the same replay buffer.

Because all agents share the network, NoisyNet noise is re-sampled independently for each agent before action
selection (`model.q_net.reset_noise()`), ensuring diverse exploration across the population.

### N-Step Returns

Rather than single-step TD targets, the agent uses 10-step returns. This allows reward signals from future timesteps to
propagate back faster, which is particularly helpful on a track where rewards are sparse early in training.

### Noisy Networks for Exploration

Rather than epsilon-greedy, exploration is handled by NoisyNet layers (factorised Gaussian noise). The Q-network remains
in `train()` mode permanently so noise is always active during inference. The target network stays in `eval()` mode,
giving deterministic Q-value targets. This design was chosen because epsilon-greedy decays can be hard to tune on
continuous-action-like problems.

### Soft Target Network Updates

Instead of hard-copying the target network every N steps, a Polyak average (`τ = 0.005`) is applied every training step.
This leads to smoother and more stable target estimates.

### Waypoint System

Corner waypoints are generated from the track layout and used both in the reward function and as a state feature (
`angle_diff`). This inductive bias massively helps the agent learn to turn, since the tile-based reward alone does not
penalize wide cornering lines.

### Loosely Object-Oriented Design

This project started as a small experiment with approximate Q-learning and grew larger than expected. Rather than
redesigning from the top down as scope expanded, the codebase evolved organically — and that shows in the structure.

The game loop in `main.py` is largely procedural. Track rendering, reward calculation, and agent stepping all live as
functions or inline code rather than being wrapped in a `Game` or `Environment` class. This kept iteration fast without
layers of abstraction to trace through.

The exception is the AI side, where proper encapsulation made sense. `DDQNAgent` and its supporting classes (
`RainbowDQN`, `PrioritizedReplayBuffer`, N-step buffers) have well-defined state and a clean `act` / `update` / `save` /
`load` interface. The `Player` class sits somewhere in between — a lightweight data container for per-agent state like
position, velocity, and episode bookkeeping.

For a single-track, single-environment experiment, this balance worked well.


---

## Model Findings

Each model was evaluated on the same fixed track. "Convergence" is defined loosely as the agent completing laps
consistently without going off-track.

---

### Approximate Q-Learning

**Approach:** Linear function approximation over hand-crafted features. A single layer of learned weights maps the
feature vector directly to Q-values, updated via TD error.

**Findings:**

The agent eventually learned to move forward and avoid grass in straight sections, but corners were not consistent. The
reward function was largely the same, only the waypoints where absent. Waypoints were only added at the DDQN model.
After long training sessions the AI only made 3 corners. I also found the model regularly stuck in a local optimal of
driving off track immediately.

The main insight from this stage was that the feature design, reward function and model capacity matter enormously. A
richer, learned representation is far more practical than manually engineered linear features.

**Verdict:** Learned basic forward motion but could not complete laps reliably.

---

### DQN (Deep Q-Network)

**Approach:** Standard DQN with an experience replay buffer and a separate target network. Exploration via
epsilon-greedy with linear decay.

**Findings:**

DQN showed much stronger early performance than approximate Q-learning, successfully learning to navigate straight
sections and mild corners. However, training was unstable on sharp corners — the overestimation bias in vanilla
Q-learning caused the agent to be overly optimistic about certain (wrong) actions, leading to oscillating behavior near
track boundaries.

Epsilon decay scheduling was sensitive: too-fast decay caused the agent to exploit a suboptimal policy, while too-slow
decay wasted training time. The fixed replay buffer (uniform sampling) also meant recent, more-relevant experiences were
diluted by older, low-quality transitions.

This model managed to complete 6 corners after long training sessions. I also found DQN to be regularly stuck in local
optimal, even doing the same strategy of ending the run early on the grass.

**Verdict:** Could complete more corners after extended training, but still inconsistent cornering.

---

### PPO (Proximal Policy Optimization)

**Approach:** On-policy actor-critic with clipped surrogate objective. Separate policy and value networks. Rollouts
collected from multiple parallel agents.

**Findings:**

PPO showed more promise exploring all parts of the track. After 2 hours of training I quickly ditched this model as it
made little to no progress.

**Verdict:** No extensive testing. Quickly went back to a DQN based approach

---

### DDQN (Double Deep Q-Network)

**Approach:** DQN with Double Q-learning — the online network selects the next action while the target network
evaluates it, decoupling action selection from value estimation to reduce overestimation bias. The network was
scaled up significantly to a 4-layer MLP (1024→512→256 hidden units) and the state vector was expanded to 18
features, adding the full ray-cast inputs and waypoint information introduced in this iteration. Exploration
used epsilon-greedy with very slow linear decay (ε = 0.75 → 0.0).

**Findings:**

The overestimation bias problem seen in vanilla DQN was largely resolved. The agent produced much more stable
learning curves and could navigate all corner types after reasonable training time. The expanded feature set and
larger network had a larger practical impact than the algorithmic change — giving the agent the ray distances and
an explicit waypoint signal dramatically improved cornering behavior compared to the earlier sparse feature set.

The main remaining bottleneck was uniform experience replay. Critical transitions (e.g., the moment before going
off track) were undersampled relative to the many uneventful straight-line transitions, slowing learning at the
track's harder corners.

**Verdict:** First algorithm to complete laps. Lap times were competitive but not optimal. Not consistent enough to
learn last part of the track well.

---

### Rainbow DQN

**Approach:** Combines Double DQN, Dueling Networks, NoisyNets, Prioritized Experience Replay (PER), and N-step returns.

**Findings:**

Rainbow DQN was the best-performing algorithm overall. Prioritized replay made a noticeable difference in how quickly
the agent learned difficult corners — high-TD-error transitions (typically near track boundaries) were replayed more
frequently, which is exactly where the agent needed the most learning signal.

NoisyNets replaced epsilon-greedy entirely, and the exploration felt qualitatively different: the agent tried diverse
maneuvers rather than random actions, and noise naturally decreased as the network's weights stabilized. Noisy nets were
easier to manage, since epsilon-decay needed more thought.

The 10-step return dramatically sped up reward propagation: good cornering lines produced positive reward signals that
arrived much earlier in the action sequence compared to 1-step TD.

Training with 6 parallel agents sharing a single Rainbow network led to faster initial learning. The model was also less
sensitive to hyperparameters than earlier approaches.

Remaining issues: Rainbow's combined components add meaningful computational overhead — with 6 parallel agents
training simultaneously, maintaining 60 fps became noticeably harder than with earlier approaches.

**Verdict:** Best lap times and most consistent performance. Recommended approach for this environment.

---

## Lessons Learned

**State representation is the most important design decision.** Both the tabular and early neural approaches suffered
not from algorithm choice but from inadequate features. Adding the waypoint angle difference as an input feature had a
larger impact than switching algorithms.

**Reward shaping requires domain knowledge.** Sparse tile-based rewards were insufficient to teach cornering. The
waypoint-based auxiliary reward required understanding the track geometry to design, but produced dramatically better
agents.

**Parallelism helps more than it complicates.** Running multiple agents simultaneously, even in a single-threaded Python
game loop, significantly improved sample diversity without architectural complexity. The agents naturally encounter
different parts of the track at different times. Multithreading added complexity that is not worth the computational
improvement.

**Off-policy methods suit this environment better than on-policy.** The ability to replay past (and prioritized)
experiences made DQN variants far more sample-efficient than PPO on this task.

**Noisy exploration is qualitatively different from epsilon-greedy.** The agent's behavior with NoisyNets looked more
purposeful — it tried coherent action sequences — compared to the erratic random actions injected by epsilon-greedy.

---

## Results

| Model                  | Best Lap Time | Corners Navigated | Lap Completion  | Consistency                   |
|------------------------|---------------|-------------------|-----------------|-------------------------------|
| Approximate Q-Learning | —             | 3 / 18            | Never           | Low                           |
| DQN                    | —             | 6 / 18            | Never           | Medium within reached corners |
| PPO                    | —             | 0 / 18            | Never           | —                             |
| DDQN                   | ~36.5s        | 18 / 18           | Regular         | High                          |
| Rainbow DQN            | ~33.7s        | 18 / 18           | Most consistent | Highest                       |
| Human (me)             | ~33.1s        | 18 / 18           | consistent      | Higher                        |

## Notes

- **Multiple tracks:** The model was not trained on multiple tracks. But the implementation would allow different
  tracks. (The finish should be on a straight-left for it to work)
- **Distributional RL (C51):** The full Rainbow implementation would include a distributional value head; this was
  omitted for simplicity but could further stabilize learning.
- **Opponent agents:** In the future I could introduce multiple competing cars and explore multi-agent RL (MARL)
  scenarios.
- **Generative AI:** Used to assist in translating academic papers on Reinforcement Learning into concrete
  implementations, and for general coding/documenting support during development.