# Road Damage and route detection

An agent that learns to find the shortest path through a road grid (10×10),
while trying to pass through as many damaged road segments (minor and major)
as possible before reaching the goal. Solved with classic **Q-Learning** (no
neural networks), with an interactive Tkinter visualization.

## How it works

- **10×10 grid** — each cell is one of five types:
  - `FREE` — open road (-0.05 penalty per step)
  - `ROAD_CLOSED` — impassable segment (-1.0 penalty, episode ends)
  - `MINOR_DAMAGE` — minor damage (-0.3 penalty, +0.2 bonus if discovered for the first time)
  - `MAJOR_DAMAGE` — major damage (-0.7 penalty, +0.5 bonus if discovered for the first time)
  - `GOAL` — target cell, bottom-right corner (+100 reward, episode ends)
- The agent picks actions (up/down/right/left) using an **ε-greedy** policy
  and learns via the Bellman equation:
  `Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]`
- Epsilon decays exponentially over episodes (more exploration early on,
  more exploitation later in training)

## Project structure

```
env_v2.py             # Tkinter environment — grid, rendering, step/reward logic
agent_brain.py         # QLearningAgent — Q-table, epsilon-greedy, Bellman update
run_agent_v2.py        # Main training loop + metrics printout + plots
plot_metrics_text.py   # Generates a PNG with a text overview of the metrics
requirements.txt
```

> Note: `run_agent_v2.py` and `plot_metrics_text.py` import modules by name
> (`env_v2`, `agent_brain`), so the files must keep these exact names.

## Running it

```bash
pip install -r requirements.txt
python run_agent_v2.py
```

Training runs inside a Tkinter window (you can watch the agent move across
the grid). After training (800 episodes by default) the script:

- prints detailed metrics to the console (success rate, TD error, convergence...)
- saves the learned Q-table to `q_table.npy`
- generates `metrike_v2.png` — an overview panel with all key numbers
- generates `rezultati_v2.png` — 6 plots (steps per episode, reward, TD error,
  epsilon decay, success rate, damage found)
- opens a separate window visualizing the shortest route found

## Configuration

All training parameters are adjustable — just edit the `CONFIG` dictionary
at the top of `run_agent_v2.py`:

```python
CONFIG = {
    'n_episodes'    : 800,     # number of training episodes
    'max_steps'     : 300,     # max steps per episode
    'learning_rate' : 0.9,     # alpha — learning rate
    'gamma'         : 0.9,     # gamma — discount factor for future rewards
    'epsilon'       : 1.0,     # initial exploration rate
    'epsilon_min'   : 0.01,    # minimum exploration rate
    'epsilon_decay' : 0.995,   # how fast epsilon decays per episode
    'render_speed'  : 0.0,     # delay between rendered steps
    'render_every'  : 100,     # render every Nth episode
}
```

| Parameter        | Default | Description                            |
|-------------------|---------|------------------------------------------|
| `n_episodes`      | 800     | number of training episodes             |
| `max_steps`       | 300     | max steps per episode                   |
| `learning_rate`   | 0.9     | α — learning rate                       |
| `gamma`           | 0.9     | γ — discount factor for future rewards  |
| `epsilon_decay`   | 0.995   | how fast exploration decays             |

The grid layout itself (`GRID_MAP` in `env_v2.py`) can also be edited to
test the agent on different maps.

## Results

_Add `metrike_v2.png` and/or `rezultati_v2.png` here after training, and
optionally a GIF/screenshot of the agent navigating the grid._

## Tech stack

Python, NumPy, Pandas, Matplotlib, Tkinter
