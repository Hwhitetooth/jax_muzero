# JAX MuZero
A JAX implementation of the [MuZero agent](https://www.nature.com/articles/s41586-020-03051-4.pdf).

Everything is implemented in [JAX](https://github.com/google/jax), including the MCTS. The entire search process can be jitted and can run on accelerators such as GPUs.

## Installation
Follow [jax installation](https://github.com/google/jax/#installation), then: 
```shell
pip install -r requirements.txt
pip install autorom
AutoROM -y
pip install gym[atari]
```


## Training
Run the following command for learning to play the Atari game Breakout:
```bash
python -m experiments.breakout
```


## Atari 100K Benchmark Results
Median human-normalized score:

![](https://github.com/Hwhitetooth/jax_muzero/blob/main/images/atari26_median_human.png)

Raw game scores:
![](https://github.com/Hwhitetooth/jax_muzero/blob/main/images/atari26_score.png)


## Repository Structure
```
.
├── algorithms              # Files for the MuZero algorithm.
│   ├── actors.py           # Agent-environment interaction.
│   ├── agents.py           # An RL agent that plans with a learned model by MCTS.
│   ├── haiku_nets.py       # Neural networks.
│   ├── muzero.py           # The training pipeline.
│   ├── replay_buffers.py   # Experience replay.
│   ├── types.py            # Customized data structures.
│   └── utils.py            # Helper functions.
├── environments            # The Atari environment interface and wrappers.
├── experiments             # Experiment configuration files.
├── vec_env                 # Vectorized environment interfaces.
├── conda_env.yml           # Conda environment specification.
├── requirements.txt        # Python dependencies.
├── LICENSE
└── README.md
```


## Resources
* NeurIPS 2020: JAX Ecosystem Meetup, [video](https://www.youtube.com/watch?v=iDxJxIyzSiM) and [slides](https://storage.googleapis.com/deepmind-media/Jax/NeurIPS%20outreach%20session.pdf)
* https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
* https://github.com/YeWR/EfficientZero
