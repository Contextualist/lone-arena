<img align="right" src="media/lone_arena-sketch-small.png" width="160" />

# Lone Arena

When comparing two LLM checkpoints, human evaluation could be tedious.
Let's strip down the evaluation process to just a single question:

![lone_arena-ui-en](media/lone_arena-ui-en.png)

Press <kbd>f</kbd> or <kbd>j</kbd> to choose the winner of each match.

Inspired by [Chatbot Arena](https://chat.lmsys.org).

## Get Started

1. In a Python (>= 3.12) environment, `pip install -r requirements.txt`
2. Fill out `config.toml` with your model endpoint infomation and prompts. See [`config-example.toml`](config-example.toml).
3. Run `python generate.py config.toml` to gather responses from models.
4. Run `python evaluate.py config.toml` to host your competition!


## Approach

Two models/checkpoints are compared by anonymous evaluation of their responses to the same prompt. For each prompt:

1. For each model, generate 8 sample responses. Run a single-elimination tournament to get top 3 responses. (8 matches x 2 models)
2. Let the best responses of two models compete, then 2nd best of two models, then 3rd best. Winner of each gets 4.8, 3.2, 2.0 points, respectively. (3 matches)

Matches are shuffled.
Number of samples and points are configurable.
In the future, I might implement [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) for comparing multiple models.

## Develop

```bash
pip install -r requirements-dev.txt
pre-commit install
```
