"""DiffusionNFT reinforcement-learning fine-tuning for the Lyra2 DMD model.

See plan: forward-process RL (re-noise on-policy clean samples; positive/negative
flow-matching loss) over a 3-stage pipeline (sample -> score offline -> train).
"""
