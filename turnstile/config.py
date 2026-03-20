"""Configuration for the Turnstile multi-turn red-teaming pipeline.

Dataclass hierarchy for structured experiment configuration.
Mirrors REDKWEEN's config.py but adapted for multi-turn conversations
against JailbreakBench goals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    mode: str = "buffered"       # "buffered" (SAC-like) | "memoryless" (PPO-like)
    buffer_size: int = 200       # FIFO cap, only used in buffered mode
    lora_iters: int = 50
    lora_lr: float = 1e-5
    batch_size: int = 4
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class ZooConfig:
    A: float = 0.0              # prob of sampling from zoo (0 = pure self-play)
    update_interval: int = 1    # add to zoo every N rounds
    max_size: int = 50


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    output_dir: str = "experiments"

    # Models
    adversary_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    victim_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_model: str = "meta-llama/Llama-Guard-3-1B"

    # Multi-turn parameters
    num_turns: int = 5
    candidates_per_round: int = 30

    # Loop parameters
    rounds: int = 10
    harden_victim: bool = False   # False = frozen victim (Phase 2 default)

    # Adversary training
    training: TrainingConfig = field(default_factory=TrainingConfig)
    zoo: ZooConfig = field(default_factory=ZooConfig)

    # Paths (adversary)
    adapter_path: str = "adapters"
    data_path: str = "data"

    # Deduplication (Jaccard on concatenated adversary turns)
    dedup_similarity_threshold: float = 0.5

    # Hidden state collection for T-SAE
    save_hidden_states: bool = True
    hidden_state_layer: int | None = None  # None = middle layer

    # Metrics
    metrics_file: str = "metrics.jsonl"

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> ExperimentConfig:
        """Build config from command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Turnstile multi-turn red-teaming experiment"
        )
        parser.add_argument("--name", type=str, default="default")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output-dir", type=str, default="experiments")
        parser.add_argument("--rounds", type=int, default=10)
        parser.add_argument("--candidates", type=int, default=30)
        parser.add_argument("--num-turns", type=int, default=5)
        parser.add_argument("--mode", type=str, default="buffered",
                            choices=["buffered", "memoryless"])
        parser.add_argument("--buffer-size", type=int, default=200)
        parser.add_argument("--lora-iters", type=int, default=50)
        parser.add_argument("--lora-lr", type=float, default=1e-5)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--A", type=float, default=0.0,
                            help="Zoo sampling probability (0=self-play)")
        parser.add_argument("--zoo-interval", type=int, default=1)
        parser.add_argument("--zoo-max-size", type=int, default=50)
        parser.add_argument("--adversary-model", type=str,
                            default="meta-llama/Llama-3.2-1B-Instruct")
        parser.add_argument("--victim-model", type=str,
                            default="meta-llama/Llama-3.1-8B-Instruct")
        parser.add_argument("--harden-victim", action="store_true",
                            help="Enable victim hardening (default: frozen)")
        parser.add_argument("--no-hidden-states", action="store_true",
                            help="Skip per-turn hidden state collection")
        parser.add_argument("--hidden-layer", type=int, default=None,
                            help="Victim layer for hidden states (None=middle)")
        parsed = parser.parse_args(args)

        return cls(
            name=parsed.name,
            seed=parsed.seed,
            output_dir=parsed.output_dir,
            adversary_model=parsed.adversary_model,
            victim_model=parsed.victim_model,
            num_turns=parsed.num_turns,
            rounds=parsed.rounds,
            candidates_per_round=parsed.candidates,
            harden_victim=parsed.harden_victim,
            training=TrainingConfig(
                mode=parsed.mode,
                buffer_size=parsed.buffer_size,
                lora_iters=parsed.lora_iters,
                lora_lr=parsed.lora_lr,
                batch_size=parsed.batch_size,
            ),
            zoo=ZooConfig(
                A=parsed.A,
                update_interval=parsed.zoo_interval,
                max_size=parsed.zoo_max_size,
            ),
            save_hidden_states=not parsed.no_hidden_states,
            hidden_state_layer=parsed.hidden_layer,
        )
