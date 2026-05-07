from __future__ import annotations

import argparse

import yaml

from .run_stage29_cc3m_phaseA_modular import run_core


def run(cfg: dict) -> None:
    run_core(cfg, stage_name="stage31_wavcaps_scaling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
