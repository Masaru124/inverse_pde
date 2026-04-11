from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average model_state_dict across checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to average.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_paths = [Path(p) for p in args.checkpoints]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = [torch.load(p, map_location="cpu", weights_only=True) for p in ckpt_paths]
    state_dicts = [c["model_state_dict"] for c in loaded]

    avg_state = {}
    n = float(len(state_dicts))

    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref)
            for sd in state_dicts:
                acc = acc + sd[key]
            avg_state[key] = acc / n
        else:
            avg_state[key] = ref

    out_ckpt = dict(loaded[0])
    out_ckpt["model_state_dict"] = avg_state
    out_ckpt["averaged_from"] = [str(p) for p in ckpt_paths]

    torch.save(out_ckpt, out_path)
    print(f"Saved averaged checkpoint: {out_path}")


if __name__ == "__main__":
    main()
