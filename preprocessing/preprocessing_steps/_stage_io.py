"""Checkpoint helpers for the numbered command-line examples."""
import json
from dataclasses import asdict
import numpy as np
from settings import CONFIG, OUTPUT_DIR

def read_stage(name):
    with np.load(OUTPUT_DIR / name, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if str(data.pop("_config")) != json.dumps(asdict(CONFIG), sort_keys=True):
        raise ValueError("Settings changed; use a new output directory and restart at step 1.")
    return data

def write_stage(name, data, **updates):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    data = {**data, **updates, "_config": json.dumps(asdict(CONFIG), sort_keys=True)}
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    print(f"Saved {path}")
