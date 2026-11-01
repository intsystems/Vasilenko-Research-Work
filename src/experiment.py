import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import torch


def environment_metadata():
    """Record the numerical environment used for a CPU experiment."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def save_results(path, config, measurements):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "environment": environment_metadata(),
        "measurements": measurements,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
