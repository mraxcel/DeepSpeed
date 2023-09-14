# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
import fsspec
import io

class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f"[Torch] Checkpoint {tag} is about to be saved!", ranks=[0])

    def makedirs(self, path, exist_ok=False):
        fsspec.makedirs(path, exist_ok=True)

    def save(self, state_dict, path: str):
        logger.info(f"[Torch] Saving {path}...")
        bytesbuffer = io.BytesIO()
        torch.save(state_dict, bytesbuffer)
        with fsspec.open(path, "wb") as f:
            f.write(bytesbuffer.getvalue())
        logger.info(f"[Torch] Saved {path}.")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        with fsspec.open(path, "rb") as f:
            partition = torch.load(f, map_location=map_location)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True
