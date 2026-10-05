# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# without WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal distributed training entry point for the released Lyra 2 experiment."""

import argparse
import importlib
import os

import torch.distributed as dist
from loguru import logger

from lyra_2._ext.imaginaire.config import Config, pretty_print_overrides
from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.lazy_config.lazy import LazyConfig
from lyra_2._ext.imaginaire.utils import distributed
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._ext.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init

def launch(config: Config) -> None:
    with distributed_init():
        distributed.init()

    try:
        config.validate()
        config.freeze()
        trainer = config.trainer.type(config)

        with model_init():
            model = instantiate(config.model)

        with data_loader_init():
            dataloader_train = instantiate(config.dataloader_train)
            dataloader_val = instantiate(config.dataloader_val)

        trainer.train(model, dataloader_train, dataloader_val)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Lyra 2 target experiment")
    parser.add_argument(
        "--config",
        default="lyra_2/_src/configs/config.py",
        help="Python config path relative to the Lyra-2 repository root",
    )
    parser.add_argument("--dryrun", action="store_true", help="Resolve and print the config without training")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help='Hydra overrides after "--", for example: -- experiment=lyra2 trainer.max_iter=2',
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        print(config.pretty_print(use_color=True))
        print(pretty_print_overrides(args.opts, use_color=True))
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, os.path.join(config.job.path_local, "config.yaml"))
        print(os.path.join(config.job.path_local, "config.yaml"))
        return
    launch(config)

if __name__ == "__main__":
    logger.info("Starting Lyra 2 training")
    main()
