import os

from hydra.core.config_store import ConfigStore
from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._src.models.lyra2_model import (
    Lyra2Model,
    Lyra2T2VConfig,
)
from lyra_2._src.models.lyra2_nft_model import Lyra2NFTModel, Lyra2NFTConfig
from lyra_2._src.models.wan_t2v_model import I4LoraConfig, EMAConfig

# Default DMD LoRA used to initialize the new/old/ref RL adapters.
DMD_LORA_PATH = os.environ.get(
    "DMD_LORA",
    "checkpoints/lora/dmd_distillation.safetensors",
)

fsdp_wan2pt1_lyra2_spatial_config = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(Lyra2Model)(
        config=Lyra2T2VConfig(fsdp_shard_size=8, state_t=20),
        _recursive_=False,
    ),
)

ddp_wan2pt1_lyra2_spatial_config = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(Lyra2Model)(
        config=Lyra2T2VConfig(state_t=20),
        _recursive_=False,
    ),
)

# DiffusionNFT: RL-tune the DMD LoRA. Three RL adapters (new/old/ref) are injected
# from DMD_LORA_PATH; base EMA is disabled (the policy EMA is the `old` adapter,
# updated per NFT epoch).
fsdp_wan2pt1_lyra2_nft_config = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(Lyra2NFTModel)(
        config=Lyra2NFTConfig(
            fsdp_shard_size=8,
            state_t=20,
            nft_enabled=True,
            # enabled=False so the base __init__ does not enable selective-activation
            # checkpointing early (that wraps block submodules and would break LoRA
            # target matching). We inject the 3 RL adapters on clean modules in
            # Lyra2NFTModel.__init__, then enable checkpointing — mirroring inference.
            lora_config=I4LoraConfig(enabled=False, pretrained_lora_path=DMD_LORA_PATH),
            ema=EMAConfig(enabled=False),
        ),
        _recursive_=False,
    ),
)

def lyra_register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1_lyra2_spatial", node=fsdp_wan2pt1_lyra2_spatial_config)
    cs.store(group="model", package="_global_", name="ddp_wan2pt1_lyra2_spatial", node=ddp_wan2pt1_lyra2_spatial_config)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1_lyra2_nft", node=fsdp_wan2pt1_lyra2_nft_config)
