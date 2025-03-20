import os
import torch
import transformers
import os
from starvector.util import checkpoint_key
import glob
import shutil
import builtins
import datetime
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
import functools
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import PrecisionType

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


from transformers import (
    AutoConfig, 
    AutoModelForCausalLM
)
from starvector.model.starvector_arch import StarVectorConfig, StarVectorForCausalLM
from starvector.train.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


def is_deepspeed(checkpoint_dir):
    # Check zero_to_fp32.py file (generated only in deepspeed training)
    return os.path.exists(os.path.join(checkpoint_dir, 'zero_to_fp32.py'))

def consolidate_deepspeed_checkpoint(checkpoint_dir):
    path_state_dict = os.path.join(checkpoint_dir, 'weights.pt')
    if not os.path.exists(path_state_dict):
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, path_state_dict)

def load_checkpoint(model, checkpoint_dir):
    candidate_files = ['weights.pt', 'pytorch_model.bin', 'model.safetensors']
    
    # Determine the correct file to load
    for candidate in candidate_files:
        path_state_dict = os.path.join(checkpoint_dir, candidate)
        if os.path.exists(path_state_dict):
            break
    else:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")

    # Load the state dict based on file type
    if path_state_dict.endswith('.safetensors'):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(path_state_dict)
    else:
        state_dict = torch.load(path_state_dict)

    # Handle FSDP or module prefix
    if list(model.state_dict().keys())[0].startswith('module'):
        new_state_dict = {'module.' + key: val for key, val in state_dict.items()}
    else:
        new_state_dict = state_dict

    # Handle Tied Weights
    if hasattr(model, 'tie_weights'):
        # Remove the lm_head.weight key if it exists and tie_weights will handle it
        new_state_dict.pop("model.svg_transformer.transformer.lm_head.weight", None)

    # Load the state dict into the model with strict=False to ignore missing keys
    model.load_state_dict(new_state_dict, strict=False)  # Allow missing keys

    # Ensure weights are tied after loading
    model.tie_weights()  # This method should tie the weights internally

    return model

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM
)
from starvector.model.starvector_arch import StarVectorConfig, StarVectorForCausalLM

def push_model_to_hub(model, new_model_name, tokenizer, processor):
    # Register the model for HF
    AutoConfig.register("starvector", StarVectorConfig)
    AutoModelForCausalLM.register(StarVectorConfig, StarVectorForCausalLM)
    StarVectorConfig.register_for_auto_class()
    StarVectorForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model.push_to_hub(new_model_name, commit_message=new_model_name, private=True)
    tokenizer.push_to_hub(new_model_name, commit_message=new_model_name, private=True)
    processor.push_to_hub(new_model_name, commit_message=new_model_name, private=True)

# push_model_to_hub(self.model, new_model_name, self.tokenizer, self.processor)
def save_checkpoint(accelerator, model, global_step, logging_dir, checkpoint_limit):
    print("Saving checkpoint! Global Step: " + str(global_step))
    save_checkpoint_dir = os.path.join(logging_dir, f"checkpoint-{global_step}")
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(save_checkpoint_dir)

    chkp_dirs = sorted(glob.glob(os.path.join(logging_dir, "checkpoint-*")), key = checkpoint_key)
    chkp_to_remove = chkp_dirs[:-checkpoint_limit]
    for chkp in chkp_to_remove:
        if accelerator.is_main_process:
            try:
                shutil.rmtree(chkp)
            except:
                print("could not remove checkpoint")
    print(f"Saved state to {save_checkpoint_dir}")

def push_model_to_hub(model, new_model_name, hf_token=None):
    tokenizer = model.model.svg_transformer.tokenizer
    # Register the model for HF
    AutoConfig.register("starvector", StarVectorConfig)
    AutoModelForCausalLM.register(StarVectorConfig, StarVectorForCausalLM)
    StarVectorConfig.register_for_auto_class()
    StarVectorForCausalLM.register_for_auto_class("AutoModelForCausalLM")


    model.push_to_hub(new_model_name, commit_message=new_model_name, private=True, token=hf_token)
    tokenizer.push_to_hub(new_model_name, commit_message=new_model_name, private=True, token=hf_token)
    
    processor = model.model.image_encoder.processor
    from starvector.data.base import ImageTrainProcessor
    if not isinstance(processor, ImageTrainProcessor):
        processor.push_to_hub(new_model_name, commit_message=new_model_name, private=True, token=hf_token)

def get_optimizer(config, model):
    optimizer = config.training.optimizer
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
            eps=config.training.adam_epsilon,
        )
    elif optimizer == "adafactor":
        optimizer = transformers.Adafactor(
            model.parameters(),
            lr=config.training.lr,
            relative_step=False,
            scale_parameter=False,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
    return optimizer


def init_distributed_mode(config):
    """
    Initializes torch distributed
    """
    assert all(var in os.environ for var in ['WORLD_SIZE', 'LOCAL_RANK', 'RANK'])

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist_url = 'env://'

    torch.cuda.set_device(local_rank)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, dist_url, local_rank), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    print_only_on_master(rank == 0)

def print_only_on_master(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        kwargs['flush'] = True
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)
    
    builtins.print = print
    
def setup_train_env_variables(config):
    """
    Set environment variables needed by FSDP and accelerate
    """
    mixed_precision = config.training.model_precision.lower()

    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.")

    os.environ["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    if config.fsdp.enable:
        # We have to manually set some of the FSDP arguments as environment variables as these are not exposed by the FSDP Plugin API
        os.environ['ACCELERATE_USE_FSDP']="true"
        os.environ['FSDP_USE_ORIG_PARAMS']=str(config.fsdp.use_orig_params).lower()
        os.environ['FSDP_FORWARD_PREFETCH']=str(config.fsdp.forward_prefetch).lower()

        if config.fsdp.cpu_ram_efficient_loading and not config.fsdp.sync_module_states:
             raise ValueError("When using `fsdp.cpu_ram_efficient_loading` set `fsdp.sync_module_states` to `True`")

        os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING']=str(config.fsdp.cpu_ram_efficient_loading).lower()
        os.environ['FSDP_SYNC_MODULE_STATES']=str(config.fsdp.sync_module_states).lower()

def load_fsdp_plugin(config, model):
    if config.fsdp.enable:
        # get mixed precsion dtype
        mixed_precision_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,
        }[config.training.model_precision]

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            auto_wrap_policy=model.model.get_fsdp_wrapping_policy(),
            mixed_precision_policy=MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=mixed_precision_dtype,
                buffer_dtype=mixed_precision_dtype,
            ),
            sharding_strategy={
                "sdp": ShardingStrategy.SHARD_GRAD_OP,
                "ddp": ShardingStrategy.NO_SHARD,
                "fsdp": ShardingStrategy.FULL_SHARD,
                "hsdp": ShardingStrategy.HYBRID_SHARD,
            }[config.fsdp.sharding_strategy],
            backward_prefetch=config.fsdp.backward_prefetch,
            cpu_offload=config.fsdp.cpu_offload,
        )
    else:
        fsdp_plugin = None
    
    return fsdp_plugin


def apply_gradient_checkpointing(model):
    """ Apply gradient checkpointing to Transformer cls of the LLM """

    def check_fn(submodule):
        return isinstance(submodule, model.model.svg_transformer.transformer_layer_cls)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )

    # Wait for all processes to finish
    torch.distributed.barrier()

    return model

def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class