import os
from starvector.util import (
    set_env_vars,
    flatten_dict,
    get_exp_id, 
    instantiate_from_config, 
    generate_id_name_eval, 
    get_last_checkpoint, 
    model_summary_table, 
    copy_code,
    )
# set_env_vars()
from starvector.train.util import (
    save_checkpoint,
    get_optimizer,
    init_distributed_mode,
    setup_train_env_variables,
    load_fsdp_plugin,
    apply_gradient_checkpointing,
)
import logging
import math
from torch.utils.data import DataLoader
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import os
import time
from starvector.metrics.util import AverageMeter
from util import save_checkpoint, get_optimizer
from starvector.util import get_output_dir
from starvector.model.builder import model_builder
from safetensors.torch import load_file as load_safetensors
from starvector.util import get_config
import torch

from starvector.train.util import load_checkpoint, is_deepspeed, consolidate_deepspeed_checkpoint
logger = get_logger(__name__, log_level="INFO")

def validate(model, dataloader, accelerator):
    loss_meter = AverageMeter()
    model.eval()
    pbar = tqdm(total=len(dataloader), ncols=100, desc="Processing", disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_size = len(batch["image"])
            loss = model(batch)
            loss_meter.update(loss.detach().item(), batch_size)
            pbar.update(1)

    val_loss = (
        accelerator.gather(torch.tensor(loss_meter.avg).to(accelerator.device))
        .float()
        .mean()
        .item()
    )

    accelerator.wait_for_everyone()
    pbar.close()

    return val_loss

def main(config=None):
    print(f"Experiment config: {config}")
    set_env_vars()

    exp_id = get_exp_id(config)
    output_dir = get_output_dir()
    logging_dir = os.path.join(output_dir, config.data.train.params.dataset_name, exp_id)

    if os.path.exists(logging_dir) and not config.training.resume_from_checkpoint:
        config.training.resume_from_checkpoint = get_last_checkpoint(logging_dir)
        config.training.continue_training = True 
            
    # Flatten config dict for logging it
    log_config = flatten_dict(OmegaConf.to_container(config, resolve=True))
    log_config['logging_dir'] = logging_dir # Add logging dir to config

    if config.fsdp.enable:
        init_distributed_mode(config)
        setup_train_env_variables(config)

    # --------------- Datasets ---------------
    train_dataset = instantiate_from_config(config.data.train)
    test_dataset = instantiate_from_config(config.data.test)
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.train.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.test.batch_size, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = config.training.n_epochs * num_update_steps_per_epoch

    global_step = 0
    first_epoch = 0

    model = model_builder(config)
    
    # Instantiate the model, fsdp and accelerator
    if config.training.resume_from_checkpoint:
        if not config.fsdp.enable:
            if is_deepspeed(config.training.resume_from_checkpoint):
                if accelerator.is_main_process:
                    consolidate_deepspeed_checkpoint(config.training.resume_from_checkpoint)
                accelerator.wait_for_everyone()
            model = load_checkpoint(model, config.training.resume_from_checkpoint)
        else: 
            model.load_state_dict(torch.load(os.path.join(config.training.resume_from_checkpoint, "pytorch_model_fsdp.bin")), strict=False)
        if config.training.continue_training:
            global_step = int(os.path.basename(config.training.resume_from_checkpoint).split("-")[1])
            resume_global_step = global_step * config.training.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * config.training.gradient_accumulation_steps)
        else:
            global_step = 0
            first_epoch = 0
            resume_step = 0
            print("Loaded checkpoint but not updating global step")
    
    if config.fsdp.enable:
        fsdp_plugin = load_fsdp_plugin(config, model)
    else:
        fsdp_plugin = None

    # Define accelerator
    kwargs_handler = None
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.model_precision,
        log_with="wandb" if config.project.use_wandb else None,
        project_dir=logging_dir,
        project_config=ProjectConfiguration(logging_dir=logging_dir),
        step_scheduler_with_optimizer=False,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=kwargs_handler
    )

    # --------------- Logging ---------------
    if accelerator.is_main_process:
        if config.project.use_wandb:
            import wandb
            wandb.init(name=exp_id, project=config.project.project, entity=config.project.entity, config=log_config)
            accelerator.init_trackers(
                project_name=config.project.project,
            )
            config.project.wandb_run_id = wandb.run.id
        else:
            run = os.path.split(__file__)[-1].split(".")[0]
            accelerator.init_trackers(run)

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

        # Copy code and dependency versions
        if config.project.copy_code:
            out_dir = os.path.join(logging_dir, "code")
            copy_code(os.path.join(os.path.dirname(__file__), "..", ".."), out_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    total_batch_size = config.data.train.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps

    if accelerator.is_main_process and config.project.use_wandb:
        wandb.log({"total_batch_size": total_batch_size})
        wandb.log({"num_update_steps_per_epoch": num_update_steps_per_epoch})
        wandb.log({"max_train_steps": max_train_steps})

    # accelerate prepare model
    model = accelerator.prepare(model)
    
    # activation/gradient checkpointing
    if config.training.use_gradient_checkpointing:
        print("apply gradient checkpointing")
        model = apply_gradient_checkpointing(model)

    optimizer = get_optimizer(config, model)

    if accelerator.is_main_process:
        print("Train dataset length: ", len(train_dataset))
        print("Test dataset length: ", len(test_dataset))

    # --------------- Training config ---------------
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.lr_warmup_steps * config.training.gradient_accumulation_steps,
        num_training_steps= (len(train_dataloader) * config.training.n_epochs),
    )
    
    optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
        
    loss_meter = AverageMeter()

    if accelerator.is_main_process:    
        model_summary_table(model)  

        if not os.path.exists(os.path.join(logging_dir, 'config.yaml')):
            with open(os.path.join(logging_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(config, f)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.n_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.data.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # --------------- Generation/Validation arguments ---------------
    generation_args = config.generation

    # Need to set some experiment specific arguments
    generation_args.project_name = config.project.project
    generation_args.use_wandb = config.project.use_wandb
    generation_args.id = generate_id_name_eval(generation_args)
    generation_args.out_path = os.path.join(logging_dir, generation_args.id)
    generation_args.start_generation_at_step = config.generation.start_generation_at_step
    generation_args.metrics = config.metrics    
    
    os.makedirs(generation_args.out_path, exist_ok=True)

    # --------------- Training loop ---------------
    total_steps = num_update_steps_per_epoch * config.training.n_epochs
    progress_bar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Training Progress")

    for epoch in range(config.training.n_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            s_time = time.time()

            if config.training.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % config.training.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                loss_meter.update(loss.detach().item(), batch['image'].shape[0])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % config.training.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    val_loss = validate(model, test_dataloader, accelerator)
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                    save_checkpoint(accelerator, model, global_step, logging_dir, config.training.checkpoints_total_limit)
                    model.train()   
            logs = {
                "loss": loss_meter.val, 
                "last_lr": lr_scheduler.get_last_lr()[0], 
                "step": global_step, 
                "step_time": time.time() - s_time,
                "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
    
    accelerator.end_training()

if __name__ == "__main__":
    main(config=get_config())