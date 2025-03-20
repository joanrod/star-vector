import os
import importlib
import hashlib
import re
import time
import subprocess
import logging
import shlex
import os
import shutil
import fnmatch
from huggingface_hub import login
import torch
from omegaconf import OmegaConf

dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "no": "no"
}

#  -------------- Metrics  -------------- 
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}"
        num /= 1000
    return f"{num:.1f}P"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def set_env_vars():
    HF_HOME = os.environ['HF_HOME']

    if HF_HOME is None:
        raise EnvironmentError("HF_HOME environment variable is not defined.")

    os.makedirs(HF_HOME, exist_ok=True)
    # os.environ['TRANSFORMERS_CACHE'] = HF_HOME
    os.environ['HUGGINGFACE_HUB_CACHE'] = HF_HOME
    os.environ['TORCH_HOME'] = HF_HOME
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_HUB_CACHE'] = HF_HOME
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    os.environ['TOKENIZERS_PARALLELISM']="False"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    HF_TOKEN = os.environ['HF_TOKEN']

    if HF_TOKEN is None:
        raise EnvironmentError("HF_TOKEN environment variable is not defined.")
    time.sleep(1) # wait for the token to be saved
    login(HF_TOKEN)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def hash_dict(exp_dict):
    """Create a hash for an experiment. Credtts to github.com/haven-ai!

    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters

    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")

    for k in sorted(exp_dict.keys()):
        if "." in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError(f"{exp_dict[k]} tuples can't be hashed yet, consider converting tuples to lists")
        elif isinstance(exp_dict[k], list) and len(exp_dict[k]) and isinstance(exp_dict[k][0], dict):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]

        dict2hash += str(k) + "/" + str(v)
    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id

def get_exp_id(config):
    exp_hash_id = hash_dict(dict(config))  
    if config.model.model_name is not None:
        model_name = config.model.model_name.split("/")[1]
    else:
        model_name = config.model.starcoder_model_name.split("/")[1] + "_" + config.model.image_encoder_type
    exp_id = f"{config.project.project}-{config.model.max_length}-{model_name}-{exp_hash_id}"
    print("\n" + "Experiment ID: " + exp_id + "\n")
    return exp_id

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("No target in config")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def generate_id_name_eval(args):
    id_name = f"len_{args.max_length}"
    if args.use_nucleus_sampling:
        id_name += "_nucleus"
        id_name += f"_top_p_{args.top_p:.2f}"
    
    if args.num_beams > 1:
        id_name += "_beam_search"
        id_name += f"_beams_{args.num_beams}"    
    else:
        if not args.use_nucleus_sampling:
            id_name += "_greedy"
    id_name += f"_rep_pen_{args.repetition_penalty:.2f}"
    id_name += f"_len_pen_{args.length_penalty:.2f}"
    id_name += f"_temp_{args.temperature:.2f}"
    return id_name

def get_last_checkpoint(log_dir):
    """Get the last checkpoint.

    Returns
    -------
    last_checkpoint: str
        The last checkpoint
    """
    
    pattern = re.compile(r"checkpoint-(\d+)")
    files = os.listdir(log_dir)
    checkpoints = [f for f in files if pattern.match(f)]
    if len(checkpoints) == 0:
        return None
    steps = [int(pattern.match(c).group(1)) for c in checkpoints]
    max_step = max(steps)
    last_checkpoint = f"checkpoint-{max_step}"
    
    return os.path.join(log_dir, last_checkpoint)

def model_summary_table(model):
    total_params = 0
    name_col_width = 20  # set the width of the name column
    print("\n")
    print(f"| {'Submodel Name'.ljust(name_col_width)} | Number of Parameters |")
    print("|" + "-" * name_col_width + "|---------------------|")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        total_params += num_params
        print(f"| {name.ljust(name_col_width)} | {num_params:>20,} |")

    print("|" + "-" * name_col_width + "|---------------------|")
    print(f"| {'Total'.ljust(name_col_width)} | {total_params:>20,} |")
    print("\n")

def checkpoint_key(checkpoint_dir):
    return int(checkpoint_dir.split("-")[-1])

def subprocess_call(cmd_string):
    """Run a terminal process.

    Parameters
    ----------
    cmd_string : str
        Command to execute in the terminal

    Returns
    -------
    [type]
        Error code or 0 if no error happened
    """
    return subprocess.check_output(shlex.split(cmd_string), shell=False, stderr=subprocess.STDOUT).decode("utf-8")

def copy_code(
        src_path, 
        dst_path, 
        verbose=1, 
        exclude_list=['__pycache__', 'wandb', '.vscode', '.ipynb_checkpoints', 'project_baselines', 'assets', 'tmp']):
    time.sleep(0.5) 
    if verbose:
        print("  > Copying code from %s to %s" % (src_path, dst_path))

    os.makedirs(dst_path, exist_ok=True)

    rsync_avialable = len(subprocess.run(['which', 'rsync'], capture_output=True, text=True).stdout) > 0

    if rsync_avialable: # TODO: validate this works
        rsync_cmd_base = f"rsync -av -r -q --delete-before --exclude='.*' --exclude '__pycache__/'"
        
        exclude_options = " ".join([f"--exclude='{filename}'" for filename in exclude_list])

        rsync_cmd = f"{rsync_cmd_base} {exclude_options} {src_path} {dst_path}"
        
        if os.path.exists(os.path.join(src_path, ".havenignore")):
            rsync_cmd += f" --exclude-from={os.path.join(src_path, '.havenignore')}"
        
        copy_code_cmd = rsync_cmd
        subprocess_call(copy_code_cmd)
    else:
        logging.warning("rsync not available. Doing a hard copy of the code folder.")
        for dirpath, dirs, files in os.walk(src_path):
            if any(ex in dirpath for ex in exclude_list):
                continue
            for filename in fnmatch.filter(files, '*'):
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst_path, src_file.replace(src_path+'/', ''))
                if src_file == dst_file:
                    continue 
                dst_dir = os.path.dirname(dst_file)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                if not os.path.isfile(dst_file):  # check if destination is already a file
                    shutil.copy2(src_file, dst_file)
    time.sleep(0.5)

def get_output_dir():
    # get the environment variable if it exists
    output_dir = os.environ.get("OUTPUT_DIR", None)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "logs")
    return output_dir

def get_config():
    base_conf = OmegaConf.load("configs/models/default.yaml")
    cli_conf = OmegaConf.from_cli()
    specific_conf = OmegaConf.load(cli_conf.pop('config')) if 'config' in cli_conf else {}
    config = OmegaConf.merge(base_conf, specific_conf, cli_conf)
    if config.training.resume_from_checkpoint:
        if not os.path.exists(os.path.join(os.path.dirname(config.training.resume_from_checkpoint), 'config.yaml')):
            config.training.resume_from_checkpoint = get_last_checkpoint(config.training.resume_from_checkpoint)
            cli_conf.training.resume_from_checkpoint = config.training.resume_from_checkpoint
        pretrained_conf = OmegaConf.load(os.path.join(os.path.dirname(config.training.resume_from_checkpoint), 'config.yaml'))
        model_resume_conf = pretrained_conf.pop('model')
        specific_conf['model'] = model_resume_conf
    config = OmegaConf.merge(config, specific_conf, cli_conf)
    return config