

from omegaconf import OmegaConf
import os
from huggingface_hub import login
from starvector.validation.svg_validator_base import validator_registry

def get_validator(validator_name, config):
    # Map short engine names to full validator class names
    ENGINE_MAPPING = {
        'vllm': 'StarVectorVLLMValidator',
        'vllm-api': 'StarVectorVLLMAPIValidator',
        'hf': 'StarVectorHFSVGValidator'
    }

    if config.model.generation_engine.lower() in ENGINE_MAPPING:
        config.model.generation_engine = ENGINE_MAPPING[config.model.generation_engine.lower()]

    # Initialize validator
    validator_name = config.model.generation_engine
    validator_class = validator_registry.get(validator_name)

    if not validator_class:
        available_validators = list(validator_registry.keys())
        raise ValueError(
            f"Validator '{validator_name}' is not recognized. "
            f"Available validators: {available_validators}. "
            f"You can use short names: {list(ENGINE_MAPPING.keys())}"
        )
    print(f"Validating with {validator_name}...")
    return validator_class

def main(config):
    validator_class = get_validator(config.model.generation_engine, config)
    validator = validator_class(config)
    
    print(f"Config: {config}")
    print(f"Saving in {validator.out_dir}")
    validator.validate()

if __name__ == "__main__":
    cli_conf = OmegaConf.from_cli()
    if 'config' not in cli_conf:
        raise ValueError("No config file provided. Please provide a config file using 'config=path/to/config.yaml'")
    
    config_path = cli_conf.pop('config')
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, cli_conf)

    # Login to HuggingFace
    # HF_TOKEN = os.getenv('HF_TOKEN')
    # if HF_TOKEN is None:
    #     raise ValueError("HF_TOKEN environment variable is not set.")
    # login(token=HF_TOKEN)

    main(config)
