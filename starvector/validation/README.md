# StarVector Validation

This module provides validation functionality for StarVector models, allowing evaluation of SVG generation quality across different model architectures and generation backends.

## Overview

The validation framework consists of:

1. A base `SVGValidator` class that handles common validation logic
2. Specific validator implementations for different backends:
   - `StarVectorHFSVGValidator`: Uses HuggingFace generation API
   - `StarVectorVLLMValidator`: Uses vLLM for faster generation
   - `StarVectorVLLMAPIValidator`: Uses vLLM through REST API

## 1. Running Validation

### Using HuggingFace Backend

```bash
# StarVector-1B
python starvector/validation/validate.py \
config=configs/generation/hf/starvector-1b/im2svg.yaml \
dataset.name=starvector/svg-stack

# StarVector-8B 
python starvector/validation/validate.py \
config=configs/generation/hf/starvector-8b/im2svg.yaml \
dataset.name=starvector/svg-stack
```

### vLLM Backend

For using the vLLM backend (StarVectorVLLMAPIValidator), first install our StarVector fork of VLLM, [here](https://github.com/starvector/vllm).

```bash
git clone https://github.com/starvector/vllm.git
cd vllm
pip install -e .
```

Then, launch the using the vllm config file (it uses StarVectorVLLMValidator):

```bash
# StarVector-1B
python starvector/validation/validate.py \
config=configs/generation/vllm/starvector-1b/im2svg.yaml \
dataset.name=starvector/svg-stack

# StarVector-8B
python starvector/validation/validate.py \
config=configs/generation/vllm/starvector-8b/im2svg.yaml \
dataset.name=starvector/svg-stack
```

## 2. Creating a New SVG Validator

To create a new validator for a different model or generation backend:

1. Create a new class inheriting from `SVGValidator`
2. Implement required abstract methods:
   - `__init__(self, config)`: Initialize the validator with the given config
   - `get_dataloader(self)`: Get the dataloader for the given dataset
   - `generate_svg(self, batch)`: Generate SVG from input batch
3. Add the new validator to the registry in `starvector/validation/__init__.py`

Example:

```python
from .svg_validator_base import SVGValidator, register_validator

@register_validator
class MyNewValidator(SVGValidator):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model/client here
        
    def generate_svg(self, batch, generate_config):
        # Implement generation logic
        # Return list of generated SVG strings
        pass
        
    def get_dataloader(self):
        # Implement dataloader logic
        pass
```

## Key Features

The validation framework provides:

- Automatic metrics calculation and logging
- WandB integration for experiment tracking
- Temperature sweep for exploring generation parameters
- Comparison plot generation
- Batch processing with configurable settings

## Configuration

Validation is configured through YAML files in `configs/generation/`. Key configuration sections:

```yaml
model:
  name: "model_name"  # HF model name or path
  task: "im2svg"      # Task type
  torch_dtype: "float16"  # Model precision

dataset:
  dataset_name: "svg-stack"  # Dataset to validate on
  batch_size: 1
  num_workers: 4

generation_params:
  temperature: 0.2
  top_p: 0.9
  max_length: 1024
  # ... other generation parameters

run:
  report_to: "wandb"  # Logging backend
  out_dir: "outputs"  # Output directory
```

## Output Structure

The validator creates the following directory structure:

```
out_dir/
├── {model}_{dataset}_{timestamp}/
│   ├── config.yaml           # Run configuration
│   ├── results/
│   │   ├── results_avg.json  # Average metrics
│   │   └── all_results.csv   # Per-sample metrics
│   └── {sample_id}/         # Per-sample outputs
│       ├── metadata.json
│       ├── {sample_id}.svg
│       ├── {sample_id}_raw.svg
│       ├── {sample_id}_gt.svg
│       ├── {sample_id}_generated.png
│       ├── {sample_id}_original.png
│       └── {sample_id}_comparison.png
```
