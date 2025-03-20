# hf https://huggingface.co/docs/transformers/main_classes/text_generation
from starvector.validation.svg_validator_base import SVGValidator, register_validator
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from starvector.data.util import rasterize_svg

class SVGValDataset(Dataset):
    def __init__(self, dataset_name, config_name, split, im_size, num_samples, processor):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.im_size = im_size
        self.num_samples = num_samples
        self.processor = processor

        if self.config_name:
            self.data = load_dataset(self.dataset_name, self.config_name, split=self.split)
        else:
            self.data = load_dataset(self.dataset_name, split=self.split)
        
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        svg_str = self.data[idx]['Svg']
        sample_id = self.data[idx]['Filename']
        image = rasterize_svg(svg_str, resolution=self.im_size)
        image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        caption = self.data[idx].get('Caption', "")
        return {
            'Svg': svg_str,
            'image': image,
            'Filename': sample_id,
            'Caption': caption
        }
    
                
@register_validator
class StarVectorHFSVGValidator(SVGValidator):
    def __init__(self, config):
        super().__init__(config)
        # Initialize HuggingFace model and tokenizer here
        self.torch_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }[config.model.torch_dtype]

        # could also use AutoModelForCausalLM
        if config.model.from_checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(self.resume_from_checkpoint, trust_remote_code=True, torch_dtype=self.torch_dtype).to(config.run.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model.name, trust_remote_code=True, torch_dtype=self.torch_dtype).to(config.run.device)
        
        self.tokenizer = self.model.model.svg_transformer.tokenizer
        self.svg_end_token_id = self.tokenizer.encode("</svg>")[0] 

    def get_dataloader(self):
        self.dataset = SVGValDataset(self.config.dataset.dataset_name, self.config.dataset.config_name, self.config.dataset.split, self.config.dataset.im_size, self.config.dataset.num_samples, self.processor)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=self.config.dataset.num_workers)
    
    def release_memory(self):
        # Clear references to free GPU memory
        self.model.model.svg_transformer.tokenizer = None
        self.model.model.svg_transformer.model = None
        
        # Force CUDA garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
    def generate_svg(self, batch, generate_config):
        if generate_config['temperature'] == 0:
            generate_config['temperature'] = 1.0
            generate_config['do_sample'] = False
        outputs = []
        batch['image'] = batch['image'].to('cuda').to(self.torch_dtype)
        # for i, batch in enumerate(batch['svg']):
        if self.task == 'im2svg':
            outputs = self.model.model.generate_im2svg(batch = batch, **generate_config)
        elif self.task == 'text2svg':
            outputs = self.model.model.generate_text2svg(batch = batch, **generate_config)
        return outputs
        