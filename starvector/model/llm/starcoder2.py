import torch.nn as nn
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    )
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from starvector.train.util import get_module_class_from_name
import torch

class StarCoderModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(config.starcoder_model_name)
        
        self.max_length = config.max_length
        model_config = AutoConfig.from_pretrained(config.starcoder_model_name, trust_remote_code=True)
        model_config.use_cache = config.use_cache
        model_config.use_bfloat16 = True
        model = AutoModelForCausalLM.from_pretrained(
            config.starcoder_model_name, 
            config=model_config, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True)
        model.resize_token_embeddings(len(self.tokenizer))
        self.transformer = model

        # Prompt the model after image
        self.prompt = '<svg'
        
        transformer_layer_cls = kwargs.get('transformer_layer_cls', 'Starcoder2DecoderLayer')
        self.transformer_layer_cls = get_module_class_from_name(self, transformer_layer_cls)

    def init_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Incude padding and eos tokens in the vocabulary
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})       
        
        self.svg_start_token = "<svg-start>"
        self.svg_end_token = "<svg-end>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        
        self.tokenizer.add_tokens([self.svg_start_token, self.image_start_token, self.text_start_token, self.svg_end_token])
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
        self.svg_end_token_id = self.tokenizer.encode(self.svg_end_token)[0]
        self.tokenizer.padding_side = "left"
        
    def get_fsdp_wrapping_policy(self):
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
        )

        return transformer_block_policy