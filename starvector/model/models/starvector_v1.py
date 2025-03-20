import torch
import torch.nn as nn
from starvector.model.models.starvector_base import StarVectorBase
from transformers import AutoProcessor

class StarVectorStarCoder(StarVectorBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.processor = AutoProcessor.from_pretrained(config._name_or_path)

    def _get_svg_transformer(self, config, **kwargs):
        from starvector.model.llm.starcoder import StarCoderModel # This uses StarCoder (V1)
        return StarCoderModel(config, **kwargs)

    def _get_embeddings(self, input_ids):
        """V1 specific embedding method"""
        return self.svg_transformer.transformer.transformer.wte(input_ids)

    def _get_svg_text(self, svg_list):
        """V1 specific SVG text preparation"""
        return [t + self.svg_transformer.tokenizer.eos_token for t in svg_list]
