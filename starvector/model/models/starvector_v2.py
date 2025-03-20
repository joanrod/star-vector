import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from functools import partial
from starvector.model.models.starvector_base import StarVectorBase
from transformers import AutoImageProcessor

class StarVectorStarCoder2(StarVectorBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.processor = AutoImageProcessor.from_pretrained(config._name_or_path, trust_remote_code=True)

    def _get_svg_transformer(self, config, **kwargs):
        from starvector.model.llm.starcoder2 import StarCoderModel # This is a different model than V1, uses StarCoder2
        return StarCoderModel(config, **kwargs)


    def get_fsdp_wrapping_policy(self):
        """V2 specific FSDP wrapping policy"""
        from starvector.model.image_encoder.image_encoder import ImageEncoder

        image_encoder_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={ImageEncoder},
        )

        llm_fsdp_wrapping_policy = self.svg_transformer.get_fsdp_wrapping_policy()
        from starvector.model.adapters.adapter import Adapter

        adapter_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={Adapter},
        )

        return partial(
            _or_policy,
            policies=[
                image_encoder_wrapping_policy,
                llm_fsdp_wrapping_policy,
                adapter_wrapping_policy,
            ],
        )

    def _get_embeddings(self, input_ids):
        """V2 specific embedding method"""
        return self.svg_transformer.transformer.model.embed_tokens(input_ids)

    def _get_svg_text(self, svg_list):
        """V2 specific SVG text preparation"""
        return [t + self.svg_transformer.svg_end_token + self.svg_transformer.tokenizer.eos_token for t in svg_list]

    def _get_im2svg_specific_kwargs(self, kwargs):
        """V2 specific generation kwargs"""
        return {
            # 'eos_token_id': self.svg_transformer.svg_end_token_id,
        }

    def _get_text2svg_specific_kwargs(self, kwargs):
        """V2 specific text2svg generation kwargs"""
        return {
            'eos_token_id': self.svg_transformer.tokenizer.eos_token_id,
        }
