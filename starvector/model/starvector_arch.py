from transformers import (
    PretrainedConfig,
    PreTrainedModel
)
from torch.nn import CrossEntropyLoss
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union
import torch

from transformers.processing_utils import ProcessorMixin
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, pad
from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers import AutoProcessor

class SimpleStarVectorProcessor(ProcessorMixin):
    attributes = ["tokenizer"]  # Only include tokenizer in attributes
    valid_kwargs = ["size", "mean", "std"]  # Add other parameters as valid kwargs
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, 
                 tokenizer=None,  # Make tokenizer the first argument
                 size=224, 
                 mean=None, 
                 std=None, 
                 **kwargs,
                 ):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # Store these as instance variables
        self.mean = mean
        self.std = std
        self.size = size
        self.normalize = transforms.Normalize(mean=mean, std=std)        
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),
            transforms.Lambda(lambda img: self._pad_to_square(img)),
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize
        ])

        # Initialize parent class with tokenizer
        super().__init__(tokenizer=tokenizer)


    def __call__(self, images=None, text=None, max_length=None, **kwargs) -> BatchFeature:
        """
        Process images and/or text inputs.
        
        Args:
            images: Optional image input(s)
            text: Optional text input(s)
            **kwargs: Additional arguments
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        image_inputs = {}
        if images is not None:
            if isinstance(images, (list, tuple)):
                images_ = torch.stack([self.transform(img) for img in images])
            else:
                images_ = self.transform(images)
            image_inputs = {"pixel_values": images_}
        
        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(
                text, truncation=True, 
                add_special_tokens=True, 
                padding='longest', 
                max_length=max_length, 
                return_tensors="pt"
            )

        return BatchFeature(data={**text_inputs, **image_inputs})

    def _pad_to_square(self, img):
        # Calculate padding to make the image square
        width, height = img.size
        max_dim = max(width, height)
        padding = [(max_dim - width) // 2, (max_dim - height) // 2]
        padding += [max_dim - width - padding[0], max_dim - height - padding[1]]
        return pad(img, padding, fill=255)  # Assuming white padding


AutoProcessor.register(SimpleStarVectorProcessor, SimpleStarVectorProcessor)


class StarVectorConfig(PretrainedConfig):
    model_type = "starvector"

    def __init__(
        self,
        starcoder_model_name: str = "bigcode/starcoderbase-1b",
        image_encoder_type: str = "clip",
        adapter_norm: str = "layer_norm",
        image_size: int = 224,
        max_length: int = 8192,
        max_length_train: int = 8192,
        use_flash_attn: bool = True,
        use_cache: bool = True,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        vocab_size: int = 49152,
        hidden_size: int = 2048,
        num_kv_heads: int = 4,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        kwargs["torch_dtype"] = torch_dtype
        self.starcoder_model_name = starcoder_model_name
        self.image_encoder_type = image_encoder_type
        self.adapter_norm = adapter_norm
        self.image_size = image_size
        self.max_length = max_length
        self.max_length_train = max_length_train
        self.use_flash_attn = use_flash_attn
        self.use_cache = use_cache
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        super().__init__(**kwargs)

class StarVectorForCausalLM(PreTrainedModel):
    config_class = StarVectorConfig
    _no_split_modules = []

    def __init__(self, config: StarVectorConfig, **kwargs):
        super().__init__(config)
        starcoder_model_name = config.starcoder_model_name
        if 'starcoder2' in starcoder_model_name:
            from starvector.model.models.starvector_v2 import StarVectorStarCoder2
            self.model = StarVectorStarCoder2(config=config, **kwargs)
        else:
            from starvector.model.models.starvector_v1 import StarVectorStarCoder
            self.model = StarVectorStarCoder(config=config, **kwargs)
            

    @property
    def supports_gradient_checkpointing(self):
        # If the underlying transformer (e.g., the one in StarCoderModel)
        # supports gradient checkpointing, delegate to it.
        if hasattr(self.model, 'svg_transformer'):
            return getattr(self.model.svg_transformer, 'supports_gradient_checkpointing', False)
        return False

    def gradient_checkpointing_enable(self):
        # Optionally, forward this call to the internal transformer.
        if hasattr(self.model, 'svg_transformer') and hasattr(self.model.svg_transformer, 'gradient_checkpointing_enable'):
            self.model.svg_transformer.gradient_checkpointing_enable()
            
    def forward(self,  vision_embeds, input_ids, num_generations, attention_mask, num_logits_to_keep) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        completion_embeds = self.model._get_embeddings(input_ids)
        inputs_embeds = torch.cat([vision_embeds.repeat(num_generations, 1, 1), completion_embeds], dim=1)

        transformer_outputs = self.model.svg_transformer.transformer.transformer(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]

        if num_logits_to_keep > 0:
            lm_logits = self.model.svg_transformer.transformer.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        else:
            lm_logits = self.model.svg_transformer.transformer.lm_head(hidden_states)

        loss = None
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def generate_im2svg(self, batch, **kwargs):
        return self.model.generate_im2svg(batch, **kwargs)
    
    def generate_im2text(self, batch, **kwargs):
        return self.model.generate_im2text(batch, **kwargs)

    def process_images(self, images):
        return self.model.image_encoder.process_images(images)

