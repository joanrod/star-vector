
from starvector.model.starvector_arch import StarVectorForCausalLM, StarVectorConfig
from starvector.data.base import ImageTrainProcessor
from starvector.util import dtype_mapping
from transformers import AutoConfig

def load_pretrained_model(model_path, device="cuda", **kwargs):
    model = StarVectorForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    tokenizer = model.model.svg_transformer.tokenizer
    image_processor = ImageTrainProcessor()
    context_len = model.model.query_length + model.model.max_length
    return tokenizer, model, image_processor, context_len

def model_builder(config):
    model_name = config.model.get("model_name", False)

    args = {
        "task": config.model.task,
        "train_image_encoder": config.training.train_image_encoder,
        "ignore_mismatched_sizes": True,
        "starcoder_model_name": config.model.starcoder_model_name,
        "train_LLM": config.training.train_LLM,
        "torch_dtype": dtype_mapping[config.training.model_precision],
        "transformer_layer_cls": config.model.get("transformer_layer_cls", False),
        "use_cache": config.model.use_cache,
    }
    if model_name:
        model = StarVectorForCausalLM.from_pretrained(model_name, **args)
    else:
        starcoder_model_config = AutoConfig.from_pretrained(config.model.starcoder_model_name)

        starvector_config = StarVectorConfig(
            max_length_train=config.model.max_length,
            image_encoder_type=config.model.image_encoder_type,
            use_flash_attn=config.model.use_flash_attn,
            adapter_norm=config.model.adapter_norm,
            starcoder_model_name=config.model.starcoder_model_name,
            torch_dtype=dtype_mapping[config.training.model_precision],
            num_attention_heads=starcoder_model_config.num_attention_heads,
            num_hidden_layers=starcoder_model_config.num_hidden_layers,
            vocab_size=starcoder_model_config.vocab_size,
            hidden_size=starcoder_model_config.hidden_size,
            num_kv_heads=getattr(starcoder_model_config, "num_key_value_heads", None),
        )
        model = StarVectorForCausalLM(starvector_config, **args)
        
    return model

    