# vllm https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html

from .svg_validator_base import SVGValidator, register_validator
from starvector.data.util import rasterize_svg, clean_svg, use_placeholder
from svgpathtools import svgstr2paths
from vllm import LLM, SamplingParams
from datasets import load_dataset
from torch.utils.data import DataLoader

@register_validator
class StarVectorVLLMValidator(SVGValidator):
    def __init__(self, config):
        super().__init__(config)

        model_name = config.model.name
        if config.model.from_checkpoint:
            model_name = config.model.from_checkpoint
        
        self.llm = LLM(model=model_name, trust_remote_code=True, dtype=config.model.torch_dtype)

        self.get_dataloader(config)

    def generate_svg(self, batch, generate_config):

        prompt_start = "<image-start>"

        model_inputs_vllm = []
        for i, sample in enumerate(batch['Svg']):
            image = rasterize_svg(sample, self.config.dataset.im_size)
            model_inputs_vllm.append({
                "prompt": prompt_start,
                "multi_modal_data": {"image": image}
            })

        sampling_params = SamplingParams(
            temperature=generate_config['temperature'],
            top_p=generate_config['top_p'],
            top_k=generate_config['top_k'],
            max_tokens=generate_config['max_length'],
            n=generate_config['num_generations'],
            frequency_penalty=generate_config['frequency_penalty'],
            repetition_penalty=generate_config['repetition_penalty'],
            presence_penalty=generate_config['presence_penalty'],
            min_p=generate_config['min_p'],
        )
        completions = self.llm.generate(model_inputs_vllm, 
            sampling_params=sampling_params, 
            use_tqdm=False)

        outputs = []
        for i in range(len(completions)):
            for j in range(len(completions[i].outputs)):
                outputs.append(completions[i].outputs[j].text)

        return outputs

    def get_dataloader(self, config):
        data = load_dataset(config.dataset.dataset_name, config.dataset.config_name, split=config.dataset.split)

        if config.dataset.num_samples != -1:
            data = data.select(range(config.dataset.num_samples))

        self.dataloader = DataLoader(data, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=self.config.dataset.num_workers)

    def release_memory(self):
        if self.llm is not None:
            # Delete the LLM instance
            del self.llm
            self.llm = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # If using PyTorch, you can also explicitly clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _handle_stream_response(self, response):
        generated_text = "<svg"
        for chunk in response:
            new_text = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
            generated_text += new_text
        return generated_text
    