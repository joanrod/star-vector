import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from starvector.model.adapters.adapter import Adapter
from starvector.model.image_encoder.image_encoder import ImageEncoder
from starvector.util import print_trainable_parameters
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[]):
        super().__init__()  # Correct super() call
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Check if any of the stop sequences are in the input_ids
        for stop_ids in self.stops:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False

class StarVectorBase(nn.Module, ABC):
    def __init__(self, config, **kwargs):
        super().__init__()
        # Task-specific layers
        self.task = kwargs.get('task', 'im2svg')
        self.model_precision = kwargs.get('model_precision', config.torch_dtype)
        # Build Code LLM (StarCoder)
        self.svg_transformer = self._get_svg_transformer(config, **kwargs)
        
        if self.use_image_encoder():
            # Build Image Encoder
            self.image_encoder = ImageEncoder(config, **kwargs)
            
            # Build Adapter
            self.image_projection = self.get_adapter(config, **kwargs).to(dtype=self.model_precision)
        else:
            self.query_length = 0
            
        self.max_length = config.max_length_train - self.query_length - 4  # for added special tokens
        
        self.train_image_encoder = kwargs.get('train_image_encoder', False)
        self.train_LLM = kwargs.get('train_LLM', False)
        self.train_connector = kwargs.get('train_connector', False)

        # Freeze parameters
        self.freze_parameters(self.train_image_encoder, self.train_LLM, self.train_connector)
        print_trainable_parameters(self)

    @abstractmethod
    def _get_svg_transformer(self, config, **kwargs):
        """Get SVG transformer model - implementation differs between versions"""
        pass

    def freze_parameters(self, train_image_encoder, train_LLM, train_connector):
        """V2 implementation of parameter freezing"""
        if self.use_image_encoder():
            for _, param in self.image_encoder.named_parameters():
                param.requires_grad = train_image_encoder
                            
            # adapter trainable
            for _, param in self.image_projection.named_parameters():
                param.requires_grad = train_connector
                
        for _, param in self.svg_transformer.named_parameters():
            param.requires_grad = train_LLM

    def use_image_encoder(self):
        """Determine if image encoder should be used based on task"""
        return self.task == 'im2svg'

    def get_adapter(self, config, **kwargs):
        """Get adapter layer for image projection"""
        vision_hidden_size, self.query_length = self.get_hidden_size_and_query_length(config.image_encoder_type)
        llm_hidden_size = self.svg_transformer.transformer.config.hidden_size
        image_projection = Adapter(
            vision_hidden_size,
            llm_hidden_size,
            adapter_norm=config.adapter_norm,
            query_length=self.query_length,
            dropout_prob=kwargs.get('dropout', 0.1)
        )
        return image_projection

    def get_hidden_size_and_query_length(self, image_encoder_type):
        """Get hidden size and query length based on encoder type"""
        if image_encoder_type == 'clip':
            hidden_size = self.image_encoder.visual_encoder.num_features
            query_length = 257
        elif image_encoder_type == 'open-clip':
            hidden_size = self.image_encoder.visual_encoder.transformer.width
            query_length = 256
        elif image_encoder_type == 'vqgan':
            hidden_size = 256
            query_length = 196
        elif image_encoder_type == 'convnext':
            hidden_size = 1024
            query_length = 49
        elif 'siglip' in image_encoder_type:
            hidden_size = self.image_encoder.visual_encoder.head.mlp.fc2.out_features
            if '512' in image_encoder_type:
                query_length = 1024
            elif '384' in image_encoder_type:
                query_length = 576
            
        return hidden_size, query_length

    def _tokenize(self, text, max_length, device, add_special_tokens=True):
        """Common tokenization logic"""
        tokens = self.svg_transformer.tokenizer(
            text,
            truncation=True,
            add_special_tokens=add_special_tokens,
            padding='longest',
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        return tokens

    def _create_targets(self, tokens):
        """Create targets with padding mask"""
        target_mask = (tokens.input_ids == self.svg_transformer.tokenizer.pad_token_id)
        return tokens.input_ids.masked_fill(target_mask, -100)

    @abstractmethod
    def _get_embeddings(self, input_ids):
        """Get embeddings from input ids - implementation differs between v1 and v2"""
        pass

    def embed_text_to_svg(self, batch, device):
        """Common text to SVG embedding logic"""
        captions = batch["caption"]
        svgs = batch["svg"]
        samples = [captions[i] + self.svg_transformer.svg_start_token + svgs[i] + self.svg_transformer.tokenizer.eos_token 
                  for i in range(len(captions))]

        tokens = self._tokenize(samples, self.max_length, device)
        targets = self._create_targets(tokens)
        inputs_embeds = self._get_embeddings(tokens.input_ids)
        
        return inputs_embeds, tokens.attention_mask, targets

    def get_image_embeddings(self, batch, device):
        """Get image embeddings"""
        image = batch["image"].to(dtype=self.model_precision)
        embedded_image = self.image_encoder(image)
        conditioning_embeds = self.image_projection(embedded_image)
        return conditioning_embeds
    
    def embed_im_to_svg(self, batch, device):
        """Common image to SVG embedding logic"""
        # Process image
        image = batch["image"].to(dtype=self.model_precision)
        embedded_image = self.image_encoder(image)
        conditioning_embeds = self.image_projection(embedded_image)
        conditioning_embeds_att = torch.ones(conditioning_embeds.size()[:-1], dtype=torch.long).to(device)

        # Get SVG text with appropriate end tokens (implemented by subclasses)
        svg_text = self._get_svg_text(batch["svg"])
        
        svg_tokens = self._tokenize(svg_text, self.max_length, device)
        svg_tokens_embeds = self._get_embeddings(svg_tokens.input_ids)

        inputs_embeds = torch.cat([conditioning_embeds, svg_tokens_embeds], dim=1)
        
        svg_targets = self._create_targets(svg_tokens)
        empty_targets = torch.ones(conditioning_embeds_att.size(), dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, svg_targets], dim=1)

        attention_mask = torch.cat([conditioning_embeds_att, svg_tokens.attention_mask], dim=1)
        
        return inputs_embeds, attention_mask, targets

    def forward(self, batch):
        """Forward pass"""
        device = batch["image"].device
        task = self.task

        # Depending 
        if task == 'text2svg':
            inputs_embeds, attention_mask, targets = self.embed_text_to_svg(batch, device)
        elif task == 'im2svg':
            inputs_embeds, attention_mask, targets = self.embed_im_to_svg(batch, device)
        
        outputs = self.svg_transformer.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        loss = outputs.loss 
        return loss

    
    @abstractmethod
    def _get_svg_text(self, svg_list):
        """Get SVG text with appropriate end tokens - implementation differs between v1 and v2"""
        pass
    
    
    def _prepare_generation_inputs(self, batch, prompt, device):
        """Common preparation for generation inputs"""
        image = batch["image"]
        image = image.to(device).to(self.model_precision)
        
        embedded_image = self.image_encoder(image)
        embedded_image = self.image_projection(embedded_image)
        embedded_att = torch.ones(embedded_image.size()[:-1], dtype=torch.long).to(device)
        
        if prompt is None:
            prompt = self.svg_transformer.prompt
        prompt = [prompt] * image.size(0)   
        
        prompt_tokens = self._tokenize(prompt, None, device, add_special_tokens=False)
        attention_mask = torch.cat([embedded_att, prompt_tokens.attention_mask], dim=1)    
        inputs_embeds = self._get_embeddings(prompt_tokens.input_ids)
        inputs_embeds = torch.cat([embedded_image, inputs_embeds], dim=1)
        
        return inputs_embeds, attention_mask, prompt_tokens

    def _get_generation_kwargs(self, base_kwargs):
        """Common generation kwargs preparation"""
        # Get token IDs for "</svg>"
        end_sequence = self.svg_transformer.tokenizer("</svg>", add_special_tokens=False)['input_ids']
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[end_sequence])])
        return {
            'inputs_embeds': base_kwargs['inputs_embeds'],
            'attention_mask': base_kwargs['attention_mask'],
            'do_sample': base_kwargs.get('use_nucleus_sampling', True),
            'top_p': base_kwargs.get('top_p', 0.9),
            'temperature': base_kwargs.get('temperature', 1),
            'num_beams': base_kwargs.get('num_beams', 2),
            'max_length': base_kwargs.get('max_length', 30),
            'min_length': base_kwargs.get('min_length', 1),
            'repetition_penalty': base_kwargs.get('repetition_penalty', 1.0),
            'length_penalty': base_kwargs.get('length_penalty', 1.0),
            'use_cache': base_kwargs.get('use_cache', True),
            'stopping_criteria': stopping_criteria
        }
    
    def generate_im2svg(self, batch, **kwargs):
        """Base implementation of image to SVG generation"""
        inputs_embeds, attention_mask, prompt_tokens = self._prepare_generation_inputs(
            batch, kwargs.get('prompt'), batch["image"].device
        )
        
        generation_kwargs = self._get_generation_kwargs(
            {**kwargs, 'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask}
        )
        # Let subclasses override these defaults if needed
        generation_kwargs.update(self._get_im2svg_specific_kwargs(kwargs))
        
        outputs = self.svg_transformer.transformer.generate(**generation_kwargs)
        outputs = torch.cat([prompt_tokens.input_ids, outputs], dim=1)
        raw_svg = self.svg_transformer.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return raw_svg

    def generate_im2svg_grpo(self, batch, **kwargs):
        """Base implementation of image to SVG generation"""
        inputs_embeds, attention_mask, prompt_tokens = self._prepare_generation_inputs(
            batch, kwargs.get('prompt'), batch["image"].device
        )
        
        generation_kwargs = self._get_generation_kwargs(
            {**kwargs, 'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask}
        )
        # Let subclasses override these defaults if needed
        generation_kwargs.update(self._get_im2svg_specific_kwargs(kwargs))

        num_return_sequences = kwargs.get('num_return_sequences', 1)
        if num_return_sequences > 1:
            generation_kwargs['num_return_sequences'] = num_return_sequences
            generation_kwargs['num_beams'] = 1

        outputs = self.svg_transformer.transformer.generate(**generation_kwargs)
        outputs = torch.cat([prompt_tokens.input_ids.repeat(num_return_sequences, 1), outputs], dim=1)
        raw_svg = self.svg_transformer.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "raw_svg": raw_svg,
            "outputs": outputs,
            "inputs_embeds": inputs_embeds,
        }
        
    
    def _get_im2svg_specific_kwargs(self, kwargs):
        """Default implementation of im2svg specific generation kwargs.
        Subclasses can override this to customize generation behavior."""
        return {
            'early_stopping': True,
            'pad_token_id': self.svg_transformer.tokenizer.pad_token_id
        }

    def generate_text2svg(self, batch, **kwargs):
        """Base implementation of text to SVG generation"""
        device = batch["image"].device
        prompt = batch["caption"]
        
        prompt_tokens = self._tokenize(
            prompt,
            max_length=kwargs.get('max_length', 30),
            device=device,
            add_special_tokens=False
        )
        
        trigger_token = self._tokenize(
            [self.svg_transformer.svg_start_token for _ in batch["caption"]],
            max_length=None,
            device=device,
            add_special_tokens=False
        )
        
        input_tokens = torch.cat([prompt_tokens.input_ids, trigger_token.input_ids], dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, trigger_token.attention_mask], dim=1)
        inputs_embeds = self._get_embeddings(input_tokens)
        max_length = kwargs.get('max_length', 30) - input_tokens.size(1)

        generation_kwargs = self._get_generation_kwargs(
            {**kwargs, 'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask},
            input_tokens.size(1)
        )
        # Let subclasses override these defaults if needed
        generation_kwargs.update(self._get_text2svg_specific_kwargs(kwargs))
        generation_kwargs['max_length'] = max_length
        
        outputs = self.svg_transformer.transformer.generate(**generation_kwargs)
        return outputs

    def _get_text2svg_specific_kwargs(self, kwargs):
        """Default implementation of text2svg specific generation kwargs.
        Subclasses can override this to customize generation behavior."""
        return {
            'eos_token_id': self.svg_transformer.tokenizer.eos_token_id,
            'early_stopping': True,
            'length_penalty': kwargs.get('length_penalty', 1.0)
        }
