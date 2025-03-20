import os
import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from starvector.model.image_encoder.clip_model import convert_weights_to_precision
from starvector.data.util import ImageTrainProcessor

class ImageEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super(ImageEncoder, self).__init__()
        
        image_size = config.image_size
        torch_dtype = kwargs.get('model_precision', config.torch_dtype)
        self.image_encoder_type = config.image_encoder_type
        if self.image_encoder_type == 'clip':
            self.visual_encoder, self.ln_vision = self.build_clip_encoder(image_size=image_size)
            convert_weights_to_precision(self, torch_dtype)
            self.processor = ImageTrainProcessor(size=config.image_size)

        elif self.image_encoder_type == 'vqgan':
            self.visual_encoder = self.build_vqgan_encoder()
            self.ln_vision = None
            self.processor = ImageTrainProcessor(size=config.image_size)

        elif self.image_encoder_type == 'convnext':
            self.visual_encoder = self.build_vqgan_encoder()
            self.ln_vision = None
            self.processor = ImageTrainProcessor(size=config.image_size)

        elif 'siglip' in self.image_encoder_type:
            if self.image_encoder_type == 'siglip_512':
                model_name = "google/siglip-base-patch16-512"
            elif self.image_encoder_type == 'siglip_384':
                model_name = "google/siglip-large-patch16-384"
            elif self.image_encoder_type == 'siglip_256':
                model_name = "google/siglip-base-patch16-256"
                
            from transformers import AutoProcessor, AutoModel

            self.visual_encoder = AutoModel.from_pretrained(
                model_name, torch_dtype = torch_dtype
            ).vision_model

            self.processor = AutoProcessor.from_pretrained(
                model_name, torch_dtype = torch_dtype
            )

    def build_clip_encoder(self, image_size):
        from starvector.model.image_encoder.clip_model import VisionTransformer, LayerNorm
        visual_encoder = VisionTransformer(
            input_resolution=image_size,
            patch_size=14,
            width=1024,
            layers=23,
            heads=16,
            use_grad_checkpointing=False)

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision
    
    def build_vqgan_encoder(self):
        from taming.modules.diffusionmodules.model import Encoder
        VQGAN_CHECKPOINT = "/path/to/vqgan_checkpoint" # You can download the checkpoint from https://github.com/EleutherAI/vqgan-clip/blob/main/README.md
        vqgan_chkp_path =  VQGAN_CHECKPOINT
        files_in_directory = os.listdir(vqgan_chkp_path + '/configs')
        vqgan_config_file = [file for file in files_in_directory if file.endswith('project.yaml')][0]
        vqgan_config = OmegaConf.load(os.path.join(vqgan_chkp_path, 'configs', vqgan_config_file))
        visual_encoder = Encoder(**vqgan_config.model.params.ddconfig)

        # Load checkpoint weights
        checkpoint = torch.load(os.path.join(vqgan_chkp_path, 'checkpoints', 'last.ckpt'))['state_dict']

        # Create a new state_dict with modified keys
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
                new_state_dict[new_key] = value

        # Load weights
        visual_encoder.load_state_dict(new_state_dict)
        return visual_encoder
    
    def build_convnext_encoder(self):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
        return model.visual

    def forward(self, image):
        if self.image_encoder_type == 'clip':
            embeds = self.visual_encoder(image)
            out = self.ln_vision(embeds)
        elif self.image_encoder_type == 'open-clip':
            out = self.visual_encoder(image)[1]
            out = self.ln_vision(out)
        elif self.image_encoder_type == 'vqgan':
            out = self.visual_encoder(image)
            size = out.size()
            out = out.view(size[0], size[1], -1)
            out = out.permute(0, 2, 1)
        elif self.image_encoder_type == 'convnext':
            out = self.visual_encoder.trunk.forward_features(image)
            size = out.size()
            out = out.view(size[0], size[1], -1)
            out = out.permute(0, 2, 1)
        elif 'siglip' in self.image_encoder_type:
            out = self.visual_encoder(image)["last_hidden_state"]
        return out

    def process_images(self, images):
        if self.image_encoder_type == 'clip':
            res = []
            for image in images:
                res.append(self.processor(image).unsqueeze(0)) # B, 3, H, W
            return res
        else:
            return self.processor(images=images, return_tensors="pt").pixel_values.unsqueeze(0)
    