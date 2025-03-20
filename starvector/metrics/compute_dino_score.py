import torch
from torch.utils.data import DataLoader
from starvector.metrics.base_metric import BaseMetric
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch.nn as nn

class DINOScoreCalculator(BaseMetric): 
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model, self.processor = self.get_DINOv2_model("base")
        self.model = self.model.to(device)
        self.device = device

        self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size):
        if model_size == "small":
            model_size = "facebook/dinov2-small"
        elif model_size == "base":
            model_size = "facebook/dinov2-base"
        elif model_size == "large":
            model_size = "facebook/dinov2-large"
        else:
            raise ValueError(f"model_size should be either 'small', 'base' or 'large', got {model_size}")
        return AutoModel.from_pretrained(model_size), AutoImageProcessor.from_pretrained(model_size)

    def process_input(self, image, processor):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
        elif isinstance(image, torch.Tensor):
            features = image.unsqueeze(0) if image.dim() == 1 else image
        else:
            raise ValueError("Input must be a file path, PIL Image, or tensor of features")
        return features

    def calculate_DINOv2_similarity_score(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        features1 = self.process_input(image1, self.processor)
        features2 = self.process_input(image2, self.processor)

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(features1, features2).item()
        sim = (sim + 1) / 2

        return sim
