from torchvision.transforms import ToTensor
import torch.nn.functional as F
from starvector.metrics.base_metric import BaseMetric
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.functional.multimodal.clip_score import _clip_score_update

class CLIPScoreCalculator(BaseMetric): 
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        self.clip_score.to('cuda')

    def CLIP_Score(self, images, captions):
        all_scores = _clip_score_update(images, captions, self.clip_score.model, self.clip_score.processor)
        return all_scores

    def collate_fn(self, batch):
        gen_imgs, captions = zip(*batch)
        tensor_gen_imgs = [transforms.ToTensor()(img) for img in gen_imgs]
        return tensor_gen_imgs, captions

    def calculate_score(self, batch, batch_size=512, update=True):
        gen_images = batch['gen_im']
        captions = batch['caption']

        # Create DataLoader with custom collate function
        data_loader = DataLoader(list(zip(gen_images, captions)), collate_fn=self.collate_fn, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_scores = []
        for batch_eval in tqdm(data_loader):
            images, captions = batch_eval
            images = [img.to('cuda', non_blocking=True) * 255 for img in images]
            list_scores = self.CLIP_Score(images, captions)[0].detach().cpu().tolist()
            all_scores.extend(list_scores)

        if not all_scores:
            print("No valid scores found for metric calculation.")
            return float("nan"), []

        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    # Rest of your code...