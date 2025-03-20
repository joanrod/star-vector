from torchvision.transforms import ToTensor, Normalize
import torch
from torch.utils.data import DataLoader
from starvector.metrics.base_metric import BaseMetric
import lpips
from tqdm import tqdm


class LPIPSDistanceCalculator(BaseMetric): 
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.model = lpips.LPIPS(net='vgg').to(device)
        self.metric = self.LPIPS
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = device

    def LPIPS(self, tensor_image1, tensor_image2):
        tensor_image1, tensor_image2 = tensor_image1.to(self.device), tensor_image2.to(self.device)
        return self.model(tensor_image1, tensor_image2)
    
    def to_tensor_transform(self, pil_img):
        return self.normalize(self.to_tensor(pil_img))

    def collate_fn(self, batch):
        gt_imgs, gen_imgs = zip(*batch)
        tensor_gt_imgs = torch.stack([self.to_tensor_transform(img) for img in gt_imgs])
        tensor_gen_imgs = torch.stack([self.to_tensor_transform(img) for img in gen_imgs])
        return tensor_gt_imgs, tensor_gen_imgs

    def calculate_score(self, batch, batch_size=8, update=True):
        gt_images = batch['gt_im']
        gen_images = batch['gen_im']

        # Create DataLoader with custom collate function
        data_loader = DataLoader(list(zip(gt_images, gen_images)), batch_size=batch_size, collate_fn=self.collate_fn, shuffle=False)
        
        values = []
        for tensor_gt_batch, tensor_gen_batch in tqdm(data_loader):
            # Compute LPIPS
            lpips_values = self.LPIPS(tensor_gt_batch, tensor_gen_batch)
            values.extend([lpips_values.squeeze().cpu().detach().tolist()] if lpips_values.numel() == 1 else lpips_values.squeeze().cpu().detach().tolist())

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        avg_score = sum(values) / len(values)
        if update:
            self.meter.update(avg_score, len(values))
            return self.meter.avg, values
        else:
            return avg_score, values
    