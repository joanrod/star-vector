from torchvision.transforms import ToTensor
import torch.nn.functional as F
from starvector.metrics.base_metric import BaseMetric
import torch

class L2DistanceCalculator(BaseMetric): 
    def __init__(self, config=None, masked_l2=False):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.metric = self.l2_distance
        self.masked_l2 = masked_l2

    def l2_distance(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('gen_im')
        image1_tensor = ToTensor()(image1)
        image2_tensor = ToTensor()(image2)

        if self.masked_l2:
            # Create binary masks: 0 for white pixels, 1 for non-white pixels
            mask1 = (image1_tensor != 1).any(dim=0).float()
            mask2 = (image2_tensor != 1).any(dim=0).float()

            # Create a combined mask for overlapping non-white pixels
            combined_mask = mask1 * mask2

            # Apply the combined mask to both images
            image1_tensor = image1_tensor * combined_mask.unsqueeze(0)
            image2_tensor = image2_tensor * combined_mask.unsqueeze(0)

        # Compute mean squared error
        mse = F.mse_loss(image1_tensor, image2_tensor)
        return mse.item()



