import os
from starvector.data.base import SVGDatasetBase
from transformers import AutoProcessor
from starvector.data.util import ImageTrainProcessor

class FontSVGDataset(SVGDatasetBase):
    def __init__(self, dataset_name, split, im_size, num_samples=-1, **kwargs):
        super().__init__(dataset_name, split, im_size, **kwargs)

        self.num_samples = num_samples
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        
        svg_str = self.data[idx]['Svg']
        sample_id = self.data[idx]['Filename']
        svg, image = self.get_svg_and_image(svg_str, sample_id)
        caption = self.data[idx].get('Caption', "")
        return {
            'svg': svg,
            'image': image,
            'id': sample_id,
            'caption': caption
            }
