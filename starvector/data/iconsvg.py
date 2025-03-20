import os
from starvector.data.base import SVGDatasetBase
from starvector.data.util import ImageTrainProcessor
from transformers import AutoProcessor

class SVGIconsDataset(SVGDatasetBase):
    def __init__(self, dataset_name, split, im_size, num_samples=-1, **kwargs):
        super().__init__(dataset_name, split, im_size, **kwargs)
   
        self.num_samples = num_samples
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))

        self.image_processor = kwargs.get('image_processor', None)
        if 'siglip' in self.image_processor:
            model_name = {'siglip_512': 'google/siglip-base-patch16-512', 
                          'siglip_384': 'google/siglip-large-patch16-384', 
                          'siglip_256': 'google/siglip-base-patch16-256'}[self.image_processor]
            self.processor = AutoProcessor.from_pretrained(model_name).image_processor
        else:
            self.processor = ImageTrainProcessor(size=self.im_size)

        
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
