import os
from starvector.data.base import SVGDatasetBase
from starvector.data.augmentation import SVGTransforms
import random
from transformers import AutoProcessor
from starvector.data.util import ImageTrainProcessor

text2svg_captions = [
    "Draw an SVG of ",
    "Draw an SVG image of ",
    "Draw an SVG picture of ",
    "Generate an SVG of ",
    "Create an SVG of ",
    "Design an SVG of ",
    "Make an SVG of ",
]

class SVGStackDataset(SVGDatasetBase):
    def __init__(self, dataset_name, split, im_size, num_samples=-1, **kwargs):
        super().__init__(dataset_name, split, im_size, num_samples, **kwargs)
        self.color_changer = SVGTransforms({'color_change' : True, 'colors' : ['#ff0000', '#0000ff', '#00ff00', '#ffff00', '#000000']})
        
        # Text2SVG specific
        self.random_caption = kwargs.get('random_caption', True)
        select_dataset_name = kwargs.get('select_dataset_name', False)
        if select_dataset_name:
            self.data = self.data.filter(lambda example: example["model_name"]==select_dataset_name)
        
        self.num_samples = num_samples
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))

        self.image_processor = kwargs.get('image_processor', None)
        if self.image_processor and 'siglip' in self.image_processor:
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
        
        # Randomly choose between 'caption_blip' and 'caption_llava'
        caption_column = random.choice(['caption_blip2', 'caption_llava'])
        caption = random.choice(text2svg_captions) + self.data[idx].get(caption_column, "")
        return {
            'svg': svg,
            'image': image,
            'id': sample_id,
            'caption': caption,
            }
