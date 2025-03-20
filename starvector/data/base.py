from torch.utils.data import Dataset
from starvector.data.util import ImageTrainProcessor, use_placeholder, rasterize_svg
from starvector.util import instantiate_from_config
import numpy as np
from datasets import load_dataset

class SVGDatasetBase(Dataset):
    def __init__(self, dataset_name, split, im_size, num_samples=-1, **kwargs):
        self.split = split
        self.im_size = im_size

        transforms = kwargs.get('transforms', False)
        if transforms:
            self.transforms = instantiate_from_config(transforms)
            self.p = self.transforms.p
        else:
            self.transforms = None
            self.p = 0.0

        normalization = kwargs.get('normalize', False)
        if normalization:
            mean = tuple(normalization.get('mean', None))
            std = tuple(normalization.get('std', None))
        else:
            mean = None
            std = None

        self.processor = ImageTrainProcessor(size=self.im_size, mean=mean, std=std)
        self.data = load_dataset(dataset_name, split=split)

        print(f"Loaded {len(self.data)} samples from {dataset_name} {split} split")

    def __len__(self):
        return len(self.data_json)
    
    def get_svg_and_image(self, svg_str, sample_id):
        do_augment = np.random.choice([True, False], p=[self.p, 1 - self.p])
        svg, image = None, None

        # Try to augment the image if conditions are met
        if self.transforms is not None and do_augment:
            try:
                svg, image = self.transforms.augment(svg_str)
            except Exception as e:
                print(f"Error augmenting {sample_id} due to {str(e)}, trying to rasterize SVG")

        # If augmentation failed or wasn't attempted, try to rasterize the SVG
        if svg is None or image is None:
            try:
                svg, image = svg_str, rasterize_svg(svg_str, self.im_size)
            except Exception as e:
                print(f"Error rasterizing {sample_id} due to {str(e)}, using placeholder image")
                svg = use_placeholder()
                image = rasterize_svg(svg, self.im_size)

        # If the image is completely white, use a placeholder image
        if np.array(image).mean() == 255.0:
            print(f"Image is full white, using placeholder image for {sample_id}")
            svg = use_placeholder()
            image = rasterize_svg(svg)

        # Process the image
        if 'siglip' in self.image_processor:
            image = self.processor(image).pixel_values[0]
        else:
            image = self.processor(image)

        return svg, image

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented by subclasses")
