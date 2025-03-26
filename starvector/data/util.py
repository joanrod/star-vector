from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, pad
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from svgpathtools import svgstr2paths
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
import numpy as np
import textwrap  
import os
import base64
import io



CIRCLE_SVG = "<svg><circle cx='50%' cy='50%' r='50%' /></svg>"
VOID_SVF = "<svg></svg>"

def load_transforms():
    transforms = {
        'train': None,
        'eval': None
    }
    return transforms

class ImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean=mean, std=std)

class ImageTrainProcessor(ImageBaseProcessor):
    def __init__(self, mean=None, std=None, size=224, **kwargs):
        super().__init__(mean, std)

        self.size = size

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._rgba_to_rgb_white(img) if img.mode == "RGBA" else img),
            transforms.Lambda(lambda img: self._pad_to_square(img)),
            transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, item):
        return self.transform(item)

    def _pad_to_square(self, img):
        # Calculate padding to make the image square
        width, height = img.size
        max_dim = max(width, height)
        padding = [(max_dim - width) // 2, (max_dim - height) // 2]
        padding += [max_dim - width - padding[0], max_dim - height - padding[1]]
        return pad(img, padding, fill=255)  # Assuming white padding

    def _rgba_to_rgb_white(self, img):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background


def encode_image_base64(pil_image):
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')  # Convert RGBA to RGB
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_image
    
# -------------- Generation utils --------------
def is_valid_svg(svg_text):
    try:
        svgstr2paths(svg_text)
        return True
    except Exception as e:
        print(f"Invalid SVG: {str(e)}")
        return False

def clean_svg(svg_text, output_width=None, output_height=None):
    soup = BeautifulSoup(svg_text, 'xml') # Read as soup to parse as xml
    svg_bs4 = soup.prettify() # Prettify to get a string

    # Store the original signal handler
    import signal
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set a timeout to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Try direct conversion without BeautifulSoup
        svg_cairo = cairosvg.svg2svg(svg_bs4, output_width=output_width, output_height=output_height).decode()
        
    except TimeoutError:
        print("SVG conversion timed out, using fallback method")
        svg_cairo = """<svg></svg>"""
    finally:
        # Always cancel the alarm and restore original handler, regardless of success or failure
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        
    svg_clean = "\n".join([line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]) # Remove xml header
    return svg_clean


def use_placeholder():
    return VOID_SVF
 
def process_and_rasterize_svg(svg_string, resolution=256, dpi = 128, scale=2):
    try:
        svgstr2paths(svg_string) # This will raise an exception if the svg is still not valid
        out_svg = svg_string
    except:
        try:
            svg = clean_svg(svg_string)
            svgstr2paths(svg) # This will raise an exception if the svg is still not valid
            out_svg = svg
        except Exception as e:
            out_svg = use_placeholder()

    raster_image = rasterize_svg(out_svg, resolution, dpi, scale)
    return out_svg, raster_image

def rasterize_svg(svg_string, resolution=224, dpi = 128, scale=2):
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution, 
            output_height=resolution,
            dpi=dpi,
            scale=scale) 
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except: 
        try:
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution, 
                output_height=resolution,
                dpi=dpi,
                scale=scale) 
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except:
            svg_raster = Image.new('RGB', (resolution, resolution), color = 'white')
    return svg_raster
    
def find_unclosed_tags(svg_content):
    all_tags_pattern = r"<(\w+)"
    self_closing_pattern = r"<\w+[^>]*\/>"
    all_tags = re.findall(all_tags_pattern, svg_content)
    self_closing_matches = re.findall(self_closing_pattern, svg_content)
    self_closing_tags = []
    
    for match in self_closing_matches:
        tag = re.search(all_tags_pattern, match)
        if tag:
            self_closing_tags.append(tag.group(1))    
    unclosed_tags = []
    
    for tag in all_tags:
        if all_tags.count(tag) > self_closing_tags.count(tag) + svg_content.count('</' + tag + '>'):
            unclosed_tags.append(tag)
    unclosed_tags = list(dict.fromkeys(unclosed_tags))
    
    return unclosed_tags


# -------------- Plotting utils --------------
def plot_images_side_by_side_with_metrics(image1, image2, l2_dist, CD, post_processed, out_path):
    array1 = np.array(image1).astype(np.float32)
    array2 = np.array(image2).astype(np.float32)
    diff = np.abs(array1 - array2).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title('generated_svg')
    axes[0].axis('off')
    axes[1].imshow(image2)
    axes[1].set_title('gt')
    axes[1].axis('off')
    axes[2].imshow(diff)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.suptitle(f"MSE: {l2_dist:.4f}, CD: {CD:.4f}, post-processed: {str(post_processed)}", fontsize=16, y=1.05)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    image = Image.open(out_path)
    plt.close(fig)
    return image

def plot_images_side_by_side(image1, image2, out_path):
    array1 = np.array(image1).astype(np.float32)
    array2 = np.array(image2).astype(np.float32)
    diff = np.abs(array1 - array2).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title('generated_svg')
    axes[0].axis('off')
    axes[1].imshow(image2)
    axes[1].set_title('gt')
    axes[1].axis('off')
    axes[2].imshow(diff)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    image = Image.open(out_path)
    plt.close(fig)
    return image

def plot_images_side_by_side_temperatures(samples_temp, metrics, sample_dir, outpath_filename):
    # Create a plot with the original image and different temperature results
    num_temps = len(samples_temp)
    fig, axes = plt.subplots(2, num_temps + 1, figsize=(15, 4), gridspec_kw={'height_ratios': [10, 2]})
    
    # Plot the original image
    gt_image_path = os.path.join(sample_dir, f'temp_{list(samples_temp.keys())[0]}', f'{outpath_filename}_or.png')
    gt_image = Image.open(gt_image_path)
    axes[0, 0].imshow(gt_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Original', horizontalalignment='center', verticalalignment='center', fontsize=16)
    axes[1, 0].axis('off')
    
    # Plot the generated images for different temperatures and metrics
    for idx, (temp, sample) in enumerate(samples_temp.items()):
        gen_image_path = os.path.join(sample_dir, f'temp_{temp}', f'{outpath_filename}.png')
        gen_image = Image.open(gen_image_path)
        axes[0, idx + 1].imshow(gen_image)
        axes[0, idx + 1].set_title(f'Temp {temp}')
        axes[0, idx + 1].axis('off')
        axes[1, idx + 1].text(0.5, 0.5, f'MSE: {metrics[temp]["mse"]:.2f}\nCD: {metrics[temp]["cd"]:.2f}', 
                            horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1, idx + 1].axis('off')
    
    # Save the comparison plot
    comparison_path = os.path.join(sample_dir, f'{outpath_filename}_comparison.png')
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    
def plot_images_and_prompt(prompt, svg_raster, gt_svg_raster, out_path):
    # First col shows caption, second col shows generated svg, third col shows gt svg
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    # Split the prompt into multiple lines if it exceeds a certain length
    prompt_lines = textwrap.wrap(prompt, width=30)
    prompt_text = '\n'.join(prompt_lines)

    # Display the prompt in the first cell
    axes[0].text(0, 0.5, prompt_text, fontsize=12, ha='left', wrap=True)
    axes[0].axis('off')
    axes[1].imshow(svg_raster)
    axes[1].set_title('generated_svg')
    axes[1].axis('off')
    axes[2].imshow(gt_svg_raster)
    axes[2].set_title('gt')
    axes[2].axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    image = Image.open(out_path)
    plt.close(fig)
    return image
    
def plot_images_and_prompt_with_metrics(prompt, svg_raster, gt_svg_raster, clip_score, post_processed, out_path):
    # First col shows caption, second col shows generated svg, third col shows gt svg
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    # Split the prompt into multiple lines if it exceeds a certain length
    prompt_lines = textwrap.wrap(prompt, width=30)
    prompt_text = '\n'.join(prompt_lines)

    # Display the prompt in the first cell
    axes[0].text(0, 0.5, prompt_text, fontsize=12, ha='left', wrap=True)
    axes[0].axis('off')
    axes[1].imshow(svg_raster)
    axes[1].set_title('generated_svg')
    axes[1].axis('off')
    axes[2].imshow(gt_svg_raster)
    axes[2].set_title('gt')
    axes[2].axis('off')
    plt.suptitle(f"CLIP Score: {clip_score:.4f}, post-processed: {str(post_processed)}", fontsize=16, y=1.05)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    image = Image.open(out_path)
    plt.close(fig)
    return image

def plot_images_and_prompt_temperatures(prompt, samples_temp, metrics, sample_dir, outpath_filename):
    # Calculate the number of temperature variations
    num_temps = len(samples_temp)
    
    # Create a plot with text, the original image, and different temperature results
    fig, axes = plt.subplots(1, num_temps + 2, figsize=(5 + 3 * (num_temps + 1), 6))
    
    # Split the prompt into multiple lines if it exceeds a certain length
    prompt_lines = textwrap.wrap(prompt, width=30)
    prompt_text = '\n'.join(prompt_lines)
    
    # Display the prompt in the first cell
    axes[0].text(0, 0.5, prompt_text, fontsize=12, ha='left', wrap=True)
    axes[0].axis('off')
    
    # Plot the GT (ground truth) image in the second cell
    gt_image_path = os.path.join(sample_dir, f'temp_{list(samples_temp.keys())[0]}', f'{outpath_filename}_or.png')
    gt_image = Image.open(gt_image_path)
    axes[1].imshow(gt_image)
    axes[1].set_title('GT Image')
    axes[1].axis('off')
    
    # Plot the generated images for different temperatures and display metrics
    for idx, (temp, sample) in enumerate(samples_temp.items()):
        gen_image_path = os.path.join(sample_dir, f'temp_{temp}', f'{outpath_filename}.png')
        gen_image = Image.open(gen_image_path)
        axes[idx + 2].imshow(gen_image)
        axes[idx + 2].set_title(f'Temp {temp}')
        axes[idx + 2].axis('off')
        clip_score = metrics[temp]["clip_score"]
        axes[idx + 2].text(0.5, -0.1, f'CLIP: {clip_score:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axes[idx + 2].transAxes)
    
    # Save the comparison plot
    comparison_path = os.path.join(sample_dir, f'{outpath_filename}_comparison.png')
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()

    return comparison_path


def plot_image_tensor(image):
    import numpy as np
    from PIL import Image
    tensor = image[0].cpu().float()
    tensor = tensor.permute(1, 2, 0)
    array = (tensor.numpy() * 255).astype(np.uint8)
    im = Image.fromarray(array)
    im.save("tmp/output_image.jpg")


def plot_grid_samples(images, num_cols=5, out_path = 'grid.png'):
    # Calculate the number of rows required for the grid
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a new figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    
    # Loop through the image files and plot them
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols

        # Open and display the image using Pillow
        if type(image) == str:
            img = Image.open(image)
        else:
            img = image
        axes[row, col].imshow(img)
        # axes[row, col].set_title(os.path.basename(image_file))
        axes[row, col].axis('off')

    # Remove empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    # Adjust spacing between subplots
    plt.tight_layout()

    # save image
    plt.savefig(out_path, dpi=300)
    image = Image.open(out_path)
    plt.close(fig)

    return image