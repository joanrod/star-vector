from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

starvector = StarVectorForCausalLM.from_pretrained(model_name, torch_dtype="auto") # add , torch_dtype="bfloat16"

starvector.cuda()
starvector.eval()

image_pil = Image.open("assets/examples/sample-18.png")
image_pil = image_pil.convert('RGB')
image = starvector.process_images([image_pil])[0].to(torch.float16).cuda()
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=4000, temperature=1.5, length_penalty=-1, repetition_penalty=3.1)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)
