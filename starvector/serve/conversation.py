import dataclasses
from typing import List
from PIL import Image
import concurrent.futures
from bs4 import BeautifulSoup
import cairosvg
from io import BytesIO

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    image_prompt: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    version: str = "Unknown"
    stop_sampling: bool = False
    skip_next: bool = False
    display_images: bool = False
    task: str = "Im2SVG"
    
    def set_task(self, task):
        self.task = task

    def get_image_prompt(self):
        return self.image_prompt

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(255, 255, 255)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images
        
    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def download_files(self):
        svg_string = self.messages[-1][-1][:-1]
        image = self.render_svg(svg_string)
        svg_out = clean_svg(svg_string)
        
        return image, svg_out   

    def rasterize_svg(self, svg_string, resolution=224, dpi = 128, scale=2):
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
                svg = self.clean_svg(svg_string)
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

    def clean_svg(self, svg_text, output_width=None, output_height=None):
        soup = BeautifulSoup(svg_text, 'xml') # Read as soup to parse as xml
        svg_bs4 = soup.prettify() # Prettify to get a string
        svg_cairo = cairosvg.svg2svg(svg_bs4, output_width=output_width, output_height=output_height).decode()
        svg_clean = "\n".join([line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]) # Remove xml header
        return svg_clean

    def render_svg(self, svg_string):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.rasterize_svg, svg_string, resolution = 512)
            try:
                result = future.result(timeout=0.1)  # Specify the timeout duration in seconds
            except concurrent.futures.TimeoutError:
                print("Timeout occurred!")
                result = None
            return result
    
    def to_gradio_svg_render(self):
        svg_string = self.messages[-1][-1][:-1]
        result = self.render_svg(svg_string)
        return result

    def to_gradio_svg_code(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret
    
    def copy(self):
        return Conversation(
            system=self.system,
            image_prompt=self.image_prompt,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            version=self.version
            
            )
    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "image_prompt": self.image_prompt,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
            }
        return {
            "system": self.system,
            "image_prompt": self.image_prompt,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }

starvector_v1 = Conversation(
    system="StarVector",
    # prompt='<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 32 32" version="1.1">',
    image_prompt='<svg',
    roles=("Human", "StarVector"),
    version="v1",
    messages=(
    ),
    offset=0,
    task="Im2SVG",
)
default_conversation = starvector_v1
conv_templates = {
    "default": default_conversation,
}

if __name__ == "__main__":
    print(default_conversation.get_image_prompt())
