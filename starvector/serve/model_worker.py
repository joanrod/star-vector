"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial
from starvector.serve.constants import WORKER_HEART_BEAT_INTERVAL, CLIP_QUERY_LENGTH
from starvector.serve.util import (build_logger, server_error_msg,
    pretty_print_semaphore)
from starvector.model.builder import load_pretrained_model
from starvector.serve.util import process_images, load_image_from_base64
from threading import Thread
from transformers import TextIteratorStreamer

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0
model_semaphore = None

def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        if "text2svg" in self.model_name.lower():
            self.task = "Text2SVG"
        elif "im2svg" in self.model_name.lower():
            self.task = "Image2SVG"
            
        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, device=self.device, load_in_8bit=load_8bit, load_in_4bit=load_4bit)
        self.model.to(torch.bfloat16)
        self.is_multimodal = 'starvector' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor, task = self.tokenizer, self.model, self.image_processor, self.task
                 
        num_beams = int(params.get("num_beams", 1))
        temperature = float(params.get("temperature", 1.0))
        len_penalty = float(params.get("len_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 8192)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=15)
        prompt = params["prompt"]
             
        if task == "Image2SVG":
            images = params.get("images", None)
            for b64_image in images: 
                if b64_image is not None and self.is_multimodal:
                    image = load_image_from_base64(b64_image)            
                    image = process_images(image, image_processor)
                    image = image.to(self.model.device, dtype=torch.float16)
                else:
                    image = None
           
            max_new_tokens = min(int(params.get("max_new_tokens", 256)), 8192)
            max_new_tokens = min(max_new_tokens, max_context_length - CLIP_QUERY_LENGTH)
            pre_pend = prompt 
            batch = {}
            batch["image"] = image
            generate_method = model.model.generate_im2svg
        else:
            max_new_tokens = min(int(params.get("max_new_tokens", 128)), 8192)
            pre_pend = ""
            batch = {}
            batch['caption'] = [prompt]
            # White PIL image
            batch['image'] = torch.zeros((3, 256, 256), dtype=torch.float16).to(self.model.device)
            generate_method = model.model.generate_text2svg
            
        if max_new_tokens < 1:
            yield json.dumps({"text": prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=generate_method, kwargs=dict(
            batch=batch,
            prompt=prompt,
            use_nucleus_sampling=True,
            num_beams=num_beams,
            temperature=temperature,
            length_penalty=len_penalty,
            top_p=top_p,
            max_length=max_new_tokens,
            streamer=streamer,
        ))
        thread.start()

        generated_text = pre_pend
        for new_text in streamer:
            if new_text == " ":
                continue
            generated_text += new_text
            # if generated_text.endswith(stop_str):
            #     generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
    
    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

app = FastAPI()

def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()

@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)

@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="joanrodai/starvector-1.4b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `starvector` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `starvector` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")