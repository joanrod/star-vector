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
from starvector.serve.util import process_images, load_image_from_base64
from threading import Thread
from transformers import TextIteratorStreamer
from openai import OpenAI

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
    def __init__(self, controller_addr, worker_addr, vllm_base_url,
                 worker_id, no_register, model_name, openai_api_key):
        
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name
        self.openai_api_key = openai_api_key

        self.client = OpenAI(   
            api_key=openai_api_key,
            base_url=vllm_base_url,
        )
        
        if "text2svg" in self.model_name.lower():
            self.task = "Text2SVG"
        elif "im2svg" in self.model_name.lower():
            self.task = "Image2SVG"
            
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

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
                    "queue_length": self.get_queue_length()}, timeout=30)
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

    def generate_stream(self, params):
        
        num_beams = int(params.get("num_beams", 1))
        temperature = float(params.get("temperature", 1.0))
        len_penalty = float(params.get("len_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = 1000

        # prompt = params["prompt"]
        prompt = "<svg "
        if self.task == "Image2SVG":
            images = params.get("images", [])
            # Get the first image if available, otherwise None
            image_base_64 = images[0] if images and len(images) > 0 else None

            if not image_base_64:
                yield json.dumps({"text": "Error: No image provided for Image2SVG task", "error_code": 1}).encode() + b"\0"
                return

            max_new_tokens = min(int(params.get("max_new_tokens", 256)), 8192)
            max_new_tokens = min(max_new_tokens, max_context_length - CLIP_QUERY_LENGTH)

            # Use the chat completions endpoint
            vllm_endpoint = f"{self.vllm_base_url}/v1/chat/completions"
            
            # Use a model name that vLLM recognizes
            # The full path including the organization is important
            model_name_for_vllm = params['model']
            
            # Format payload for the chat completions endpoint
            request_payload = {
                "model": model_name_for_vllm,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<image-start>"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base_64}"}}
                        ]
                    }
                ],
                "max_tokens": 7500,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True
            }
            
            # Log the request for debugging
            logger.info(f"Request to vLLM: {vllm_endpoint}")
            logger.info(f"Using model: {model_name_for_vllm}")
            
            # Use requests instead of OpenAI client
            response = requests.post(
                vllm_endpoint, 
                json=request_payload,
                stream=True,
                headers={"Content-Type": "application/json"}
            )
            
            # Log the response status for debugging
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    logger.error(f"Error from vLLM server: {error_detail}")
                except json.JSONDecodeError:
                    logger.error(f"Error from vLLM server: {response.text}")
                
                yield json.dumps({"text": f"Error communicating with model server: {response.status_code}", "error_code": 1}).encode() + b"\0"
                return
            
            # Process the streaming response
            output_text = ""
            for line in response.iter_lines():
                if line:
                    # Skip the "data: " prefix if present
                    if line.startswith(b"data: "):
                        line = line[6:]
                    
                    if line.strip() == b"[DONE]":
                        break
                    
                    try:
                        data = json.loads(line)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                output_text += content
                                yield json.dumps({"text": output_text, "error_code": 0}).encode() + b"\0"
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse line as JSON: {line}")
                        continue
            
            # Send final output if not already sent
            if output_text:
                yield json.dumps({"text": output_text, "error_code": 0}).encode() + b"\0"

        elif self.task == "Text2SVG":
            # Implementation for Text2SVG task would go here
            yield json.dumps({"text": "Text2SVG task not implemented yet", "error_code": 1}).encode() + b"\0"
            return

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
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `starvector` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--openai-api-key", type=str, default="EMPTY")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000")
    

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `starvector` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         args.vllm_base_url,
                         worker_id,
                         args.no_register,
                         args.model_name,
                         args.openai_api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")