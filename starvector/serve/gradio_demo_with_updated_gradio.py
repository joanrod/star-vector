import argparse
import datetime
import json
import os
import time
import gradio as gr
import requests
from starvector.serve.conversation import default_conversation
from starvector.serve.constants import LOGDIR, CLIP_QUERY_LENGTH
from starvector.serve.util import (build_logger, server_error_msg)

logger = build_logger("gradio_web_server", "gradio_web_server.log")
headers = {"User-Agent": "StarVector Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "starvector-1.4b": "aaaaaaa",
}

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, None, None, None) + (disable_btn,) * 6

def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, None, None) + (disable_btn,) * 6

def send_image(state, image, image_process_mode, request: gr.Request):
    logger.info(f"send_image. ip: {request.client.host}.")
    state.stop_sampling = False
    if image is None:
        state.skip_next = True
        return (state, None, None, image) + (no_change_btn,) * 6

    if image is not None:
        text = (image, image_process_mode)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], "▌")
    state.skip_next = False
    msg = state.to_gradio_svg_code()[0][1]
    return (state, msg, state.to_gradio_svg_render(), image) + (no_change_btn,) * 6

def stop_sampling(state, image, request: gr.Request):
    logger.info(f"stop_sampling. ip: {request.client.host}")
    state.stop_sampling = True
    return (state, None, None, image) + (disable_btn,) * 6

def http_bot(state, model_selector, num_beams, temperature, len_penalty, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, None, None) + (no_change_btn,) * 6
        return

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, None, None, disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, disable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "num_beams": int(num_beams),
        "temperature": float(temperature),
        "len_penalty": float(len_penalty),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 8192-CLIP_QUERY_LENGTH),
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.messages[-1][-1], state.to_gradio_svg_render()) + (disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn)

    try:
        # Stream output
        if state.stop_sampling:
            state.messages[1][-1] = "▌"
            yield (state, state.messages[-1][-1], state.to_gradio_svg_render()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, disable_btn)
            return
        
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=100)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    # output = data["text"].strip().replace('<', '&lt;').replace('>', '&gt;') # trick to avoid the SVG getting rendered
                    output = data["text"].strip()
                    state.messages[-1][-1] = output + "▌"
                    st = state.to_gradio_svg_code()
                    yield (state, st[-1][1], state.to_gradio_svg_render()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, enable_btn)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    
                    yield (state, st[-1][1], state.to_gradio_svg_render()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, disable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, None, None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn, disable_btn)
        return

    yield (state, state.messages[-1][-1], state.to_gradio_svg_render()) + (enable_btn,) * 6

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "svg": state.messages[-1][-1],
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# 💫 StarVector: Generating Scalable Vector Graphics Code from Images and Text
[[Project Page](https://starvector.github.io)] [[Code](https://github.com/joanrod/star-vector)] [[Model](https://huggingface.co/joanrodai/starvector-1.4b)] | 📚 [[StarVector](https://arxiv.org/abs/2312.11556)] 
""")

sub_title_markdown = (""" Throw an image and vectorize it! The model expects vector-like images to generate the corresponding svg code.""")
tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

.gradio-container{
    max-width: 1200px!important
}

#svg_render{
    padding: 20px !important;
}

#svg_code{
    height: 200px !important; 
    overflow: scroll !important; 
    white-space: unset !important; 
    flex-shrink: unset !important;
}


h1{display: flex;align-items: center;justify-content: center;gap: .25em}
*{transition: width 0.5s ease, flex-grow 0.5s ease}
"""

def build_demo(embed_mode, concurrency_count=10):
    with gr.Blocks(title="StarVector", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        if not embed_mode:
            gr.Markdown(title_markdown)
            gr.Markdown(sub_title_markdown)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Resize", "Pad", "Default"],
                    value="Pad",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/sample-4.png"],
                    [f"{cur_dir}/examples/sample-7.png"],
                    [f"{cur_dir}/examples/sample-16.png"],
                    [f"{cur_dir}/examples/sample-17.png"],
                    [f"{cur_dir}/examples/sample-18.png"],
                    [f"{cur_dir}/examples/sample-0.png"],
                    [f"{cur_dir}/examples/sample-1.png"],
                    [f"{cur_dir}/examples/sample-6.png"],
                ], inputs=[imagebox])
                
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary")

                with gr.Accordion("Parameters", open=True) as parameter_row:
                    num_beams = gr.Slider(minimum=1, maximum=10, value=1, step=1, interactive=True, label="Num Beams", visible=False,)
                    temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.8, step=0.05, interactive=True, label="Temperature",)
                    len_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.6, step=0.05, interactive=True, label="Length Penalty",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=8192, value=2000, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                with gr.Row():
                    svg_code = gr.Code(label="SVG Code", elem_id='svg_code', min_width=200, interactive=False, lines=5)
                with gr.Row():
                     gr.Image(width=50, height=256, label="Rendered SVG", elem_id='svg_render')
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False, visible=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False, visible=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)                

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn, stop_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, svg_code, svg_render, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, num_beams, temperature, len_penalty,  top_p, max_output_tokens],
            [state, svg_code, svg_render] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            send_image,
            [state, imagebox, image_process_mode],
            [state, svg_code, svg_render, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, num_beams, temperature, len_penalty, top_p, max_output_tokens],
            [state, svg_code, svg_render] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, svg_code, svg_render] + btn_list,
            queue=False
        )

        stop_btn.click(
            stop_sampling,
            [state, imagebox],
            [state, imagebox] + btn_list,
            queue=False
        ).then(
            clear_history,
            None,
            [state, svg_code, svg_render] + btn_list,
            queue=False
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=15)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )