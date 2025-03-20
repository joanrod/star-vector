from PIL import Image
from vllm import LLM, SamplingParams

model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    max_tokens=7900,
    n=1,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
    top_k=-1,
    min_p=0.0,
)
llm = LLM(model=model_name, trust_remote_code=True, max_model_len=8192)

prompt_start = "<image-start>"
images = [Image.open('assets/examples/sample-18.png')]
model_inputs_vllm = []
for i in range(len(images)):
    model_inputs_vllm.append({
        "prompt": prompt_start,
        "multi_modal_data": {"image": images[i]}
    })
    
outputs = llm.generate(model_inputs_vllm, 
    sampling_params=sampling_params, 
    use_tqdm=False)

completions = []
for i in range(len(outputs)):
    for j in range(len(outputs[i].outputs)):
        completions.append(outputs[i].outputs[j].text)

print(completions)
