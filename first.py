!pip install vllm

from vllm import LLM, SamplingParams

# can get in more than one prompt samples
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# The SamplingParams class specifies the parameters for the sampling process when the LLM generate the next token
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize vLLM’s engine for offline inference with the LLM class 
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct")

# we also get a batch of outputs
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
