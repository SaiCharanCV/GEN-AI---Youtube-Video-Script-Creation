from mlx_lm import load, generate
import time
import mlx_lm
print(mlx_lm.__file__)


model_path = "Phi3_8B_Q6"
model, tokenizer = load(model_path)
prompt = """<|system|>
You are a helpful AI assistant. Follow the user instructions exactly.

<|user|>
Generate a detailed explanation ONLY in the specified language.
- Do not include translations.
- Do not include bullet points unless needed.
- Do not repeat the instructions.
Topic: Stranger Things Web series. Create a Script for Youtube Video. It should be energetic and engaging.
Language: English
Output: Should only content

<|assistant|>
"""


split_trigger = "Generate a detailed explanation ONLY in the specified language"
t = time.time()
output = generate(model, 
                  tokenizer,
                  prompt, 
                  max_tokens=1024)
elapsed = time.time() - t

result = output.split(split_trigger,1)

print(result[0].strip())
print(f"Inference time: {elapsed:.2f} seconds")
