import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# device = ("cuda" if torch.cuda.is_available() else "mps")
# torch.mps.set_per_process_memory_fraction(0.7)
device = ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "nlpai-lab/KULLM3"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

s = "네이버에 대해서 알고 있니?"
conversation = [{'role': 'user', 'content': s}]
inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt').to(device)
_ = model.generate(inputs, streamer=streamer, max_new_tokens=1024, use_cache=True)