import sys
import numpy as np
import torch                               # PyTorch, used here only as the reference implementation
from transformers import AutoModelForCausalLM, AutoTokenizer   # Hugging Face's standard model loader
from mini_llama import load_model, forward # your own code from this stage

model_dir, prompt = sys.argv[1], sys.argv[2]   # model folder and prompt text
tok = AutoTokenizer.from_pretrained(model_dir)                 # reference tokenizer
ref_model = AutoModelForCausalLM.from_pretrained(model_dir).float()   # reference model, converted to float32
ids = tok(prompt)["input_ids"]             # prompt as token ids
with torch.no_grad():                      # we are not training, so skip gradient bookkeeping
    ref = ref_model(torch.tensor([ids])).logits[0, -1].numpy()   # reference scores for the next token, as a numpy array
W, cfg = load_model(model_dir)
mine = forward(W, cfg, ids)                # your implementation's scores for the same tokens
print("max abs logit difference:", float(np.abs(ref - mine).max()))   # largest disagreement between the two
print("same top-5 tokens:", list(np.argsort(-ref)[:5]) == list(np.argsort(-mine)[:5]))   # do both rank the same 5 tokens highest?