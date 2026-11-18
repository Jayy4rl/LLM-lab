import os, sys
import numpy as np
from mini_llama import load_model, forward, softmax   # reuse your Stage 3 code


def quant_int8(w):
    """One scale per output row; values stored as int8 in [-127, 127]."""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0   # per row: the largest absolute weight, divided by 127
    scale[scale == 0] = 1.0                # avoid dividing by zero for all-zero rows
    q = np.round(w / scale).astype(np.int8)   # divide, round to the nearest integer, store as 8-bit
    return q, scale.astype(np.float32)     # the integers plus the scales needed to undo it


def dequant_int8(q, scale):                # undo the quantization (approximately)
    return q.astype(np.float32) * scale    # integers times scale give back approximate weights


def quant_4bit(w, block=32):
    """One scale per block of 32 weights; values in [-8, 7] (4 bits). Kept in int8 containers here."""
    rows, cols = w.shape                   # matrix dimensions
    wb = w.reshape(rows, cols // block, block)   # view each row as blocks of 32 numbers
    scale = np.abs(wb).max(axis=2, keepdims=True) / 7.0   # per block: largest absolute weight, divided by 7
    scale[scale == 0] = 1.0
    q = np.clip(np.round(wb / scale), -8, 7).astype(np.int8)   # divide, round, force into the 4-bit range -8..7
    return q, scale.astype(np.float16)     # scales stored as 16-bit floats


def dequant_4bit(q, scale, shape):
    return (q.astype(np.float32) * scale.astype(np.float32)).reshape(shape)   # multiply back, restore the original shape


def kl(p_logits, q_logits):                # how different two next-token distributions are (0 = identical)
    p, q = softmax(p_logits), softmax(q_logits)   # scores to probabilities
    return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())   # KL divergence; 1e-12 avoids log(0)


def quantize_model(W, mode):               # return a copy of the weights after a quantize/dequantize round trip
    W2, bits, total_w = dict(W), 0, 0      # copy of the dictionary, bits stored so far, weights counted so far
    for name, w in W.items():
        if not name.endswith("_proj.weight"):   # only the seven projection matrices per layer
            continue                       # skip everything else
        total_w += w.size
        if mode == "int8":
            q, s = quant_int8(w)
            W2[name] = dequant_int8(q, s)  # store the degraded weights so we can measure the damage
            bits += w.size * 8 + s.size * 32   # 8 bits per weight plus 32 bits per scale
        elif mode == "q4":
            q, s = quant_4bit(w)
            W2[name] = dequant_4bit(q, s, w.shape)
            bits += w.size * 4 + s.size * 16   # 4 bits per weight plus 16 bits per scale
    return W2, bits / 8, total_w           # modified weights, total bytes, weight count


if __name__ == "__main__":                 # run this part only when the file is executed directly
    from tokenizers import Tokenizer
    model_dir = sys.argv[1]
    W, cfg = load_model(model_dir)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    prompts = ["The capital of France is", "Water boils at a temperature of", "def add(a, b):\n    return"]   # three test prompts
    ref = [forward(W, cfg, tok.encode(p).ids) for p in prompts]   # original model's next-token scores for each
    print(f"{'mode':6} {'projection MB':>14} {'bits/weight':>12} {'top-1 kept':>11} {'mean KL':>10}")   # table header
    for mode in ["fp32", "int8", "q4"]:
        if mode == "fp32":
            W2, nbytes, nbits = W, sum(w.size * 4 for n, w in W.items() if n.endswith("_proj.weight")), 32.0   # baseline: 4 bytes per weight
        else:
            W2, nbytes, total_w = quantize_model(W, mode)
            nbits = nbytes * 8 / total_w   # average bits per weight, scales included
        outs = [forward(W2, cfg, tok.encode(p).ids) for p in prompts]   # scores from the degraded model
        same = sum(int(a.argmax() == b.argmax()) for a, b in zip(ref, outs))   # prompts where the top prediction is unchanged
        mkl = float(np.mean([kl(a, b) for a, b in zip(ref, outs)]))   # average distribution difference
        print(f"{mode:6} {nbytes / 1e6:14.1f} {nbits:12.2f} {same:>9}/{len(prompts)} {mkl:10.5f}")