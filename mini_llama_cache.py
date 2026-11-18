import math, os, sys, time                 # standard library
import numpy as np
from mini_llama import load_model, rmsnorm, silu, softmax, rope_tables, apply_rope, dims   # reuse the Stage 3 building blocks


def step(W, cfg, ids, cache, start):
    """Run only the NEW token ids (positions start, start+1, ...). cache[i] holds (K, V) of layer i."""
    H, nh, nkv, hd, eps, theta = dims(cfg)
    T = len(ids)                           # how many NEW tokens this call processes
    x = W["model.embed_tokens.weight"][ids]   # embedding rows for the new tokens
    cos, sin = rope_tables(np.arange(start, start + T), hd, theta)   # angles for their positions
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."           # name prefix of this layer's tensors
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)   # normalize before attention
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)    # queries for the new tokens only
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)  # new keys
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)  # new values
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # rotate by position
        if cache[i] is not None:                                   # append to what earlier steps stored
            k = np.concatenate([cache[i][0], k], axis=1)           # new keys go after the old ones
            v = np.concatenate([cache[i][1], v], axis=1)           # same for values
        cache[i] = (k, v)                                          # [nkv, S, hd] save the full history for the next step
        S = k.shape[1]                                             # total tokens so far, old plus new
        kr = np.repeat(k, nh // nkv, axis=0)                       # copy each KV head for its group of query heads
        vr = np.repeat(v, nh // nkv, axis=0)
        scores = q @ kr.transpose(0, 2, 1) / math.sqrt(hd)         # [nh, T, S] new tokens look at every token so far
        scores = scores + np.triu(np.full((T, S), -np.inf, dtype=np.float32), k=start + 1)   # block looking at future positions
        a = (softmax(scores) @ vr).transpose(1, 0, 2).reshape(T, nh * hd)   # weighted mix of values, heads glued together
        x = x + a @ W[p + "self_attn.o_proj.weight"].T             # output projection plus residual
        h = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)   # normalize before the MLP
        x = x + (silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T   # MLP: gate x up, project down, residual
    x = rmsnorm(x[-1], W["model.norm.weight"], eps)                # final norm on the last position
    head = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]   # reuse the embedding if no separate head
    return x @ head.T                                              # one score per possible next token


def sample(logits, temperature, top_k, rng):   # choose the next token id from the scores
    if temperature == 0:
        return int(logits.argmax())                                # greedy: always the top scorer
    idx = np.argpartition(logits, -top_k)[-top_k:]                 # ids of the top_k highest scores
    p = softmax((logits[idx] / temperature).astype(np.float64))    # scores to probabilities; low temperature sharpens, high flattens
    return int(rng.choice(idx, p=p / p.sum()))                     # draw one id at random using those probabilities


def generate(W, cfg, ids, n_new, temperature=0.0, top_k=40, seed=0):   # parameters after ids have default values
    rng = np.random.default_rng(seed)      # random number generator with a fixed seed: repeatable results
    cache = [None] * cfg["num_hidden_layers"]   # one empty cache slot per layer
    out = list(ids)                        # copy of the prompt ids; generated ids get appended here
    t0 = time.time()
    logits = step(W, cfg, out, cache, 0)   # prefill: the whole prompt at once
    t_prefill = time.time() - t0
    t0 = time.time()
    for _ in range(n_new):                 # decode loop: one new token per pass
        nxt = sample(logits, temperature, top_k, rng)   # pick the next token
        out.append(nxt)
        logits = step(W, cfg, [nxt], cache, len(out) - 1)   # decode: process only that one token, at its position
    t_decode = time.time() - t0
    kv_bytes = sum(k.nbytes + v.nbytes for k, v in cache)   # total memory held by the cache
    return out, t_prefill, t_decode, kv_bytes   # a function can return several values


if __name__ == "__main__":                 # run this part only when the file is executed directly
    from tokenizers import Tokenizer
    model_dir, prompt = sys.argv[1], sys.argv[2]   # model folder and prompt text
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 30   # tokens to generate
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # 0 means greedy
    W, cfg = load_model(model_dir)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(prompt).ids
    out, tp, td, kv = generate(W, cfg, ids, n_new, temperature)   # unpack the four returned values
    print(tok.decode(out))
    print(f"prefill: {len(ids)} tokens in {tp:.2f}s ({len(ids) / tp:.1f} tok/s)")
    print(f"decode : {n_new} tokens in {td:.2f}s ({n_new / td:.2f} tok/s)")
    print(f"KV cache: {kv:,} bytes for {len(out)} tokens ({kv // len(out):,} bytes per token)")   # // is integer division