import json, math, os, struct, sys, time   # standard library: JSON, math, paths, binary decoding, args, timing
import numpy as np                         # numeric library, used under the short name np


def load_model(model_dir):                 # read every weight into a dictionary: tensor name -> numpy array
    path = os.path.join(model_dir, "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]   # header length (see Stage 1, Part A)
        header = json.loads(f.read(n))     # tensor names, dtypes, shapes, byte ranges
    header.pop("__metadata__", None)       # drop an optional extra key
    base = 8 + n                           # where tensor data starts in the file
    mm = np.memmap(path, dtype=np.uint8, mode="r")   # the whole file as bytes; the OS pages the file in on demand
    W = {}                                 # the result: name -> array
    for name, t in header.items():         # for every tensor in the file
        s, e = t["data_offsets"]           # its start and end byte
        assert t["dtype"] == "BF16", (name, t["dtype"])   # stop if it is not bfloat16
        raw = np.asarray(mm[base + s: base + e]).view(np.uint16)   # slice out its bytes, read them as 16-bit integers
        # bfloat16 is the top 16 bits of a float32: shift left 16 and reinterpret
        W[name] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(t["shape"])   # widen, shift, reread as float32, restore the 2D shape
    cfg = json.load(open(os.path.join(model_dir, "config.json")))   # the model settings
    return W, cfg                          # return both


def rmsnorm(x, w, eps):                    # normalize each row of x to a steady size
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w   # divide by the row's root-mean-square, then scale by learned weights w


def silu(x):                               # smooth on/off activation function
    return x / (1.0 + np.exp(-x))          # x times sigmoid(x)


def softmax(x):                            # turn scores into probabilities that sum to 1
    x = x - x.max(-1, keepdims=True)       # subtract the row maximum so exp() cannot overflow
    e = np.exp(x)                          # exponentiate every score
    return e / e.sum(-1, keepdims=True)    # divide by the row total


def rope_tables(positions, head_dim, theta):   # precompute the rotation angles RoPE uses for each position
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2) / head_dim)    # [hd/2] rotation speeds, one per pair of numbers
    ang = np.outer(positions, inv)                                 # [T, hd/2] angle = position x speed
    ang = np.concatenate([ang, ang], axis=-1)                      # [T, hd] repeat so it covers the full head size
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)   # cos and sin tables, as float32


def apply_rope(x, cos, sin):                                       # x: [heads, T, hd]
    h = x.shape[-1] // 2                                           # half the vector length
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)       # swap the two halves, negating one: the 90-degree partner of x
    return x * cos + rot * sin                                     # rotate every pair of numbers by its position's angle


def dims(cfg):                             # pull the size settings out of the config
    H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]   # hidden width, query heads, key/value heads
    hd = cfg.get("head_dim") or H // nh    # size of one head
    return H, nh, nkv, hd, cfg["rms_norm_eps"], cfg.get("rope_theta", 10000.0)   # also the norm epsilon and RoPE base


def forward(W, cfg, ids):
    """Full forward pass over all ids with no cache. Returns logits for the last position."""
    H, nh, nkv, hd, eps, theta = dims(cfg)
    T = len(ids)                           # number of tokens in the sequence
    x = W["model.embed_tokens.weight"][ids]                        # [T, H] look up one embedding row per token id
    cos, sin = rope_tables(np.arange(T), hd, theta)                # rotation tables for positions 0..T-1
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)  # -infinity above the diagonal: blocks attention to the future
    for i in range(cfg["num_hidden_layers"]):                      # run the layers one after another
        p = f"model.layers.{i}."           # name prefix of this layer's tensors
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)       # normalize before attention
        q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)    # queries: multiply, split into heads, put heads first
        k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)  # keys, same steps but fewer heads
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)  # values
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)    # encode position by rotating queries and keys
        k = np.repeat(k, nh // nkv, axis=0)                        # each KV head serves a group of query heads: copy it
        v = np.repeat(v, nh // nkv, axis=0)
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(hd) + mask   # [nh, T, T] how much each token looks at each other token
        a = softmax(scores) @ v                                    # [nh, T, hd] weighted mix of the values
        a = a.transpose(1, 0, 2).reshape(T, nh * hd)               # glue the heads back into one vector per token
        x = x + a @ W[p + "self_attn.o_proj.weight"].T             # output projection, added to the input (first residual)
        h = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)   # normalize before the MLP
        g = h @ W[p + "mlp.gate_proj.weight"].T                    # MLP gate branch
        u = h @ W[p + "mlp.up_proj.weight"].T                      # MLP value branch
        x = x + (silu(g) * u) @ W[p + "mlp.down_proj.weight"].T    # gate x value, project down, add back (second residual)
    x = rmsnorm(x[-1], W["model.norm.weight"], eps)                # final norm, only for the last position
    head = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]   # reuse the embedding if there is no separate head
    return x @ head.T                                              # [vocab] one score per possible next token


if __name__ == "__main__":                 # run this part only when the file is executed directly
    from tokenizers import Tokenizer
    model_dir, prompt = sys.argv[1], sys.argv[2]   # command-line arguments: model folder and prompt text
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 20   # how many tokens to generate
    t0 = time.time()                       # start a timer
    W, cfg = load_model(model_dir)
    print(f"loaded {len(W)} tensors in {time.time() - t0:.1f}s")
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(prompt).ids           # prompt as a list of token ids
    print("prompt tokens:", ids)
    t0 = time.time()
    for _ in range(n_new):                 # generate one token per loop pass
        logits = forward(W, cfg, ids)      # recomputes the whole sequence every step
        ids.append(int(logits.argmax()))   # greedy: take the highest-scoring token and add it to the sequence
    dt = time.time() - t0                  # elapsed seconds
    print(tok.decode(ids))                 # token ids back to text
    print(f"{n_new / dt:.2f} tokens/s without a KV cache")