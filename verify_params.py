import json, os, sys

cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))   # read the settings file into a dictionary
V, H, L = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"]   # vocabulary size, hidden width, layer count
nh, nkv, I = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["intermediate_size"]   # query heads, key/value heads, MLP width
hd = cfg.get("head_dim") or H // nh        # size of one head: config value if present, else hidden / heads (// = integer division)

embed = V * H                              # embedding table: one row of H numbers per token
attn = H * (nh * hd) + 2 * H * (nkv * hd) + (nh * hd) * H   # q projection + (k and v) projections + o projection
mlp = 3 * H * I                            # gate, up and down matrices
norms = 2 * H                              # two RMSNorm weight vectors per layer
per_layer = attn + mlp + norms             # parameters in one layer
head = 0 if cfg.get("tie_word_embeddings") else V * H   # the output head is free when it reuses the embedding
total = embed + L * per_layer + H + head   # + H for the final norm

print(f"embedding : {embed:>14,}")         # :>14, means right-aligned in 14 characters, with separators
print(f"per layer : {per_layer:>14,}  (attention {attn:,}, mlp {mlp:,}, norms {norms:,})")
print(f"all layers: {L * per_layer:>14,}")
print(f"output head: {head:>13,}")
print(f"TOTAL     : {total:>14,}")
print(f"bytes at 2 bytes/param (bf16): {total * 2:,}")   # file size estimate: 2 bytes per bfloat16 value