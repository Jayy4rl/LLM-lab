import json, os, struct, sys               # standard library: JSON, file paths, binary decoding, command-line args
import numpy as np                         # numeric library, used under the short name np
from tokenizers import Tokenizer           # Hugging Face tokenizer library

model_dir = sys.argv[1]                    # first argument: the model folder
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"   # second argument if given, else a default

# 1. Text -> token ids
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))   # load the tokenizer rules from the model folder
enc = tok.encode(prompt)                   # split the prompt into tokens: enc.ids are numbers, enc.tokens the text pieces
print("vocab size:", tok.get_vocab_size()) # how many different tokens exist
print("characters:", len(prompt), "| tokens:", len(enc.ids))
print("pieces:", list(zip(enc.ids, enc.tokens)))   # zip pairs each id with its text piece
print("round trip:", repr(tok.decode(enc.ids)))    # ids back to text; repr shows quotes and escapes

# 2. Token id -> embedding vector, reading only the bytes of that row from disk
path = os.path.join(model_dir, "model.safetensors")   # join folder and file name
with open(path, "rb") as f:                # read the header exactly as in Stage 1, Part A
    n = struct.unpack("<Q", f.read(8))[0]  # header length
    header = json.loads(f.read(n))         # header as a dictionary
t = header["model.embed_tokens.weight"]    # info about the embedding matrix
assert t["dtype"] == "BF16"                # stop if the numbers are not bfloat16
vocab, hidden = t["shape"]                 # unpack two dimensions: rows = tokens, columns = vector size
mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + t["data_offsets"][0], shape=(vocab, hidden))   # view those bytes as a 2D array of 16-bit ints, without reading them yet

def embed(token_id):                       # return the embedding vector for one token id
    bits = np.asarray(mm[token_id]).astype(np.uint32) << 16   # read one row, widen to 32 bits, shift: bfloat16 -> float32 bit pattern
    return bits.view(np.float32)           # reinterpret those bits as float32 numbers

print()
for tid, piece in zip(enc.ids, enc.tokens):   # loop over (id, text) pairs
    v = embed(tid)                         # fetch this token's vector
    print(f"id {tid:6} {piece!r:14} -> {hidden} floats, first 4 = {np.round(v[:4], 4)}, length {np.linalg.norm(v):.3f}")   # norm = vector length

# 3. Do related words sit closer together?
def cosine(a, b):                          # direction similarity of two vectors: 1 same, 0 unrelated, -1 opposite
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   # dot product divided by the two lengths

words = [" Paris", " London", " banana"]
vecs = {w: embed(tok.encode(w).ids[0]) for w in words}   # dictionary: word -> vector of its first token
print()
for a, b in [(" Paris", " London"), (" Paris", " banana"), (" London", " banana")]:   # the three pairs to compare
    print(f"cosine({a!r}, {b!r}) = {cosine(vecs[a], vecs[b]):.3f}")