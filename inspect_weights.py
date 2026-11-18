import glob, json, os, re, sys             # glob: find files by pattern; re: regular expressions
from collections import OrderedDict        # a dictionary that remembers insertion order
from safetensors import safe_open          # library function that opens a safetensors file

model_dir = sys.argv[1]                    # first command-line argument: the model folder
cfg = json.load(open(os.path.join(model_dir, "config.json")))   # read config.json into a dictionary
keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size", "tie_word_embeddings"]   # the settings to show
print({k: cfg.get(k) for k in keys}, "\n")  # dictionary comprehension: just those settings (.get gives None if missing)

groups = OrderedDict()                     # collapsed tensor name -> [shape, dtype, size, copies]
total = 0                                  # running total of parameters
for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):   # every weight file, in order (big models are split)
    with safe_open(f, framework="numpy") as st:   # open the file; "numpy" only selects the array flavor
        for name in st.keys():             # every tensor name in this file
            sl = st.get_slice(name)        # lazy handle: reads the header only, not the data
            shape, dtype = sl.get_shape(), sl.get_dtype()   # dimensions and number format
            n = 1
            for d in shape:                # multiply the dimensions to count the values
                n *= d
            total += n
            g = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)   # turn "layers.17." into "layers.*." so all layers share one key
            if g in groups:
                groups[g][3] += 1          # seen before: bump the copy count
            else:
                groups[g] = [shape, dtype, n, 1]   # first time: store the details with a count of 1

print(f"{'tensor (layer index collapsed)':56} {'shape':16} {'dtype':6} copies")   # header row; :56 pads to 56 characters
for g, (shape, dtype, n, c) in groups.items():   # unpack each stored list into four variables
    print(f"{g:56} {str(shape):16} {dtype:6} x{c}")   # one aligned row per kind of tensor
print(f"\ntotal parameters: {total:,}")