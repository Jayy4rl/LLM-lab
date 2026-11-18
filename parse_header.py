import json, os, struct, sys

path = sys.argv[1]

with open(path, "rb") as f:

    n = struct.unpack("<Q", f.read(8))[0]

    header = json.loads(f.read(n))

data_start = 8 + n

header.pop("__metadata__", None)

print(f"header bytes: {n:,} | tensor data starts at byte {data_start:,}")

print(f"file size: {os.path.getsize(path):,} | tensors in file:{len(header)}")

print("\nfirst 5 entries:")

for name, t in list(header.items())[:5]:

    print(f" {name} {t['dtype']} {t['shape']} bytes {t['data_offsets']}")


name = "model.embed_tokens.weight"

t = header[name]                           # its info: dtype, shape, data_offsets
start, end = t["data_offsets"]             # byte range of this tensor, counted from data_start
assert t["dtype"] == "BF16", t["dtype"]    # stop with an error if the numbers are not bfloat16
numel = 1                                  # will hold the total number of values in the tensor
for d in t["shape"]:                       # multiply all the dimensions together
    numel *= d
print(f"\n{name}: {numel:,} values, {end - start:,} bytes -> {(end - start) / numel} bytes per value")

with open(path, "rb") as f:
    f.seek(data_start + start)             # jump to where this tensor begins in the file
    raw = f.read(16)                       # read 16 bytes = 8 values x 2 bytes each
bits = struct.unpack("<8H", raw)           # decode as 8 little-endian unsigned 16-bit integers
# bfloat16 is simply the top 16 bits of a float32, so shift left 16 and reinterpret
vals = [struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in bits]   # per value: widen to 32 bits, reread the bytes as a float
print("first 8 values:", [round(v, 5) for v in vals])   # round each to 5 decimals for display