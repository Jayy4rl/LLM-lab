import mmap, sys                           # mmap: map a file into memory

def rss():                                 # return how much RAM this process is actually using
    for line in open("/proc/self/status"): # Linux exposes process info as a text file; read it line by line
        if line.startswith("VmRSS"):       # the line that reports resident memory
            return line.split(":")[1].strip()   # keep the part after the colon, trimmed

f = open(sys.argv[1], "rb")                # open the file given on the command line
m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)   # map the whole file (length 0 = all); nothing is read yet
print("mapped", f"{len(m):,}", "bytes | RSS after mmap:", rss())

acc = 0                                    # running sum, only so the reads cannot be skipped
for i in range(0, len(m), 4096):           # step through the file one 4096-byte page at a time
    acc += m[i]                            # touch one byte per page: this triggers the page fault that loads it
print("RSS after touching every page:", rss())