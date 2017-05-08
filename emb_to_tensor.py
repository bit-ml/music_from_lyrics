import torch
import sys

if __name__ == "__main__":
    emb_file = sys.argv[1]
    with open(emb_file, "r") as f:
        ls = f.read().strip().split(" ")[2:]
    print(ls[:100])
    arr = [float(x) for x in ls]
    print(arr)
