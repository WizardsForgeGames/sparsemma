import sys, os, statistics; sys.path.insert(0, os.getcwd())
import torch, torch.nn as nn
from python.quantize_utils import Int8Linear
def t(f,x,n=300):
    for _ in range(50): f(x)
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record(); f(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
torch.manual_seed(0)
for K,N in [(1536,6144),(6144,1536),(1280,5120),(5120,1280)]:
    lin=nn.Linear(K,N).cuda().half(); sp=Int8Linear.from_linear_sparse(lin); sp.mode='sparse'
    row=[]
    for M in (64,256,1024,4096):
        x=torch.randn(M,K,device='cuda',dtype=torch.half); row.append(f"M={M}:{t(sp,x):.4f}")
    print(f"{K}x{N} "+"  ".join(row))
