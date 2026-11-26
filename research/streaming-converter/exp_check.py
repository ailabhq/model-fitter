import json,sys,hashlib,torch,base64,os
if len(sys.argv)>1 and sys.argv[1]=="nas":
    from safetensors import safe_open
    d="/scratch/src/qwen3.8-27b"; idx=json.load(open(d+"/model.safetensors.index.json"))["weight_map"]
    names=sorted(n for n in idx if n.endswith("A_log"))
    out={}
    for n in names:
        with safe_open(d+"/"+idx[n],"pt") as f: t=f.get_tensor(n)
        out[n]=(str(t.dtype), base64.b64encode(t.contiguous().view(torch.uint8).numpy().tobytes()).decode(), hashlib.sha256((-torch.exp(t.float())).numpy().tobytes()).hexdigest())
    print(json.dumps(out))
else:
    src=json.load(sys.stdin); same=diff=0; ulps=[]
    import numpy as np
    for n,(dt,b,h) in src.items():
        t=torch.frombuffer(bytearray(base64.b64decode(b)),dtype=torch.uint8).view(getattr(torch,dt.split(".")[1]))
        r=(-torch.exp(t.float())).numpy()
        if hashlib.sha256(r.tobytes()).hexdigest()==h: same+=1
        else: diff+=1
    print(json.dumps({"tensors":len(src),"same_exp_result":same,"different_exp_result":diff,"torch_mac":torch.__version__}))
