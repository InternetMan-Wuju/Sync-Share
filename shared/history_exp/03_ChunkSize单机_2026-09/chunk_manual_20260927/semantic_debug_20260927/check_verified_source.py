#!/usr/bin/env python3
"""Read restored key tensors on CPU and save their fingerprints and statistics."""
import hashlib
import json
import time
from pathlib import Path
import torch
import safetensors

torch.set_num_threads(16)
base = Path('/models/glm-q')
index = json.loads((base/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
results = []
for name in ['lm_head.weight', 'model.embed_tokens.weight', 'rot.weight']:
    start = time.monotonic()
    with safetensors.safe_open(base/index[name], framework='pt', device='cpu') as source:
        tensor = source.get_tensor(name)
    flat = tensor.reshape(-1)
    sha = hashlib.sha256()
    raw = flat.view(torch.uint8).numpy()
    for offset in range(0, len(raw), 32*1024*1024):
        sha.update(raw[offset:offset+32*1024*1024].tobytes())
    nonfinite = extreme = 0
    peak = 0.0
    for offset in range(0, flat.numel(), 1024*1024):
        values = flat[offset:offset+1024*1024].float()
        finite = torch.isfinite(values)
        nonfinite += int((~finite).sum())
        values = values[finite].abs()
        if values.numel():
            peak = max(peak, float(values.max()))
            extreme += int((values > 1e10).sum())
    result = {'name':name, 'shape':list(tensor.shape), 'dtype':str(tensor.dtype), 'sha256_tensor_bytes':sha.hexdigest(), 'nonfinite':nonfinite, 'abs_gt_1e10':extreme, 'finite_abs_max':peak, 'elapsed_s':time.monotonic()-start}
    results.append(result)
    Path('/output/verified_key_tensors.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    del tensor, flat, raw
    if nonfinite or extreme:
        raise RuntimeError('abnormal restored tensor')
