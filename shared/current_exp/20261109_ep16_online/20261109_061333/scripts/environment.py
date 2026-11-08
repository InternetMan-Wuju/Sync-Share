import hashlib,importlib.metadata as m,json,platform
from pathlib import Path
root=Path('/sgl-workspace/sglang/python/sglang/srt')
files=['models/deepseek_v2.py','models/glm4_moe.py','hardware_backend/npu/moe/topk.py','layers/moe/topk.py','layers/moe/fused_moe_triton/layer.py','layers/moe/ep_moe/layer.py','layers/quantization/modelslim/modelslim.py','hardware_backend/npu/quantization/moe_methods.py','layers/moe/moe_runner/ascend.py','layers/moe/token_dispatcher/deepep.py','distributed/parallel_state.py','eplb/expert_location.py']
result={'packages':{n:m.version(n) for n in ['torch','torch-npu','sglang','sgl-kernel-npu','transformers']},'platform':platform.platform(),'python':platform.python_version(),'source_sha256':{str(root/f):hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'model_config_sha256':hashlib.sha256(Path('/models/glm-q/config.json').read_bytes()).hexdigest(),'readonly_weight_manifest_sha256':'7870fe31cb26405e7b28e67023c9ce0d413e0506f40fc7704710e38b40b2abc5'}
Path('/runexp/ENVIRONMENT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['packages']))
