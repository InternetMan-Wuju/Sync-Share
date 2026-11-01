#!/usr/bin/env bash
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 - <<'PY'
import hashlib,inspect,json
from pathlib import Path
import deep_ep
import deep_ep_cpp
root=Path(deep_ep.__file__).parent
print(json.dumps({'deep_ep_path':str(root),'extension_path':str(Path(deep_ep_cpp.__file__).resolve()),'extension_sha256':hashlib.sha256(Path(deep_ep_cpp.__file__).read_bytes()).hexdigest(),'dispatch_signature':str(inspect.signature(deep_ep.Buffer.dispatch))},ensure_ascii=False))
PY
