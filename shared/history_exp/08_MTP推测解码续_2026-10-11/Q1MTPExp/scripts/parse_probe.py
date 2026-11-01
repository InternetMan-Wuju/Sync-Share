#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1]))
r = d["requests"][0]
a = d.get("aggregate") or {}
print("yes" if r["meta"].get("spec_verify_ct") else "no", a.get("AL_weighted", "-"))
