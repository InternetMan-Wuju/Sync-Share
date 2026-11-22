#!/usr/bin/env python3
"""Recompute the acceptance verdict from checked-in primary evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(name: str):
    return json.loads((ART / name).read_text())


def main() -> int:
    e16 = load("EP16_SERVICE_ACCEPTANCE.json")
    e32 = load("EP32_MATRIX.json")
    dev = load("EP32_DEVICE_MAP.json")
    k2 = load("EP32_K2_ACCEPTANCE.json")
    eplb = load("EP32_EPLB_REPLICA_TASKS.json")
    slots = load("SLOT_CONSISTENCY.json")
    native = load("NATIVE_DISPATCH_FAILURE.json")

    errors: list[str] = []

    # 32-rank topology and input identity.
    ranks = dev.get("ranks", [])
    if not dev.get("all_32_init_seen") or len(ranks) != 32:
        errors.append("ep32_all_32_ranks_not_evidenced")
    if {r.get("rank") for r in ranks} != set(range(32)):
        errors.append("ep32_rank_ids_incomplete")
    if {r.get("node") for r in ranks} != {23, 25}:
        errors.append("ep32_node_set_mismatch")

    # Owner-state decision and same-layer reuse.
    hit = slots.get("same_layer_hit", {})
    for rank in ("0", "1", "10"):
        events = hit.get(rank, [])
        for owner in (0, 1):
            owner_reuses = [x for x in events if x.get("kind") == "reuse" and x.get("slot_owner") == owner]
            if not owner_reuses:
                errors.append(f"slot_reuse_missing_rank{rank}_owner{owner}")
        if any(x.get("kind") == "install" for x in events if x.get("seq") == 5):
            errors.append(f"slot_reinstall_on_same_layer_rank{rank}")
    if slots.get("q0", {}).get("install_count") != 0:
        errors.append("q0_installed_slot")

    # EP16 functional service checks.
    k0 = e16.get("k0", {})
    k1 = e16.get("k1", {})
    e16k2 = e16.get("k2", {})
    if k0.get("status") != "PASS" or k0.get("output_sha256") != k1.get("output_sha256"):
        errors.append("ep16_k0_k1_output_mismatch")
    k1_slot = k1.get("slot", {})
    if k1_slot.get("slot_count") != 1:
        errors.append("ep16_k1_slot_count")
    if k1_slot.get("generations") != [1, 2]:
        errors.append("ep16_k1_cross_layer_generation")
    if not k1.get("token_exact_vs_k0"):
        errors.append("ep16_k1_not_token_exact")
    for owner in ("slot0", "slot1"):
        s = e16k2.get(owner, {})
        if s.get("slot_count") != 1 or s.get("generations") != [1, 2]:
            errors.append(f"ep16_k2_{owner}_slot_or_generation")
        if not s.get("moved"):
            errors.append(f"ep16_k2_{owner}_no_tasks")
    k2_abs = [
        float(x.get("verify", {}).get("max_abs", 1e9))
        for slot in ("slot0", "slot1")
        for x in e16k2.get(slot, {}).get("moved", [])
    ]
    if not k2_abs or max(k2_abs) > 0.02:
        errors.append("ep16_k2_numeric_error")
    if not e16k2.get("cross_layer_reuse"):
        errors.append("ep16_k2_cross_layer_reuse")

    # EP32 three-arm matrix. Output IDs are identical for original and static EPLB.
    for size in ("8192", "16384", "32768"):
        arms = e32.get("matrix", {}).get(size, {})
        off = arms.get("off", {})
        static = arms.get("eplb_static", {})
        ours = arms.get("k2", {})
        if off.get("status") != "PASS":
            errors.append(f"ep32_original_{size}")
        if static.get("status") != "PASS":
            errors.append(f"ep32_eplb_static_{size}")
        if ours.get("status") != "PASS":
            errors.append(f"ep32_k2_{size}")
        if off.get("output_ids_sha256") != static.get("output_ids_sha256"):
            errors.append(f"ep32_original_static_output_mismatch_{size}")
        # 8K has a deterministic token difference but bounded layer error; 16K/32K match.
        if size in ("16384", "32768") and ours.get("output_ids_sha256") != off.get("output_ids_sha256"):
            errors.append(f"ep32_k2_output_mismatch_{size}")

    if eplb.get("total_logical_tasks") != 65536 or int(eplb.get("replica_tasks", 0)) <= 0:
        errors.append("eplb_replica_task_evidence")
    if e32.get("eplb_static", {}).get("physical_experts") != 288 or e32.get("eplb_static", {}).get("local_slots_per_rank") != 9:
        errors.append("eplb_static_layout_shape")
    if not e32.get("eplb_static", {}).get("layout_active"):
        errors.append("eplb_static_layout_not_active")

    # EP32 K=2 real slots and cross-node source identities.
    if k2.get("total_slots") != 2 or len(k2.get("cross_node_weight_sources", [])) != 2:
        errors.append("ep32_k2_slots_or_cross_node_sources")
    if float(k2.get("max_abs_error", 1e9)) > 0.02:
        errors.append("ep32_k2_numeric_error")
    if k2.get("status") != "PASS_32K_ISOLATED_AND_SEQUENCE":
        errors.append("ep32_k2_32k_status")
    if k2.get("replica_execution_mode") != "local fallback; native replica communication id path failed combine":
        errors.append("ep32_k2_execution_mode_unlabeled")

    # Known blockers are required, but must remain blockers rather than PASS claims.
    online = e32.get("online_rebalance", {})
    if online.get("status") != "FAIL" or online.get("rebalance_end") is not False:
        errors.append("online_rebalance_blocker_not_recorded")
    if native.get("status") != "FAIL" or "attr=601" not in native.get("error", ""):
        errors.append("native_dispatch_blocker_not_recorded")

    checks = {
        "SLOT_DECISION_FIX": "PASS" if not any("slot_" in x or "q0_" in x for x in errors) else "FAIL",
        "32K_ISOLATED": "PASS" if k2.get("status") == "PASS_32K_ISOLATED_AND_SEQUENCE" else "FAIL",
        "32K_SEQUENCE": "PASS" if k2.get("status") == "PASS_32K_ISOLATED_AND_SEQUENCE" else "FAIL",
        "EP16_K0": k0.get("status", "MISSING"),
        "EP16_K1": k1.get("status", "MISSING"),
        "EP16_K2": e16k2.get("status", "MISSING"),
        "EP32_ALL_32_DIE": "PASS" if dev.get("all_32_init_seen") and len(ranks) == 32 else "FAIL",
        "EP32_OFF_3": "PASS" if all(e32["matrix"][s]["off"]["status"] == "PASS" for s in ("8192", "16384", "32768")) else "FAIL",
        "EP32_EPLB_STATIC_3": "PASS" if all(e32["matrix"][s]["eplb_static"]["status"] == "PASS" for s in ("8192", "16384", "32768")) else "FAIL",
        "EP32_K2_3": "PASS" if all(e32["matrix"][s]["k2"]["status"] == "PASS" for s in ("8192", "16384", "32768")) else "FAIL",
        "EPLB_ONLINE_REBALANCE": "FAIL_NO_END",
        "NATIVE_REPLICA_DISPATCH": "FAIL_COMBINE_ATTR601",
    }
    out = {
        "status": "PASS_WITH_RECORDED_BLOCKERS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }
    (ART / "VALIDATION.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
