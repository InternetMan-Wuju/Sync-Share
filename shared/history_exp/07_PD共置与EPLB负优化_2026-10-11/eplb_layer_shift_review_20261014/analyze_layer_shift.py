#!/usr/bin/env python3
"""Replay archived logical routing counts; all outputs stay in this directory.

These are demand/critical-work proxies, NOT measured kernel or model latency.
Rank permutations are checked on every recorded nonzero step and MoE layer.
Replica results assume equal splitting, because the dump lacks actual replica
dispatch counts. The stock policy is fitted to recorded counts (including the
historical decode multiplicity); evaluation separates phases and corrects /8
only when combining phases, following the archived recorder audit.
"""
import csv
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "cache" / "matplotlib"))
os.environ.setdefault("TMPDIR", str(HERE / "cache"))
sys.dont_write_bytecode = True

import numpy as np
import torch

torch.set_num_threads(2)
REF = HERE.parent / "pd_colloc_20261011" / "patches" / "stock_alg_ref"
sys.path.insert(0, str(REF))
import deepseek

RESULTS = HERE.parent / "pd_colloc_20261011" / "results"
RUNS = [
    "A_agentx_20261012_050236", "A_random_20261012_062417",
    "B_agentx_20261012_052314", "B_random_20261012_064400",
    "C_agentx_20261012_054412", "C_random_20261012_071801",
]
EP = 64
DECODE_FIX = 8.0


def place(heat, physical=256):
    return deepseek.rebalance_experts(
        torch.from_numpy(heat), physical, 1, 1, EP, False
    )[0].numpy()


def rank_demand(counts, pm):
    """[steps,layers,experts] -> [steps,layers,ranks], conserving demand."""
    layers, physical = pm.shape
    nrep = np.stack([np.bincount(row, minlength=256) for row in pm])
    gathered = np.take_along_axis(counts, pm[None, :, :], axis=2)
    gathered /= np.take_along_axis(nrep, pm, axis=1)[None, :, :]
    return gathered.reshape(-1, layers, EP, physical // EP).sum(axis=3)


def stats_init():
    return dict(rank_tot=np.zeros(EP), sum_layer_max=0.,
                sum_layer_mean=0., nonzero_step_layers=0, nonzero_steps=0,
                hotspot_lower_bound=0.)


def accumulate(out, rd, counts):
    out["rank_tot"] += rd.sum(axis=(0, 1))
    out["sum_layer_max"] += float(rd.max(axis=2).sum())
    out["sum_layer_mean"] += float(rd.mean(axis=2).sum())
    out["nonzero_step_layers"] += int((rd.sum(axis=2) > 0).sum())
    out["nonzero_steps"] += int((rd.sum(axis=(1, 2)) > 0).sum())
    out["hotspot_lower_bound"] += float(counts.max(axis=2).sum())
    assert np.isclose(rd.sum(), counts.sum(), rtol=1e-12, atol=1e-6)


def csv_write(name, rows):
    with (HERE / "tables" / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def global_permutation(base_rd):
    """Reassign whole rank blocks to flatten cumulative load; no regrouping."""
    cumulative = np.zeros(EP)
    perms = []
    for layer_load in base_rd:
        src = np.argsort(-layer_load, kind="stable")
        dst = np.argsort(cumulative, kind="stable")
        permutation = np.empty(EP, dtype=np.int64)
        permutation[dst] = src
        cumulative += layer_load[permutation]
        perms.append(permutation)
    return np.stack(perms)


def main():
    aggregate_rows, window_rows, perf_rows, manifest = [], [], [], []
    example = None
    verified_step_layers = 0
    for run in RUNS:
        accum = {}
        prev_heat = None
        n_windows = 0
        for p in sorted((RESULTS / run / "windows").glob("*.pt")):
            o = torch.load(p, map_location="cpu", weights_only=False)
            phase_counts = {
                phase: o.get("logical_count_" + phase)
                for phase in ("prefill", "decode")
            }
            present = [t for t in phase_counts.values() if t is not None]
            heat = sum(t.to(torch.float64).sum(0).numpy() for t in present)
            if heat.sum() == 0:
                continue
            layers, experts = heat.shape
            assert experts == 256 and o["ep_size"] == EP
            identity = np.tile(np.arange(experts), (layers, 1))
            stock = place(heat)
            roll_perm = (np.arange(EP)[None, :] - np.arange(layers)[:, None]) % EP
            base_heat_rd = rank_demand(heat[None, :, :], stock)[0]
            global_perm = global_permutation(base_heat_rd)
            slot_shift = np.stack([np.roll(row, l) for l, row in enumerate(stock)])
            maps = {
                "observed_layout": o["physical_to_logical_map"].numpy(),
                "identity": identity,
                "stock_fit_same_window": stock,
                "stock_slot_shift": slot_shift,
                "replicas64_fit_equal_split": place(heat, 320),
            }
            if prev_heat is not None:
                maps["stock_fit_previous_window"] = place(prev_heat)
            window_accum = {}
            for phase, tensor in phase_counts.items():
                if tensor is None:
                    continue
                scale = 1. if phase == "prefill" else 1. / DECODE_FIX
                for start in range(0, len(tensor), 64):
                    counts = tensor[start:start + 64].to(torch.float64).numpy() * scale
                    if not counts.any():
                        continue
                    derived = {}
                    for method, pm in maps.items():
                        rd = rank_demand(counts, pm)
                        derived[method] = rd
                    base = derived["stock_fit_same_window"]
                    derived["stock_rank_shift"] = np.take_along_axis(
                        base, roll_perm[None, :, :], axis=2
                    )
                    derived["stock_global_rank_permutation"] = np.take_along_axis(
                        base, global_perm[None, :, :], axis=2
                    )
                    verified_step_layers += int((base.sum(2) > 0).sum())
                    for method in ("stock_rank_shift", "stock_global_rank_permutation"):
                        assert np.array_equal(base.max(2), derived[method].max(2))
                        assert np.array_equal(
                            np.sort(base, axis=2), np.sort(derived[method], axis=2)
                        )
                    if "stock_fit_previous_window" in derived:
                        stale = derived["stock_fit_previous_window"]
                        derived["stock_previous_rank_shift"] = np.take_along_axis(
                            stale, roll_perm[None, :, :], axis=2
                        )
                        assert np.array_equal(stale.max(2), derived["stock_previous_rank_shift"].max(2))
                    for method, rd in derived.items():
                        for label in (phase, "total_corrected"):
                            key = (method, label)
                            accumulate(accum.setdefault(key, stats_init()), rd, counts)
                            accumulate(window_accum.setdefault(key, stats_init()), rd, counts)
            for (method, phase), values in window_accum.items():
                if values["sum_layer_mean"] > 0:
                    window_rows.append(dict(
                        run=run, window=p.name, phase=phase, method=method,
                        cumulative_maxmean=values["rank_tot"].max() / values["rank_tot"].mean(),
                        sequential_work_ratio=values["sum_layer_max"] / values["sum_layer_mean"],
                        critical_demand=values["sum_layer_max"],
                        nonzero_step_layers=values["nonzero_step_layers"],
                    ))
            if example is None and run.startswith("B_agentx") and int(o["dump_index"]) == 2:
                pre_heat = phase_counts["prefill"].to(torch.float64).sum(0).numpy()
                ex_base = rank_demand(pre_heat[None], stock)[0]
                example = dict(
                    run=run, window=p.name, stock=ex_base,
                    shifted=np.take_along_axis(ex_base, roll_perm, axis=1),
                    global_permutation=np.take_along_axis(ex_base, global_perm, axis=1),
                    identity=rank_demand(pre_heat[None], identity)[0],
                    replicas64=rank_demand(pre_heat[None], maps["replicas64_fit_equal_split"])[0],
                )
            manifest.append(dict(path=str(p.relative_to(HERE.parent.parent)),
                                 bytes=p.stat().st_size, windows_index=int(o["dump_index"])))
            n_windows += 1
            prev_heat = heat
            del o, phase_counts, present, tensor, derived
        for (method, phase), values in accum.items():
            aggregate_rows.append(dict(
                run=run, phase=phase, method=method, windows=n_windows,
                cumulative_maxmean=values["rank_tot"].max() / values["rank_tot"].mean(),
                sequential_work_ratio=values["sum_layer_max"] / values["sum_layer_mean"],
                critical_demand=values["sum_layer_max"],
                mean_demand=values["sum_layer_mean"],
                hotspot_floor_no_replica=values["hotspot_lower_bound"] / values["sum_layer_mean"],
                nonzero_steps=values["nonzero_steps"],
                nonzero_step_layers=values["nonzero_step_layers"],
                dispatch_assumption="equal replica splitting" if method == "replicas64_fit_equal_split" or (method == "observed_layout" and run.startswith("C_")) else "unique expert placement",
            ))
        print(run, "windows", n_windows, flush=True)
        for row in aggregate_rows:
            if row["run"] == run and row["phase"] == "total_corrected":
                print(" ", row["method"], "cumulative", round(row["cumulative_maxmean"], 4),
                      "sequential", round(row["sequential_work_ratio"], 4), flush=True)
    csv_write("aggregate_metrics.csv", aggregate_rows)
    csv_write("window_metrics.csv", window_rows)
    with (HERE.parent / "Q4Exp" / "tables" / "perf.csv").open() as f:
        for row in csv.DictReader(f):
            durations = [float(x) for x in row["rebalance_durations_s"].split(";") if x]
            group, load = row["group"], row["load"]
            dirname = {
                ("A0", "agentX"): "A0_agentx_20261012_044134",
                ("A0", "random"): "A0_random_20261012_060410",
                ("A", "agentX"): RUNS[0], ("A", "random"): RUNS[1],
                ("B", "agentX"): RUNS[2], ("B", "random"): RUNS[3],
                ("C", "agentX"): RUNS[4], ("C", "random"): RUNS[5],
            }[(group, load)]
            summary = json.loads((RESULTS / dirname / "bench" / "summary.json").read_text())
            wall = float(summary["actual_wall_s"])
            perf_rows.append(dict(
                run=dirname, group=group, workload=load, wall_s=wall,
                throughput_tok_s=float(summary["output_tokens_per_s"]),
                reported_rebalance_duration_sum_s=sum(durations),
                reported_rebalance_fraction_of_wall=sum(durations) / wall,
                caution="timed rebalance scope; not proven server-wide pause overlap",
            ))
    csv_write("performance_confounds.csv", perf_rows)
    evidence = dict(
        verified_nonzero_step_layer_pairs=verified_step_layers,
        invariant="whole-rank permutations preserve every step/layer rank-load multiset and maximum",
        windows_read=len(manifest), input_manifest=manifest,
        stock_source=str(REF / "deepseek.py"),
        stock_sha256=hashlib.sha256((REF / "deepseek.py").read_bytes()).hexdigest(),
        limitations=["logical demand is not measured latency", "replica traffic assumes equal split",
                     "historical decode /8 correction inherits the archived audit",
                     "previous-window fits use only strictly earlier windows"],
    )
    (HERE / "verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    np.savez_compressed(HERE / "example_rank_loads.npz", **example)
    print("VERIFIED", verified_step_layers, "nonzero step-layer pairs", flush=True)


if __name__ == "__main__":
    main()
