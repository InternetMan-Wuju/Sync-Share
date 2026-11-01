\--ep-size
\--ep-num-redundant-experts 
\--ep-dispatch-algorithm
\--init-expert-location

\--enable-eplb
\--eplb-algorithm
\--eplb-rebalance-num-iterations
\--eplb-rebalance-layers-per-chunk
\--eplb-min-rebalancing-utilization-threshold

\--expert-distribution-recorder-mode
\--expert-distribution-recorder-buffer-size
\--expert-balancedness-report-mode

| 参数 | Prefill | Decode | 是否显式设置 |
|---|---:|---:|---|
| `--ep-size` | **16（实际生效）** | **32** | P 未直接写；D 显式 `32` |
| `--ep-num-redundant-experts` | `0` | `0` | 未传，使用默认值 |
| `--ep-dispatch-algorithm` | `None` | `None` | 未传，使用默认值 |
| `--init-expert-location` | `trivial` | `trivial` | 未传，使用默认值 |
| `--enable-eplb` | 开启 | 开启 | 显式开启 |
| `--eplb-algorithm` | `auto` | `auto` | 未传，使用默认值 |
| `--eplb-rebalance-num-iterations` | `1000` | `1000` | 未传，使用默认值 |
| `--eplb-rebalance-layers-per-chunk` | `None` | `None` | 未传，使用默认值 |
| `--eplb-min-rebalancing-utilization-threshold` | `1.0` | `1.0` | 未传，使用默认值 |
| `--expert-distribution-recorder-mode` | `stat` | `stat` | 显式设置 |
| `--expert-distribution-recorder-buffer-size` | `-1` | `-1` | 显式设置 |
| `--expert-balancedness-report-mode` | `off` | `off` | 未传，使用默认值 |


| 服务 | `n_group` | 运行时 `nnodes` | `auto` 最终结果 |
|---|---:|---:|---|
| Prefill P0 | 1 | 1 | **`deepseek_hierarchical`** |
| Prefill P1 | 1 | 1 | **`deepseek_hierarchical`** |
| Decode D（跨两机） | 1 | 2 | **`deepseek`** |
