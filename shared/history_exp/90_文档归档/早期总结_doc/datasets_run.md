# GLM-5.2 × 910C：数据集与压测说明（交给实验 Agent）

本文件说明：现在有哪些数据集、各自格式/路径，以及用哪两个脚本驱动
SGLang 上的 GLM-5.2 压测（尤其是 EP / EPLB 实验）。EP/EPLB 实验方案见
同目录的 `ep_exp.md` 和 `ep_exp2.md`。

## 1. 环境约定

- 工作目录：`/home/chengyetai/workspace/glm52_ep64_chengyetai`
- 数据集目录：`/home/chengyetai/workspace/glm52_ep64_chengyetai/dataset`
- 服务：SGLang 的 OpenAI 兼容接口，默认 `http://127.0.0.1:30000`
- 请求体里模型名固定为 `glm-5.2`（脚本已写死）
- 客户端只用 Python 标准库（`urllib` + `concurrent.futures`），无额外依赖

## 2. 数据集清单

一共三份，都已经是“dolly / alpaca 单轮指令”格式，可直接作为请求集。

| 数据集 | 文件路径（相对 dataset/） | 条数 | 字段 | 语言 |
|---|---|---|---|---|
| databricks-dolly-15k | `databricks-dolly-15k.jsonl` | 15,011 | instruction / context / response / category | 英文 |
| KenithZ-dolly-zh-51k | `KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl` | 51,179 | instruction / context / response / category | 中文 |
| alpaca-gpt4-data-en | `alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl` | 52,002 | instruction / context / response | 英文 |

合计：**118,192 条**。全部跑完远超一小时，所以必须用下面的“时间预算 / 抽样 / 续跑”。

补充说明：

- `alpaca-gpt4-data-en/` 目录里还保留了原始文件：
  - `train.csv`：52,002 行，列 `instruction,input,output,text`（原始 CSV）
  - `alpaca-gpt4-data-en.json`：仓库元数据，可忽略
  - `alpaca-gpt4-data-en.jsonl`：本对话中转换出的 dolly 格式 JSONL（`input→context`、`output→response`，丢弃 `text`）
- 许可：dolly 原版 CC-BY-SA-3.0；KenithZ-dolly-zh-51k 为 MIT；alpaca-gpt4-data-en 为 CC BY-NC 4.0（非商用）。

## 3. 数据格式与 prompt 构造

统一字段含义：

```text
instruction  用户指令（必填）
context      额外上下文（可为空）
response     参考回答（仅用于评测/记录，压测请求里不发给模型）
category     类别（只有 dolly / KenithZ 有；alpaca-en 没有）
```

压测时把一行变成一条 user 请求：

```text
prompt = instruction
如果 context 非空：
    prompt = f"{instruction}\n\nContext:\n{context}"
```

示例（databricks-dolly-15k）：

```json
{"instruction": "When did Virgin Australia start operating?",
 "context": "Virgin Australia ... commenced services on 31 August 2000 ...",
 "response": "Virgin Australia commenced services on 31 August 2000 ...",
 "category": "closed_qa"}
```

## 4. 压测脚本

### 4.1 已有：`dataset/bench_dolly.py`

单文件、固定条数的压测。参数：

```text
--url         服务地址，默认 http://127.0.0.1:30000
--jsonl       单个 JSONL 文件
--out         结果输出（summary + results）
--limit       最多跑多少条，默认 32
--concurrency 并发，默认 4
--max-tokens  每个请求 max_tokens，默认 256
--timeout     单请求超时秒数，默认 600
```

调用它的 shell：`run_dolly.sh`、`ep_lab/run_dolly_ep_cell.sh`、
`ep_lab/run_dolly_perpass.sh`、`collect_ep_metrics.sh`。

### 4.2 新增：`dataset/bench_multi.py`（本对话新增，未改动旧脚本）

为“时间不够、不知道一个文件跑多久”设计，支持：

```text
--jsonl A B C ...   多个文件按顺序接续；先跑完 A 再接 B、C
--duration SECONDS  wall-clock 预算，到点不再提交新请求，在途请求跑完
--sample N          抽样条数（0 = 不抽样）
--stratify FIELD    分桶字段：category 或 dataset（配合 --sample）
--seed N            抽样随机种子（固定后每次抽样结果一致，便于复现）
--limit N           总任务数上限（0 = 不限）
--per-file-limit N  每个文件行数上限（0 = 不限）
--resume            跳过 --progress 里已完成的 key，断点续跑
--dry-run           只预览任务流/分桶，不发请求
--out / --progress  最终结果 / 增量进度（progress 累加，用于续跑）
```

完整参数与实现见：
`/home/chengyetai/workspace/glm52_ep64_chengyetai/dataset/bench_multi.py`。

### 4.3 新增包装：`dataset/run_multi.sh`

默认按下面顺序跑三份文件，输出到 `dataset/multi_runs/<RUN_NAME>/`：

```text
databricks-dolly-15k.jsonl
KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl
alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl
```

可用环境变量覆盖：

```text
URL          默认 http://127.0.0.1:30000
DURATION     默认 3600
CONCURRENCY  默认 4
MAX_TOKENS   默认 256
SAMPLE       默认 0（>0 时开启抽样）
STRATIFY     默认 category
SEED         默认 0
RESUME       默认 0（=1 开启续跑）
RUN_NAME / OUT_DIR   运行名 / 输出目录
```

## 5. 推荐跑法

### 5.1 先预览，确认抽样和任务流

```bash
cd /home/chengyetai/workspace/glm52_ep64_chengyetai/dataset

./bench_multi.py \
  --jsonl databricks-dolly-15k.jsonl \
           KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl \
           alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl \
  --sample 2000 --stratify dataset --seed 0 --dry-run
```

`--dry-run` 会打印：加载数、选中数、按文件分布、按分桶分布、前 10 条顺序。

### 5.2 短时校准，估算 1 小时能跑多少

```bash
./bench_multi.py \
  --url http://127.0.0.1:30000 \
  --jsonl databricks-dolly-15k.jsonl \
           KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl \
           alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl \
  --duration 60 --concurrency 4 --max-tokens 256 \
  --out /tmp/calib.json --progress /tmp/calib.progress.jsonl
```

看输出 summary 里的 `requests_per_s`、`tokens_per_s`，再决定 1 小时覆盖量。

### 5.3 跑满 1 小时（多文件接续，不用估）

```bash
./bench_multi.py \
  --url http://127.0.0.1:30000 \
  --jsonl databricks-dolly-15k.jsonl \
           KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl \
           alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl \
  --duration 3600 --concurrency 4 --max-tokens 256 \
  --out multi_bench.json --progress multi_bench.progress.jsonl
```

### 5.4 三份数据按比例抽样（代表性最好）

```bash
./bench_multi.py \
  --url http://127.0.0.1:30000 \
  --jsonl databricks-dolly-15k.jsonl \
           KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl \
           alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl \
  --sample 2000 --stratify dataset --seed 0 \
  --duration 3600 --concurrency 4 --max-tokens 256 \
  --out multi_bench.json --progress multi_bench.progress.jsonl
```

### 5.5 中断后续跑

```bash
./bench_multi.py \
  --url http://127.0.0.1:30000 \
  --jsonl databricks-dolly-15k.jsonl \
           KenithZ-dolly-zh-51k/KenithZ-dolly-zh-51k.jsonl \
           alpaca-gpt4-data-en/alpaca-gpt4-data-en.jsonl \
  --progress multi_bench.progress.jsonl --resume \
  --duration 3600 --concurrency 4 --max-tokens 256 \
  --out multi_bench_resume.json
```

或直接：

```bash
DURATION=3600 ./run_multi.sh
RESUME=1 DURATION=3600 ./run_multi.sh
```

## 6. 给 EP / EPLB 实验的对接建议

`ep_exp.md` / `ep_exp2.md` 要求“所有 EP / 并发 / 配置使用同一个固定混合请求集、
同一请求顺序、同一采样参数”。对应做法：

- 固定 `--sample N --stratify dataset --seed 0`（或 `--stratify category`），
  生成固定请求集；
- 每次实验用同一个 `--seed` 和 `--sample`，保证请求清单一致；
- `--duration` 只作为停止条件，不影响“请求顺序”本身；
- 用 `--dry-run` 生成的顺序 + `--progress` 记录完成 key，方便对齐与续跑；
- 如需更精细的“输入长度 × 输出长度”分桶，可在 `bench_multi.py` 上再扩展一个
  按 token/字符长度分桶的抽样逻辑（当前已支持 `category` / `dataset` 两个字段）。

## 7. 结果文件说明

`bench_multi.py` 每次会写两个文件：

- `--out`（如 `multi_bench.json`）：本次运行的 `summary` + `results` 数组；
- `--progress`（如 `multi_bench.progress.jsonl`）：**累计**的逐条结果，
  每条含 `key/dataset/line_no/status/e2e_s/error/usage/finish_reason/content_chars`。

`summary` 关键字段：

```text
loaded / selected / submitted / completed
duration_budget_s / wall_s / stopped_by_deadline
ok / err
e2e_p50_s / e2e_p90_s / e2e_p99_s
prompt_tokens / completion_tokens
requests_per_s / tokens_per_s
by_file
```

## 8. 注意事项

- `--stratify category` 时：中文 dolly 的 `category` 为空串、alpaca-en 没有
  `category`，会被当成两个大桶；若要“三个数据集都覆盖到”，用
  `--stratify dataset`。
- 续跑时建议不要同时改 `--sample/--seed/--stratify`，否则抽样结果会变；
  最稳的续跑方式是“不加抽样，只加 `--resume`”。
- 服务端口和模型名：脚本默认 `http://127.0.0.1:30000`、模型 `glm-5.2`；
  若服务不在本机或模型名不同，用 `--url` 传入并同步改脚本里的 `"model"` 字段。
