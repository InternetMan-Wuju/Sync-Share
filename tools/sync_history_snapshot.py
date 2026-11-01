#!/usr/bin/env python3
"""Build a compact, remote-readable mirror of history_exp.

The full archive stays on the local data volume. This script copies only
documentation, experiment scripts/patches, selected summary tables, and key
figures into Share/shared/history_exp.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SHARE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SHARE_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "history_exp"
DEST_ROOT = SHARE_ROOT / "shared" / "history_exp"
TEMP_ROOT = SHARE_ROOT / "shared" / ".history_exp.tmp"

DOC_EXTS = {".md"}
TEXT_EXTS = {".txt"}
CODE_EXTS = {".py", ".sh", ".patch", ".diff", ".yaml", ".yml", ".toml", ".service", ".conf"}
DATA_EXTS = {".json", ".csv"}
FIGURE_EXTS = {".png", ".svg", ".pdf"}

# Raw results, generated data, upstream source snapshots, caches, and duplicated
# patch trees are not part of the remote-readable mirror.
EXCLUDE_PARTS = {
    "results", "logs", "datasets", "weights", "metrics",
    "__pycache__", ".cache", "cache", "verifier-triton-cache",
    "node_modules", "source", "source_snapshot", "source_current_image",
    "source_node20_united", "source_v0516", "source_v16", "source_v19",
    "readonly_source23", "readonly_source25", "pdf_tools", "pr32209",
    "weight_overlay", "network", "audit_seed", "snapshot_node23",
    "snapshot_node25", "snapshot_node31", "snapshot_datalib",
    "readonly_resume", "readonly_weight_source",
}
SUMMARY_KEYS = (
    "summary", "report", "index", "status", "coverage", "overview",
    "all_experiments", "汇总", "指标", "结果", "实测", "覆盖", "核对", "分析",
)


def should_include(path: Path, rel: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    size = path.stat().st_size
    ext = path.suffix.lower()
    parts = set(rel.parts)

    if ext in DOC_EXTS:
        return True
    if parts & EXCLUDE_PARTS:
        return False

    if ext in TEXT_EXTS and size <= 1_000_000:
        return True
    if ext in CODE_EXTS and size <= 1_000_000:
        return True
    if ext in DATA_EXTS and size <= 500_000:
        lowered = path.name.lower()
        if any(key in lowered for key in SUMMARY_KEYS):
            return True

    if ext == ".csv" and "tables" in parts and size <= 2_000_000:
        return True

    if ext in FIGURE_EXTS and size <= 2_000_000:
        if "pdf_review" in parts:
            return False
        if ("figs" in parts or "figures" in parts) and not (
            parts & {"archive", "runs", "results", "logs", "snapshots"}
        ):
            return True
        if "analysis" in parts:
            return True
    return False


def collect(source_root: Path) -> list[tuple[Path, Path, int]]:
    selected: list[tuple[Path, Path, int]] = []
    for path in source_root.rglob("*"):
        rel = path.relative_to(source_root)
        try:
            if should_include(path, rel):
                selected.append((path, rel, path.stat().st_size))
        except OSError as exc:
            print(f"SKIP {rel}: {exc}")
    return sorted(selected, key=lambda item: str(item[1]))


def scan_archive(source_root: Path) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in source_root.rglob("*"):
        try:
            if path.is_file():
                key = path.relative_to(source_root).parts[0]
                stats[key][0] += 1
                stats[key][1] += path.stat().st_size
        except OSError:
            continue
    return {key: (value[0], value[1]) for key, value in stats.items()}


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(selected: list[tuple[Path, Path, int]], archive: dict[str, tuple[int, int]]) -> str:
    selected_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _, rel, size in selected:
        selected_stats[rel.parts[0]][0] += 1
        selected_stats[rel.parts[0]][1] += size

    lines = [
        "# history_exp 远端精选镜像清单",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        f"- 本地完整归档：`{SOURCE_ROOT}`",
        f"- 精选镜像：`{DEST_ROOT}`",
        f"- 已同步：**{len(selected)} 个文件，{human_size(sum(s for _, _, s in selected))}**",
        f"- 原始归档：**{sum(n for n, _ in archive.values())} 个文件，{human_size(sum(s for _, s in archive.values()))}**",
        "",
        "## 收录规则",
        "",
        "- 全部 `.md` 实验文档、README、总结、结论与分析。",
        "- 实验脚本、分析脚本、shell 启动脚本、补丁和配置。",
        "- 名称含 summary/report/index/status/coverage/overview/汇总/指标/结果/实测/覆盖/核对/分析的小型 JSON/CSV。",
        "- 顶层 `tables/` 下的 CSV 汇总表。",
        "- 顶层 `figs/`、`figures/` 和分析目录中的关键图。",
        "",
        "## 未同步内容",
        "",
        "- `results/` 中除 Markdown 和精选图表外的原始结果。",
        "- `logs/`、`metrics/`、`datasets/`、`weights/`。",
        "- `.pt`、`.safetensors`、大型 manifest、原始请求/响应和缓存。",
        "- 上游源码完整快照、重复 patch tree、Triton cache 和 PDF 工具依赖。",
        "- 单文件超过 1 MiB 的文本/代码或超过 2 MiB 的图表。",
        "",
        "原始数据仍保存在本地 `/data_lib/cyt/workspace/history_exp/`；需要时可单独索取具体路径。",
        "",
        "## 文件统计",
        "",
        "| 目录 | 镜像文件数 | 镜像大小 | 原归档文件数 | 原归档大小 |",
        "|---|---:|---:|---:|---:|",
    ]
    all_keys = sorted(set(selected_stats) | set(archive))
    for key in all_keys:
        selected_count, selected_size = selected_stats.get(key, [0, 0])
        archive_count, archive_size = archive.get(key, (0, 0))
        lines.append(
            f"| `{key}` | {selected_count} | {human_size(selected_size)} | "
            f"{archive_count} | {human_size(archive_size)} |"
        )
    lines += [
        "",
        "## 校验",
        "",
        "文件哈希见仓库中的 `SHA256SUMS`。重新生成镜像可执行：",
        "",
        "```bash",
        "python3 tools/sync_history_snapshot.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"source archive not found: {SOURCE_ROOT}")
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
    TEMP_ROOT.mkdir(parents=True)

    selected = collect(SOURCE_ROOT)
    copied: list[tuple[Path, Path, int]] = []
    for source, rel, size in selected:
        target = TEMP_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
            copied.append((source, rel, size))
        except OSError as exc:
            print(f"SKIP {rel}: {exc}")

    archive = scan_archive(SOURCE_ROOT)
    manifest = build_manifest(copied, archive)
    (TEMP_ROOT / "MANIFEST.md").write_text(manifest, encoding="utf-8")

    source_readme = SOURCE_ROOT / "README.md"
    mirror_readme = TEMP_ROOT / "README.md"
    if source_readme.is_file():
        notice = (
            "> **远端精选镜像**：这里保留实验总结、重要文档、关键代码/补丁和精选结果。\n"
            "> 原始 `.pt`、权重、日志、数据集和大结果未提交到 Git，仍保存在本地完整归档中。\n"
            "> 收录范围与统计见 [`MANIFEST.md`](MANIFEST.md)。\n\n"
        )
        mirror_readme.write_text(
            notice + source_readme.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    sums = []
    for target in sorted(TEMP_ROOT.rglob("*")):
        if target.is_file() and target.name != "SHA256SUMS":
            sums.append(f"{sha256(target)}  {target.relative_to(TEMP_ROOT).as_posix()}")
    (TEMP_ROOT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    TEMP_ROOT.replace(DEST_ROOT)

    total_size = sum(size for _, _, size in copied)
    print(f"Built {DEST_ROOT}")
    print(f"Copied {len(copied)} files, {human_size(total_size)}")
    print(f"SHA256 entries: {len(sums)}")


if __name__ == "__main__":
    main()
