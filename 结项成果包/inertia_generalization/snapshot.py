"""Read-only SHA-256 preservation baseline for all preexisting V1/V2 assets."""
from pathlib import Path
import json
from .protocol import ROOT,CONFIG_DIR,sha256,write_json


def capture_prior_assets():
    target=CONFIG_DIR/"prior_assets_snapshot.json"
    if target.exists():return json.loads(target.read_text(encoding="utf-8"))
    names=("01_遴选原始参考","02_配置与方法","03_训练模型与日志","04_实验数据与图表",
           "05_三轴拓展演示","06_结项说明与报告","07_单轴三轴多算法对比",
           "satellite_benchmark","multi_algorithm_benchmark")
    rows=[]
    for name in names:
        for path in sorted((ROOT/name).rglob("*")):
            if path.is_file() and not any(p in {"__pycache__",".pytest_cache"} for p in path.parts):
                rows.append({"relative":path.relative_to(ROOT).as_posix(),"sha256":sha256(path),"bytes":path.stat().st_size})
    snapshot={"scope":"V1/V2 numbered directories and implementation modules, excluding caches",
              "file_count":len(rows),"files":rows}
    write_json(target,snapshot);return snapshot
