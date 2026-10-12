"""Record the execution environment used for the delivered experiments."""
from importlib.metadata import version
from pathlib import Path
import json
import platform
import sys

root = Path(__file__).resolve().parents[1]
packages = ["numpy", "matplotlib", "torch", "gymnasium", "stable-baselines3", "pytest"]
versions = {name: version(name) for name in packages}
record = {"python": sys.version, "executable": sys.executable,
          "platform": platform.platform(), "packages": versions, "device": "cpu"}
(root / "02_配置与方法" / "environment.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "requirements-tested.txt").write_text(
    "# 实际验证使用 Python 3.12.4；torch 为 CPU 构建。\n" +
    "\n".join(f"{name}=={v}" for name, v in versions.items()) + "\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False))
