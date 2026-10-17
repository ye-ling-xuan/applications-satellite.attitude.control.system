"""Render the authored overview only; never changes experiments or selected inputs."""
from pathlib import Path
import csv
import json
import re
import markdown

ROOT = Path(__file__).resolve().parent
body = markdown.markdown((ROOT/"本轮工作说明.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
style = "body{font-family:'Microsoft YaHei',sans-serif;max-width:1100px;margin:32px auto;padding:0 22px;line-height:1.9;color:#263447}h1,h2{line-height:1.45}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #cad3dd;text-align:left}th{background:#edf3f8}img{width:100%;height:auto}a{color:#146493}strong{color:#172d48}"
page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本轮研究结果与说明</title><style>'+style+'</style></head><body>'+body+'</body></html>'
(ROOT/"结果总览.html").write_text(page, encoding="utf-8")
# Editorial corrections affect reading copies only, never frozen source/data.
analysis = ROOT/"05_图表与分析"
detail = analysis/"研究结果说明.md"
if detail.is_file():
    text = detail.read_text(encoding="utf-8")
    text = re.sub(r"(\d+)个完整回合", r"\1条评估记录（含物理失败后冻结并补齐时钟的记录）", text)
    text = text.replace("控制代价和RMS包含全部完整回合", "控制代价和RMS包含全部评估记录")
    text = text.replace("条件。\n- 单轴", "条件。\n\n- 单轴")
    note = "本轮最终选中的PID积分增益均为0，实际为PD控制；结果表沿用PID对照名称，不代表用到了积分作用。\n\n"
    if note not in text:
        text = text.replace("## 为什么这次比较与旧结果不同\n\n", "## 为什么这次比较与旧结果不同\n\n"+note)
    with (ROOT/"04_独立测试"/"episode_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    physical = sum(r["failed"] == "True" for r in rows)
    failure_note = (f"本轮全部评估记录中有{physical}条发生物理失败（不同控制器复用初态，按评估回合计数）。失败发生后状态冻结、后续力矩为0，时钟补齐到20秒；填充段不表示卫星仍在正常物理运动。首次失败所在区间的实际力矩仍计入代价。失败策略的低代价不能作性能优势。\n\n")
    if failure_note not in text:
        text = text.replace("## 完整结果与统计限制\n\n", "## 完整结果与统计限制\n\n"+failure_note)
    detail.write_text(text, encoding="utf-8")
    rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
    (analysis/"结果阅读本.html").write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>详细研究结果</title><style>'+style+'</style></head><body>'+rendered+'</body></html>', encoding="utf-8")
    (ROOT/"06_核验"/"阅读本文字更正记录.json").write_text(json.dumps({
        "changes": ["区分20秒评估记录与物理失败后的冻结填充", "明确最终Ki为0", "加入物理失败回合计数", "修正列表空行"],
        "physical_failure_episode_count": physical,
        "controller_model_code_or_test_data_changed": False,
        "reason": "independent result-text review after evaluation; no retuning or additional training",
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(ROOT/"结果总览.html")
