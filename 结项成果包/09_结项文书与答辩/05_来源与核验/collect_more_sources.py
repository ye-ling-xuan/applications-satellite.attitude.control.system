from collect_sources import fetch, HERE
from concurrent.futures import ThreadPoolExecutor
import json

sources = [
    ("A01F", "PPO full paper", "https://arxiv.org/pdf/1707.06347"),
    ("A02F", "SAC full paper", "https://arxiv.org/pdf/1801.01290"),
    ("A03F", "TD3 full paper", "https://arxiv.org/pdf/1802.09477"),
    ("A04F", "Domain randomization full paper", "https://arxiv.org/pdf/1703.06907"),
    ("A05F", "Dynamics randomization full paper", "https://arxiv.org/pdf/1710.06537"),
    ("S01", "Retagne satellite adaptive RL original article", "https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1402846/full"),
    ("S02", "Xu model based satellite RL Crossref abstract", "https://api.crossref.org/works/10.1108/IR-05-2018-0086"),
    ("R01", "JPL NAIF rotations required reading", "https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html"),
    ("R02", "NI PID theory explained", "https://www.ni.com/en/shop/labview/pid-theory-explained.html"),
    ("R03", "Basilisk spacecraft dynamics official documentation", "https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/spacecraft/spacecraft.html"),
    ("M01", "Control neural networks original Crossref abstract", "https://api.crossref.org/works/10.1109/72.80202"),
]

with ThreadPoolExecutor(max_workers=6) as pool:
    records = list(pool.map(fetch, sources))
(HERE / "补充文献获取清单.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
for r in records:
    print(r["id"], r["verified"], r.get("status", ""), r.get("bytes", ""), r.get("error", ""))
