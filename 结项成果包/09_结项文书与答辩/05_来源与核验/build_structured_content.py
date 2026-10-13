"""Build structured companion files without changing reports or research data."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

source_files = [HERE / "文献获取清单.json", HERE / "补充文献获取清单.json"]
retrieval = {}
for path in source_files:
    for row in json.loads(path.read_text(encoding="utf-8")):
        retrieval[row["id"]] = row
row = json.loads((HERE / "运动学来源获取.json").read_text(encoding="utf-8"))
retrieval[row["id"]] = row

metadata = [
 ("N01", "NASA", None, "Guidance Navigation and Control", "official_chapter_sections", ["ADCS构成", "反作用轮动量管理", "任务差异"], None),
 ("R04", "Autonomous Vehicle Systems Laboratory", None, "Basilisk RigidBodyKinematics", "official_api_relevant_functions", ["姿态参数与运动学函数", "本文公式按本项目约定推导"], None),
 ("R03", "Autonomous Vehicle Systems Laboratory", None, "Basilisk Module spacecraft", "official_module_overview", ["6DOF与效应器划分"], None),
 ("R02", "NI", None, "PID Theory Explained", "official_article_body", ["PID三项作用", "整定"], None),
 ("S02", "Ke Xu; Fengge Wu; Junsuo Zhao", 2019, "Model-based deep reinforcement learning with heuristic search for satellite attitude control", "crossref_abstract_and_metadata", ["卫星智能控制可采用基于模型学习路径"], "10.1108/IR-05-2018-0086"),
 ("D01", "Farama Foundation", None, "Gymnasium Env API", "official_api_body", ["reset/step接口"], None),
 ("D02", "Farama Foundation", None, "Handling Time Limits", "official_tutorial_body", ["terminated/truncated与价值自举"], None),
 ("D03", "Stable-Baselines3 Contributors", None, "Reinforcement Learning Tips and Tricks", "official_documentation_body", ["多种子", "测试隔离", "动作缩放", "包装器"], None),
 ("A01", "John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov", 2017, "Proximal Policy Optimization Algorithms", "abstract_and_full_paper_relevant_sections", ["PPO裁剪目标", "on-policy采样"], None),
 ("A02", "Tuomas Haarnoja; Aurick Zhou; Pieter Abbeel; Sergey Levine", 2018, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", "abstract_and_full_paper_relevant_sections", ["最大熵目标", "soft policy iteration", "函数近似边界"], None),
 ("D04", "Stable-Baselines3 Contributors", None, "SAC Documentation", "official_documentation_body", ["当前实现与原始论文差别", "熵系数", "推理设置"], None),
 ("A03", "Scott Fujimoto; Herke van Hoof; David Meger", 2018, "Addressing Function Approximation Error in Actor-Critic Methods", "abstract_and_full_paper_method_sections", ["双Q", "延迟更新", "目标策略平滑"], None),
 ("P01", "Antonin Raffin; Ashley Hill; Adam Gleave; Anssi Kanervisto; Maximilian Ernestus; Noah Dormann", 2021, "Stable-Baselines3: Reliable Reinforcement Learning Implementations", "journal_abstract_and_metadata", ["SB3共同接口与测试背景"], None),
 ("A04", "Josh Tobin; Rachel Fong; Alex Ray; Jonas Schneider; Wojciech Zaremba; Pieter Abbeel", 2017, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", "abstract_and_full_paper_relevant_sections", ["视觉域随机化", "非卫星部署证据"], None),
 ("A05", "Xue Bin Peng; Marcin Andrychowicz; Wojciech Zaremba; Pieter Abbeel", 2017, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", "abstract_and_full_paper_relevant_sections", ["动力学抽样", "回合固定参数", "记忆策略"], None),
 ("A06", "Wenshuai Zhao; Jorge Peña Queralta; Tomi Westerlund", 2020, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey", "abstract_only", ["迁移方法分类"], None),
 ("S01", "Wiebke Retagne; Jonas Dauer; Günther Waxenegger-Wilfing", 2024, "Adaptive satellite attitude control for varying masses using deep reinforcement learning", "original_article_selected_theory_method_results_limitations", ["历史观测", "Basilisk", "反作用轮", "与本项目瞬时观测区别"], "10.3389/frobt.2024.1402846"),
]
references = []
for sid, authors, year, title, level, claims, doi in metadata:
    rr = retrieval[sid]
    assert rr["verified"], sid
    cache_path = HERE / rr["cache_file"]
    assert cache_path.exists(), cache_path
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == rr["sha256"]
    references.append({"id": sid, "authors": authors, "year": year, "title": title, "doi": doi,
      "url": rr["requested_url"], "retrieved_date": rr["retrieved_date"], "reading_level": level,
      "supports": claims, "cache_file": rr["cache_file"], "sha256": rr["sha256"],
      "does_not_imply_team_members_read": True})
(HERE / "正文文献结构化清单.json").write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding="utf-8")

slides_path = PACKAGE / "03_答辩内容" / "答辩大纲与逐页讲稿.md"
source = slides_path.read_text(encoding="utf-8")
parts = re.split(r"(?m)^## 第(\d+)页 (.+)\n", source)
durations = [15,30,35,45,45,50,40,50,70,35,40,25]
slides = []
for i in range(1,len(parts),3):
    number,title,body = int(parts[i]),parts[i+1],parts[i+2]
    if number==12:
        body=body.split("\n## 备份页建议")[0]
    view=re.search(r"\*\*画面内容\*\*：(.*?)(?:\n\n|$)",body,re.S).group(1).strip()
    notes=re.search(r"\*\*讲稿(?:框架)? \d+秒\*\*：(.*)",body,re.S).group(1).strip()
    slides.append({"number":number,"title":title,"display_content":view,"speaker_notes":notes,"suggested_seconds":durations[number-1],"status":"pending_v3_final_results" if number==9 else "draft_content_complete"})
assert len(slides)==12
assert sum(durations)==480
deck={"project_title":"人工智能在卫星姿态控制系统中的应用","suggested_minutes":8,"school_time_confirmed":False,
 "members":["刘绍轩","姚星宇","谢斯捷","陈茂菘"],"advisor":"董瑞琦 副教授","slides":slides,
 "final_data_pending":["08最终研究实际数量与核验状态","第9页惯量泛化曲线与边界","第12页最终研究结论"],
 "forbidden_claims":["AI全面优于PID","在线自适应或已辨识惯量","实际节能百分比","已完成STK与硬件在轨","已获学校优秀"]}
(PACKAGE / "03_答辩内容" / "答辩内容.json").write_text(json.dumps(deck,ensure_ascii=False,indent=2),encoding="utf-8")

checks=[]
for path in (PACKAGE / "01_两篇学习综述").glob("*.md"):
    text=path.read_text(encoding="utf-8")
    refs=text.split("## 参考文献",1)
    assert len(refs)==2
    body,bibliography=refs
    cited={int(n) for n in re.findall(r"\[(\d+)\]",body)}
    listed={int(n) for n in re.findall(r"(?m)^\[(\d+)\]",bibliography)}
    assert cited <= listed,(path,cited-listed)
    checks.append({"path":str(path.relative_to(PACKAGE)),"characters":len(text),"body_cited":sorted(cited),"bibliography_count":len(listed),"citation_numbers_match":cited==listed})
(HERE / "内容静态核验.json").write_text(json.dumps({"references_cached_and_hashed":len(references),"slides":len(slides),"duration_seconds":sum(durations),"reviews":checks,"formal_docx_pptx_rendered":False,"V3_final_integrated":False},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(checks,ensure_ascii=False))
