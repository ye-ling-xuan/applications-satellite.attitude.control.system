from pathlib import Path
import json

HERE = Path(__file__).resolve().parent

info = {
  "verification_date": "2026-10-13",
  "excluded_source": "大创报名材料/人工智能在卫星姿态控制系统中的应用-项目计划书.docx",
  "exclusion_reason": "用户明确要求忽略该计划书",
  "school": "哈尔滨工业大学（深圳）",
  "project_type": "大一年度项目",
  "project_title": "人工智能在卫星姿态控制系统中的应用",
  "leader": "刘绍轩",
  "members_in_original_order": [
    {"name": "刘绍轩", "student_id": "2025310898", "college_in_reports": "基础学部", "role": "负责人", "individual_contribution": None},
    {"name": "姚星宇", "student_id": "2025311243", "college_in_reports": "基础学部", "role": "成员", "individual_contribution": None},
    {"name": "谢斯捷", "student_id": "2025310504", "college_in_reports": "基础学部", "role": "成员", "individual_contribution": None},
    {"name": "陈茂菘", "student_id": "2025310535", "college_in_reports": "基础学部", "role": "成员", "individual_contribution": None}
  ],
  "advisor": {"name": "董瑞琦", "title_in_reports": "副教授", "college_in_reports": "智能学部/智科学院", "current_status_confirmed": False},
  "sources": {
    "proposal": {"path": "项目报告/《人工智能在卫星姿态控制系统中的应用》大一年度项目立项报告_20251026163255.pdf", "form_date": "2025-10-25", "pages": 15, "identity_pages_visual_checked": [2,10]},
    "midterm": {"path": "项目报告/中期检查报告.docx", "form_date": "2026-06-22", "read_method": "DOCX XML paragraphs and tables", "page_numbers_claimed": False}
  },
  "commitments": [
    {"id": "C01", "source": "proposal", "locator": "第10页4.1", "quote": "中期完成的卫星姿态系统搭建学习综述一篇。", "deliverable": "01_两篇学习综述/卫星姿态系统搭建学习综述.md", "status": "内容已编写待团队阅读与正式排版"},
    {"id": "C02", "source": "proposal", "locator": "第10至11页4.2", "quote": "结题完成卫星智能姿态控制系统学习综述一篇。", "deliverable": "01_两篇学习综述/卫星智能姿态控制系统学习综述.md", "status": "内容已编写待最终研究整合与正式排版"},
    {"id": "C03", "source": "proposal", "locator": "第8至9页实施方案", "quote": "最终筛选10-15篇易理解的文献", "status": "已检索核验公开原文摘要与官方资料不冒充团队全部读完"},
    {"id": "C04", "source": "proposal", "locator": "第6至8页", "item": "STK仿真与理论对比", "status": "未找到完成证据"},
    {"id": "C05", "source": "proposal", "locator": "第7页模型辨识", "item": "机器学习模型辨识与预测控制框架", "status": "未找到完成证据"},
    {"id": "C06", "source": "midterm", "locator": "二（三）结题预期目标", "item": "控制精度稳定性控制能量达到或优于PID", "status": "已有系统比较但不支持全面优于强PID"},
    {"id": "C07", "source": "midterm", "locator": "二（二）下一步工作计划", "item": "SAC TD3探索与更高效调参", "status": "SAC TD3已正式比较贝叶斯优化未完成"}
  ],
  "version_status_at_drafting": {
    "V1": {"status": "已完成", "source": "06_结项说明与报告/结项技术报告.md"},
    "V2": {"status": "已完成", "source": "07_单轴三轴多算法对比/新融合分析.md"},
    "V3": {"status": "研究执行中未导入最终结果", "source": "08_惯量域随机化与泛化边界研究/研究计划.md", "planned_models": 12, "planned_test_episodes": 10920, "actual_models": None, "actual_test_episodes": None, "result_conclusion": None}
  },
  "funding": {"proposal_planned_cny": 3000, "approved_cny": None, "midterm_spending_quote": "无。", "later_spending_cny": None},
  "schedule": {"proposal_expected_completion": "次年7月", "actual_school_deadline": None, "defense_minutes": None, "current_school_template": None},
  "forbidden_inferences": ["不能从文件夹命名推断个人贡献", "不能把域随机化称在线辨识或自适应", "不能把部署校准称新增训练算法", "不能把控制代价称电能实测", "不能捏造导师批准范围变更", "不能捏造会议记录签字经费或优秀等级"],
  "pending_user_fields": ["当前学校模板", "实际截止时间", "答辩时长", "项目编号", "现学院信息", "真实成员贡献与本人已读文献", "后续经费和票据", "导师意见与签字"]
}
(HERE / "原立项承诺与信息.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
print("Original commitments JSON saved")
