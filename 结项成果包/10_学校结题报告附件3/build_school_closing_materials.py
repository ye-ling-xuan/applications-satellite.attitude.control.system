"""Prepare template-mapped text and a reading PDF; never edit the source DOCX.

The school template was inspected read-only using PowerShell/.NET. This tool
authors text/PDF only: it does not create or render a filled Word form.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parent
CONTENT = HERE / "01_可粘贴正文"
EVIDENCE = HERE / "02_模板证据与核验"
PDF_DIR = HERE / "03_正文阅读本"
RENDER = ROOT / "tmp/artifacts/school_template_pdf_review"
SOURCE = PROJECT / "附件3：大一年度项目结题报告.docx"
TEMPLATE = HERE / "00_原始模板" / SOURCE.name
SOURCE_SHA = "a3d5eeb6941cbe2165bf60f44ac9870b2cfac197bf2563bda713c249c7f2cd72"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".artifact_deps"))
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def plain(text):
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    return text.replace("**", "").replace("`", "")


def counts(text):
    visible = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    visible = re.sub(r"^#{1,6}\s+", "", visible, flags=re.M).replace("**", "").replace("`", "")
    return {"visible_text_non_whitespace_characters": len(re.sub(r"\s", "", visible)),
            "plain_text_with_link_urls_non_whitespace_characters": len(re.sub(r"\s", "", plain(text))),
            "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
            "count_note": "可见文本计数包含标点、英文、数字与参考文献，排除Markdown和链接目的地；纯文本另含可查询URL。汉字数仅汉字。学校未给出具体字数软件或精确上下限，不将字符数冒称Word字数。"}


def verify_sources():
    for path in (SOURCE, TEMPLATE):
        if sha(path) != SOURCE_SHA:
            raise ValueError("School template changed: " + str(path))


def prepare():
    verify_sources()
    original = read(ROOT / "09_结项文书与答辩/05_来源与核验/原立项承诺与信息.json")
    members = original["members_in_original_order"]
    member_phones = ["17701020096", "18875867919", "14707549107", "14775390505"]
    fields = {
        "prepared_client_date": "2026-10-14",
        "source_template_sha256": SOURCE_SHA,
        "historical_identity_source": "项目报告/中期检查报告.docx",
        "historical_identity_source_sha256": sha(PROJECT / "项目报告/中期检查报告.docx"),
        "historical_form_date": "2026-06-22",
        "read_method": "Root read-only .NET ZipFile/XML inspection of cover and first member table; names and IDs also cross-checked with previously verified identity JSON",
        "historical_contact_college_title_reuse_confirmed_by_user": True,
        "user_confirmation_record": "02_模板证据与核验/用户确认记录.json",
        "user_confirmation_record_sha256": sha(EVIDENCE / "用户确认记录.json"),
        "unknown_fields_left_blank": True,
        "all_personal_fields_complete": False,
        "project_title": original["project_title"],
        "leader": original["leader"],
        "leader_student_id": members[0]["student_id"],
        "leader_phone_in_midterm": member_phones[0],
        "leader_email_in_midterm": "1430522866@qq.com",
        "leader_college_in_midterm": "基础学部",
        "leader_current_college": "基础学部",
        "leader_major": None,
        "advisor_name": original["advisor"]["name"],
        "advisor_title_in_midterm": "副教授",
        "advisor_phone_in_midterm": "15646515687",
        "advisor_email_in_midterm": "rykidong@163.com",
        "advisor_college_in_midterm": "智能学部/智科学院",
        "advisor_current_college": "智能学部/智科学院",
        "advisor_major": None,
        "actual_form_date": None,
        "members": [{"name": m["name"], "gender_in_midterm": "男", "college_in_midterm": m["college_in_reports"],
                     "student_id": m["student_id"], "phone_in_midterm": phone,
                     "current_college_confirmed": True, "phone_reuse_confirmed": True, "signature": None}
                    for m, phone in zip(members, member_phones, strict=True)],
        "fifth_member_row": "leave blank; original reports have four members",
        "advisor_opinion": None, "expert_opinion": None,
        "section_six_status": "user plans to apply; separate future direction and expected outcomes prepared, not current completed work",
        "continuation_intention_confirmed_by_user": True,
        "future_plan_text_sha256": sha(CONTENT / "后续大创工作方向与预期成果.md"),
        "template_contains_project_number_field": False,
        "template_contains_funding_field": False,
        "native_filled_docx_produced": False,
    }
    save(CONTENT / "回填字段.json", fields)
    ref = ["# 附件3封面及成员回填参考", "", "信息取自原立项与2026年6月22日中期报告。用户已确认沿用中期报告的联系方式、学院与教师职称，并要求不确定项留空。下列院系及专业栏仅填已知院系；专业、实际填表日期、签字与意见均空白。", "",
           "| 封面字段 | 回填值 |", "|---|---|"]
    for label, value in [
        ("项目名称", fields["project_title"]), ("负责人", fields["leader"]),
        ("负责人学号", fields["leader_student_id"]), ("负责人联系电话", fields["leader_phone_in_midterm"]),
        ("负责人电子邮箱", fields["leader_email_in_midterm"]),
        ("负责人院系及专业", fields["leader_current_college"]),
        ("指导教师", fields["advisor_name"]), ("指导教师职称", fields["advisor_title_in_midterm"]),
        ("教师联系电话", fields["advisor_phone_in_midterm"]), ("教师电子邮箱", fields["advisor_email_in_midterm"]),
        ("教师院系及专业", fields["advisor_current_college"]),
        ("填表日期", "")]:
        ref.append(f"| {label} | {value} |")
    ref += ["", "## 一 项目团队成员", "", "按原件顺序列四人，第五个空成员位保留为空；学院与电话按用户确认沿用中期原件值，本人签字空白。", "",
            "| 姓名 | 性别 | 所在学院 | 学号 | 联系电话 | 本人签字 |", "|---|---|---|---|---|---|"]
    for member in fields["members"]:
        ref.append("| " + " | ".join([member["name"], member["gender_in_midterm"], member["college_in_midterm"], member["student_id"], member["phone_in_midterm"], ""]) + " |")
    ref.append("|  |  |  |  |  |  |")
    ref += ["", "## 二 指导教师意见", "", "留空，由指导教师本人填写意见、签名和日期。", "",
            "## 三 项目专家组意见", "", "留空，由实际评审专家填写意见、组长签名和日期。", "",
            "## 六 拟依托本项目继续开展大创计划", "", "用户已确认计划申请。两小项正文见《后续大创工作方向与预期成果》；该方向另写，未使用此前被用户排除的大创报名计划书。它属于未来拟开展工作，不计入本次结题成果。", ""]
    (CONTENT / "封面及成员回填参考.md").write_text("\n".join(ref), encoding="utf-8")
    report = (CONTENT / "项目研究结题报告.md").read_text(encoding="utf-8")
    achievements = (CONTENT / "项目成果.md").read_text(encoding="utf-8")
    future = (CONTENT / "后续大创工作方向与预期成果.md").read_text(encoding="utf-8")
    for name, text in [("项目成果", achievements), ("项目研究结题报告", report), ("后续大创工作方向与预期成果", future)]:
        (CONTENT / (name + ".txt")).write_text(plain(text), encoding="utf-8-sig")
    combined = ("# 附件3结题报告填报稿\n\n编制日期2026年10月14日，此日期不是实际填表日期。此为按学校栏目组织的可编辑填报文字，尚未回填或渲染原生Word表格。已知联系方式、学院与教师职称按用户确认沿用中期报告；不确定项空白。下文包含回填工作说明，教师和专家意见及签字由当事人填写。\n\n"
                + "\n".join(ref[2:]) + "\n\n## 四 项目成果\n\n" + achievements
                + "\n\n## 五 项目研究结题报告\n\n" + re.sub(r"^# ", "### ", report, flags=re.M))
    # Keep school order: sixth optional section comes after the report.
    sixth = "## 六 拟依托本项目继续开展大创计划"
    start = combined.index(sixth)
    end = combined.index("## 四 项目成果", start)
    optional = combined[start:end].strip()
    future_body = re.sub(r"\A# [^\n]+\n+", "", future)
    future_body = re.sub(r"^## ", "### ", future_body, flags=re.M)
    combined = combined[:start] + combined[end:] + "\n\n" + optional + "\n\n" + future_body
    (HERE / "附件3填报稿.md").write_text(combined, encoding="utf-8")
    headings = re.findall(r"^# (.+)$", report, flags=re.M)
    required = ["1 课题背景", "2 课题研究内容与方法", "3 研究结果", "4 创新点", "5 结束语", "6 参考文献"]
    if headings != required:
        raise ValueError((headings, required))
    pdf_source = ("# 人工智能在卫星姿态控制系统中的应用\n\n附件3第四与第五部分正文阅读本。供审阅文字使用，原生学校Word表格尚未回填及渲染。\n\n"
                  + "## 四 项目成果\n\n" + achievements + "\n\n## 五 项目研究结题报告\n\n"
                  + re.sub(r"^# ", "### ", report, flags=re.M))
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    (PDF_DIR / "附件3正文阅读本.md").write_text(pdf_source, encoding="utf-8")
    save(EVIDENCE / "正文覆盖与字数.json", {
        "client_date": "2026-10-14", "template_sha256": SOURCE_SHA,
        "status": "text_prepared_native_Word_pending", "fifth_section_headings": headings,
        "six_headings_match_template": True, "school_length_instruction": "字数应在3000字左右",
        "fifth_section": counts(report), "fourth_section": counts(achievements),
        "source_text_sha256": sha(CONTENT / "项目研究结题报告.md"),
        "achievement_text_sha256": sha(CONTENT / "项目成果.md"),
        "source_template_preserved": True, "source_copy_identical": True,
        "optional_sixth_section_filled": True, "sixth_section_is_future_work": True,
        "future_plan_text_sha256": sha(CONTENT / "后续大创工作方向与预期成果.md"),
        "sixth_section": counts(future),
        "user_confirmation_record_sha256": sha(EVIDENCE / "用户确认记录.json"),
        "historical_contact_college_title_reuse_confirmed_by_user": True,
        "unknown_fields_left_blank": True, "personal_confirmation_inferred": False,
        "no_new_research": True, "new_template_received": True,
    })
    print(json.dumps({"status": "prepared", "report": counts(report), "achievements": counts(achievements)}, ensure_ascii=False))


def build_pdf():
    verify_sources()
    from tools.build_closing_pdfs import build
    item = build(PDF_DIR / "附件3正文阅读本.md", PDF_DIR / "附件3正文阅读本.pdf")
    save(PDF_DIR / "生成清单.json", [item])
    print(json.dumps(item, ensure_ascii=False))


def render():
    from pypdf import PdfReader
    pdf = PDF_DIR / "附件3正文阅读本.pdf"
    RENDER.mkdir(parents=True, exist_ok=True)
    subprocess.run(["D:/LaTeX/texlive/2025/bin/windows/pdftoppm.exe", "-r", "110", "-png", str(pdf), str(RENDER / "page")], check=True, capture_output=True)
    pages = [RENDER / f"page-{i}.png" for i in range(1, len(PdfReader(pdf).pages) + 1)]
    save(RENDER / "render_manifest.json", {"pdf_sha256": sha(pdf), "pages": [{"relative": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in pages], "visual_review": "pending"})
    print(json.dumps({"pages": len(pages), "render_directory": str(RENDER)}, ensure_ascii=False))


def record_pass():
    items = read(PDF_DIR / "生成清单.json")
    rendering = read(RENDER / "render_manifest.json")
    item = items[0]
    assert sha(ROOT / item["source"]) == item["source_sha256"]
    assert sha(ROOT / item["output"]) == item["pdf_sha256"] == rendering["pdf_sha256"]
    for page in rendering["pages"]:
        assert sha(ROOT / page["relative"]) == page["sha256"]
    item["visual_qa"] = "passed"
    rendering.update({"status": "passed", "visual_review": "passed", "reviewer": "root", "reviewed_pages": list(range(1, len(rendering["pages"]) + 1)),
                      "scope": "All reading-copy PDF pages opened and checked; this does not render or validate the source or a filled DOCX form"})
    save(PDF_DIR / "生成清单.json", items)
    save(PDF_DIR / "visual_review.json", rendering)
    print("Recorded full reading-PDF visual review")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "build-pdf", "render", "record-visual-pass"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.build_pdf:
        build_pdf()
    elif args.render:
        render()
    else:
        record_pass()


if __name__ == "__main__":
    main()
