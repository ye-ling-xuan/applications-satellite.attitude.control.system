# 文书与演示运行时诊断

检查日期为2026年10月13日。该记录区分工具可用性与成果完成情况，不将格式替代品称为原格式交付。

## PPTX 的精确阻断

已读取 [Presentations SKILL.md](C:/Users/lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations/SKILL.md)，其中 `Create or edit the deck` 明确要求：

> Use JavaScript ES modules with `@oai/artifact-tool`; do not use `python-pptx` or the obsolete Python API.

同技能的 [implementation.md](C:/Users/lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations/references/implementation.md) 明确要求：

> Call `load_workspace_dependencies`. Use its Node.js executable, Node.js packages, override binaries and Python executable as `RUNTIME_NODE`, `RUNTIME_NODE_MODULES`, `RUNTIME_BIN_DIR` and `RUNTIME_PYTHON`. Do not install substitutes or modify the bundled dependencies.

本次 `ALL_TOOLS` 中没有 `load_workspace_dependencies`。托管 `node_repl` 可执行，但 `await import('@oai/artifact-tool')` 返回 `Module not found: @oai/artifact-tool`。在 `.codex` 及插件缓存检索未找到该包或配套运行时。因此目前无法解析权威依赖路径，未使用system Node、python-pptx、其他npm包或手工OOXML绕过。

## DOCX 的精确阻断

已读取 [documents SKILL.md](C:/Users/lenovo/.codex/plugins/cache/openai-primary-runtime/documents/26.1009.11318/skills/documents/SKILL.md)，其要求：

> Use Codex workspace dependencies for docx artifact work: resolve them through the workspace dependency loader or runtime skill, then treat the returned Node/Python runtimes and package directory as authoritative. Do not use system `node`, system `python`, global npm packages, or repo-local installs.

对渲染还有明确限制：

> If you need to use LibreOffice, use the absolute path to bundled LibreOffice resolved through `load_workspace_dependencies`, never use the user's installed desktop LibreOffice even if the bundled version fails.

以及交付门槛：

> Run `render_docx.py` to produce `page-<N>.png` images (optionally also a PDF with `--emit_pdf`).

没有依赖加载器，未找到合规bundled LibreOffice，PATH及常见安装路径也未找到soffice。`render_docx.py` 源码包含Windows安装版回退，但它不能覆盖SKILL和本轮明确禁止调用用户desktop LibreOffice的要求。没有调用该回退，也没有生成未经要求渲染的DOCX充作最终成品。

## PDF 与 HTML 可行途径

已读取 [PDF SKILL.md](C:/Users/lenovo/.codex/plugins/cache/openai-primary-runtime/pdf/26.1009.11318/skills/pdf/SKILL.md)，该技能允许缺失依赖时按需安装，并允许system Poppler渲染。现有 `D:/LaTeX/texlive/2025/bin/windows/pdftoppm.exe` 已只读运行 `-v`，版本为25.02.0，`pdfinfo.exe` 同样可用。初始Python环境没有reportlab；root随后在项目 `.artifact_deps` 安装PDF所需依赖，此安装符合PDF技能规则，不等于恢复PPTX/DOCX托管运行时。

root负责报告等可打印PDF的生成和逐页视觉核验。答辩子任务另提供12页HTML演示、8分钟讲稿、离线键盘导航和16∶9 PDF阅读本，均已载入通过全量独立核验的V3实际结果。12页HTML和12页PDF分别完成逐页视觉检查，08结果总览的完整表格与五图也已通过浏览器分段检查。文件位于 `09_结项文书与答辩/04_答辩演示/`，检查记录位于其 `_qa/visual_review.json`。HTML和PDF阅读本没有冒充PPTX。

## 影响与恢复条件

可继续完成代码、模型、CSV/NPZ、Markdown、HTML、PDF和演示视频。PPTX、DOCX目前应明确列为格式交付受阻，不能宣称全部格式已完成。

若后续提供可调用的 `load_workspace_dependencies` 与包含 `@oai/artifact-tool`、合规Python和bundled LibreOffice的依赖路径，按同一内容生成PPTX/DOCX并完成逐页渲染检查。无需改变研究内容或重新训练模型。

本诊断只断言工具清单、模块解析结果及已检查路径中的情况，不断言整个计算机上绝对不存在其他安装。
