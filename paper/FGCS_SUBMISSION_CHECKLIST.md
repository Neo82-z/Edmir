# FGCS 首投记录与投稿检查表

更新：2026-10-25（Asia/Shanghai）。用户已确定 **Future Generation Computer Systems（FGCS）为首投目标**。本文是本项目的投稿记录；普通研究论文为当前准备路线，未选择专刊、未提交稿件。

研究范围继续限定为八卡算子验证和一个真实 MoE 模型的八卡推理对照。实验完成条件见 [P8_PUBLICATION_PLAN.md](P8_PUBLICATION_PLAN.md)、[机制证据设计](P8_MECHANISM_EVIDENCE_PLAN.md)和 [MoE 收尾方案](P8_MOE_CLOSEOUT_PLAN.md)。选择期刊不意味着现稿已经具备投稿条件。

## 1. 核验范围与来源等级

本轮直接核实了 Elsevier 的期刊范围、LaTeX、AI、相似度检查、作者伦理、图件和 Highlights 等公开政策，以及 FGCS 已发表文章中的双通讯作者实例。FGCS 当前 Guide for Authors、开放获取页面返回 HTTP 403，浏览器读取也未成功；没有登录投稿系统。

以下使用四类标记，防止把准备建议当成期刊硬性条款：

- **已核实／通用**：Elsevier 当前公开政策；FGCS 专门规则仍优先。
- **已核实／专刊**：FGCS 编辑部发布的具体专刊要求，不能自动推广到普通稿件。
- **待官网复核**：近期第三方转录或旧版指南中的信息，保留线索，不宣称已读到当前原文。
- **项目决定**：我们为投稿质量与可复现性设定的要求，不是期刊规定的实验数量、卡型或性能门槛。

正式上传前重读 [FGCS 作者指南][S1]，逐项确认下表的待复核项，并记录日期。2014 年 Author Information Pack 的搜索抓取日期不是规则生效日期，不能用它覆盖当前指南。

## 2. 期刊范围与稿件定位

**已核实／期刊范围**：官方范围包含高性能／高吞吐计算方法、分布式资源管理、程序与算法设计、大规模通信与计算、扩展性与性能理论。[S2]

**项目决定**：按普通原创研究论文准备，文章类型暂以 Full Length Article 为候选，最终以投稿系统的实际选项为准。论文围绕“直接读取映射的执行代价边界及其候选选择价值”组织；MoE 实验检验应用收益。不能把 GPU 数量、旧数据拟合成绩或修复实现错误单独包装成方法创新。

投稿信需说明与并行／分布式系统性能研究的关系，交代与 Ernst/Warpspeed、FLUX、TileLink、COMET 等工作的具体区别。当前贡献判断见 [机制证据设计](P8_MECHANISM_EVIDENCE_PLAN.md)，不另开新的实验方向。

## 3. 格式、篇幅与材料

| 项目 | 当前依据与准备要求 | 状态 |
|---|---|---|
| 正文篇幅 | 编辑部 2025 年专刊公告要求双栏、单倍行距、最多 18 页，含图表和参考文献。近期第三方转录称 Full Length Article 也适用。我们按该上限准备；普通稿的现行规则、附录是否计入及例外仍待核实 | 专刊已核实；普通稿待复核 [S3][S4] |
| LaTeX 模板 | 上述公告推荐 `\documentclass[5p,times]{elsarticle}`。从 Elsevier 官方模板入口取得模板；当前 MLSys 样式需要迁移 | 专刊已核实；项目采用此准备路线 [S3][S5] |
| 摘要 | 近期转录为不超过 250 个英文词；先按这一上限编辑，摘要中的收益须来自已完成测量 | 待官网复核 [S4] |
| 关键词 | 近期转录为 6–10 个；旧版指南为最多 6 个。本项目暂准备 6 个，正式上传时确认当前数量 | 待官网复核 [S4] |
| 参考文献 | 近期转录为按引用顺序编号；迁移模板时更换 MLSys 参考文献样式并重新核对文内编号 | 待官网复核 [S4] |
| Highlights | 预先准备单独的 Word 文件，3–5 条，每条不超过 85 个字符，包含空格。通用页面说明最迟在最终文件阶段提交；FGCS 是否初投即必交仍需确认 | 数量／长度／文件格式为已核实通用要求 [S6] |
| 图文摘要 | 本轮未确认 FGCS 是否要求；暂不增加该制作项，提交前查看系统必填项 | 待官网复核 |
| 图表 | 性能图优先保留矢量 PDF/EPS、嵌入字体；表格保留可编辑源。若使用位图，按图件种类检查分辨率，不以截图代替原始绘图输出 | 已核实通用格式；项目交付要求 [S7] |
| 源文件 | 准备 PDF 和可独立构建的 `.tex/.bib/.bbl/.bst`、图件及非标准样式；具体上传类型遵循系统提示 | 已核实通用要求 [S5] |
| 目录结构 | Elsevier LaTeX 指南明确 EM 不处理含子目录的 LaTeX 投稿。另建同层目录导出包并改写引用路径；开发仓库保留现有结构 | 已核实通用要求 [S5] |
| 作者信息 | 收集完整姓名、单位、邮箱和通讯信息，预先确认署名顺序与贡献。ORCID 的绑定／可见性要求有转录线索，但具体范围未核实 | 作者责任已核实；ORCID 细则待核实 [S4][S8] |
| 审稿匿名性 | 近期转录为单匿名评审；上传时以当前指南为准，不直接沿用会议匿名稿的封面与声明 | 待官网复核 [S4] |

本轮只记录要求，不迁移模板、不生成待填作者信息、不改变论文中的实验结论。

## 4. 作者、声明与学术诚信

### 两位通讯作者

**已核实／出版实例**：FGCS 的《A terminology for scientific workflow systems》（2026）在首页同时标注 Frédéric Suter 和 Tainã Coleman 为通讯作者；Elsevier 支持文档也明确一篇文章可以有一位或多位通讯作者。[S9][S10]

**项目准备**：可按两位通讯作者制作标题页、脚注和邮箱。上传系统若只指定一位流程联系人，另在投稿信说明两位出版通讯作者，并在校样核对两人标记。出版署名与学校考核认定分别确认；未找到可据以宣称“最多两位”的当前规则。

### 必须如实准备的声明

- 所有作者认可最终稿和提交行为，作者名单与顺序在首次投稿前确定；后续修改需按编辑流程处理。不得同时向多刊提交同一稿件。[S8]
- 利益冲突由全体作者核实，有或没有均如实填写；准备资助信息和资助方作用，不能替作者虚构“无资助”或“无冲突”。按系统要求使用 [Elsevier 声明工具][S11]。
- 预先准备 CRediT 贡献表，按真实工作填写 Conceptualization、Methodology、Software、Validation、Writing 等角色；FGCS 的具体文件位置／必交阶段待确认。[S12]
- 说明相关预印本、会议论文或其他重叠版本。2025 年专刊公告中的“会议扩展至少 40% 新内容”只作为该公告要求记录，不当作普通稿通用标准，也不是查重阈值。[S3]

### AI 使用与相似度

**已核实／当前 AI 政策**：投稿时披露实质性的 AI 辅助写作，说明工具、用途和人工审核；研究中的 AI 辅助编程、分析应在 Methods 说明。AI 不能署名。AI 辅助解释图需在图注和声明披露，数据图必须可由真实数据和可复现方法生成；图文摘要不得使用通用生成式 AI 工具生成。[S13]

**项目执行**：整理本项目实际使用工具／版本、涉及的文字与代码、人工审查和验证记录；本项目用途已经超过语言润色，最终声明不能缩写成仅校对语法。所有性能图保持“原始数据→分析脚本→图表”的可追踪关系。

**没有核实到 FGCS 公开统一的查重率或 AIGC 检测百分比上限。** 不把 10%、15%、20% 等数字写成官方门槛。Elsevier 的相似度指南要求结合匹配来源判断，分数本身不能直接判定抄袭。[S14] 本项目逐段核对来源、引用和实际贡献，文字修订以准确清晰为目的。

## 5. 数据、代码与研究证据

**数据政策状态**：近期 FGCS 指南转录要求投稿时提供数据可用性说明。Elsevier 通用数据政策分不同档次，不能据通用页面就断言 FGCS 必须无条件公开所有代码，或允许只写“按需提供”。FGCS 的具体数据档次及系统字段仍待官网复核。[S4][S15]

**本项目交付决定**：准备可供审阅、可定位到版本的代码与数据发布包；以下是我们的可靠性要求，不是期刊指定的硬件或通过线：

| 交付内容 | 验收依据 |
|---|---|
| 代码版本与环境 | commit／源码包哈希、构建命令、驱动与库版本、GPU 拓扑、二进制和执行策略身份 |
| 原始测量 | 保留重复样本、正确性结果、计时边界与异常原因；主计时与 profiler 数据分开 |
| 冻结预测与独立评价 | 开发集、诊断集、最终评价集分别标注；已看过的历史形状不得称为未见测试 |
| 强基线和选择预算 | 逐源／合并 host 身份清楚；同预算比较；报告实际选择开销和摊销条件 |
| 机制证据 | 对应 H1–H4；逻辑字节、物理事务、拟合系数、实测等待分别解释 |
| 真实 MoE 验证 | 模型／引擎版本、集成位置、精度与负载、有效输出口径和端到端结果 |
| 图表复现 | 每个关键数字能追到原始记录；失败案例和不支持假设的结果同样保留 |
| 可用性声明 | 提供实际可访问的版本链接；需要长期归档时再制作 DOI 版本；限制与许可证如实说明 |

主张边界：P8 历史 19/19 命中属于回顾性开发成绩；旧弱 host 下的选择改进不等于超过新强基线。若新结果不支持加速主张，收窄结论；不得删除失败标签或持续用最终评价集调参直至“胜出”。

## 6. 当前稿件差距与后续顺序

本轮检查了相邻论文仓库的 `latex/main.tex`、`latex/README.md` 和 `latex/review/execution-revision-delivery.json`。

| 项目 | 当前状态 | 后续动作 |
|---|---|---|
| 首投选择 | FGCS 已确定；旧计划仍写 JPDC 优先 | 本轮已更新计划入口 |
| 模板 | `article` + `mlsys2025`，带研究初稿提示 | 迁移到 Elsevier 模板后再检查篇幅 |
| 页数 | 上次交付记录为 16 页 MLSys 稿 | 不能推定换模板后仍为 16 页；重新编译统计正文、参考文献和附录 |
| 作者信息 | 正文仅显示 Neo，单位及双通讯标记未建立 | 待作者团队确认信息后填入 |
| 八卡证据 | 新运行时、同批强基线、独立评价尚未完成 | 按现有 P8 计划完成 |
| MoE 证据 | 真实集成与端到端推理收益尚未完成 | 按单模型、单引擎方案完成 |
| 投稿附件 | 未形成 FGCS 专用完整材料包 | 补 Highlights、投稿信、各项声明及数据入口 |
| 投稿源码包 | 当前研究交付包包含多层目录 | 另建扁平导出包；独立解压编译并检查系统生成 PDF |

执行顺序：**完成两阶段证据 → 确定结论与贡献 → 模板迁移与图表整理 → 作者及声明核对 → 官网规则终检 → 上传并审阅系统 PDF → 提交**。

### 最终提交清单

- [x] FGCS 首投目标写入项目记录。
- [x] 核对本地稿件模板和现有证据缺口。
- [ ] 复核当前普通稿文章类型、18 页规则、摘要／关键词、匿名性、ORCID、附件和数据政策。
- [ ] 八卡主验证、独立评价、强基线与机制证据完成，失败原因保留。
- [ ] 一个真实 MoE 模型的八卡集成实验完成；摘要和结论与结果一致。
- [ ] 作者名单、顺序、两位通讯作者及 CRediT 经全体作者确认。
- [ ] 利益冲突、资助、AI 使用、数据可用性和相关版本声明完整。
- [ ] 正文、参考文献、图表编号和 Highlights 核对完成。
- [ ] 投稿 PDF 与同层目录源码包独立构建通过；可复现材料链接可访问。
- [ ] 全体作者批准最终稿；核对系统生成 PDF 后正式提交。

## 7. 费用与时间记录

本项目暂以订阅发表路线准备，最终服从资助方要求。FGCS 支持开放获取；本轮未直接核实当前 APC、税费及减免，不能把第三方价格当报价，也不以付费 OA 推断更易录用。[S2][S4]

上轮查询到的 72 天外审后决定、183 天录用是第三方于 2026-10-01 转录的统计，仍未直接复核；含返修的录用时间与首次决定不同。9–12 个月是本项目建议的排期缓冲，**不是官方周期或承诺**。[S4]

## 8. 来源索引

以下链接均于 2026-10-25 查询。正文已标注直接核实、专刊限定或转录待复核的区别；没有用投稿经验网站推断录用概率。

[S1]: https://www.sciencedirect.com/journal/future-generation-computer-systems/publish/guide-for-authors
[S2]: https://shop.elsevier.com/journals/future-generation-computer-systems/0167-739X
[S3]: https://lists.mcs.anl.gov/pipermail/hpc-announce/2025-November/011602.html
[S4]: https://journalcompass.com/journals/future-generation-computer-systems-the-international-journal-of-escien-0167739x.html
[S5]: https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
[S6]: https://www.elsevier.com/researcher/author/tools-and-resources/highlights
[S7]: https://www-prod.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-formats-checklist
[S8]: https://www.elsevier.com/about/policies-and-standards/publishing-ethics
[S9]: https://web.cels.anl.gov/~woz/papers/Workflows_2025.pdf
[S10]: https://www.elsevier.support/sciencedirect/answer/how-do-i-contact-the-corresponding-author-of-an-article
[S11]: https://declarations.elsevier.com/
[S12]: https://www.elsevier.com/researcher/author/policies-and-guidelines/credit-author-statement
[S13]: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals
[S14]: https://www.elsevier.support/publishing/answer/editor-guide-to-crosscheck-plagiarism-reports
[S15]: https://www.elsevier.com/en-in/researcher/author/tools-and-resources/research-data/data-guidelines

- [S1：FGCS 当前作者指南（本轮 403，正式投稿前必复核）][S1]
- [S2：Elsevier 官方期刊范围][S2]
- [S3：FGCS 编辑部专刊公告，2025-11-21；篇幅／模板为该公告要求][S3]
- [S4：近期指南与统计转录，非本轮直接核实的官网原文][S4]
- [S5：Elsevier LaTeX 与 EM 源文件规则][S5]
- [S6：Highlights 通用规则][S6]
- [S7：图件格式检查表][S7]
- [S8：Elsevier 作者与出版伦理][S8]
- [S9：FGCS 双通讯作者出版实例，首页][S9]
- [S10：Elsevier 对多位通讯作者的说明][S10]
- [S11：利益冲突声明工具][S11]
- [S12：CRediT 贡献声明][S12]
- [S13：Elsevier 当前期刊 AI 政策，含代码与图件][S13]
- [S14：相似度检查报告解释][S14]
- [S15：研究数据政策分档说明；FGCS 所属档次待核实][S15]
