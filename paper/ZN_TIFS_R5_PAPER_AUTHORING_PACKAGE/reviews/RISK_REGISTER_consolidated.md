<!--
ZN_TIFS_R5_FULL_ARCHIVE -- derived analysis document (NOT an original paper file).
Origin (workspace):      3/chinese_rewrite_r5/audit + _review + 3/review + 3/stage2/review + 3/final + out/multi_bridge_expansion/tifs_final_consolidation
Original landing path:   _arch_scratch\risk_register.md
Produced:                2026-09-14 (archive session)
Nature:                  derived consolidation; every row cites an original review
                         document or frozen experiment artifact in this archive.
How to verify:           each citation can be checked against the original files under
                         reviews\ and experiment_results\ (see ARCHIVE_INDEX.csv).
-->

# Consolidated Risk Register — RC-UOT-Q / EC-UOT (TIFS submission)

**Scope.** Read-only forensic consolidation of every substantive concern, weakness, threat, unresolved issue and open decision raised across the R5 / R5B / R5C review, audit and decision trail.

**Sources read (relative to workspace root `<REPO>`).**

| Group | Documents |
|---|---|
| Top-level review | `_review/TIFS_EXPERIMENT_REVIEW_R5.md`, `_review/CHAPTER5_REVISION_R5.md`, `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` |
| R5/R5B/R5C returns | `3/chinese_rewrite_r5/final/R5_FIRST_ROUND_RETURN.md`, `R5B_FINAL_RETURN.md`, `R5C_FINAL_RETURN.md`, `R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` |
| R5 audit set | `3/chinese_rewrite_r5/audit/` — `R5_HOSTILE_REVIEW_{A,B,C}.md`, `R5_HOSTILE_REVIEW_SYNTHESIS.md`, `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md`, `R5B_SCOPED_HOSTILE_CHECK_RESULTS.md`, `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md`, `R5_STRUCTURAL_CLAIM_AUDIT.md`, `R5_BASELINE_FAIRNESS_PROTOCOL.md`, `R5_CONFIRMATORY_RESAMPLING_AUDIT.md`, `R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md`, `R5_TITLE_NOVELTY_DECISION.md`, `R5_SCALABILITY_PROTOCOL.md`, `R5_122_PAIR_RECHECK.md`, `R5_EVIDENCE_CONSTRAINT_ABLATION_AUDIT.md`, `R5_REAL_MERGE_EVIDENCE_AUDIT.md`, `R5_REFERENCE_FIX_REPORT.md`, `R5_TIFS_EVIDENCE_NARRATIVE_ORDER.md`, `R5C_CN_EN_CLAIM_PARITY_AUDIT.md`, `R5C_V5_ADEQUACY_FORMULA_AUDIT.md`, `R5C_PRIOR_ART_CORRECTION_NOTE.md`, `R5B_CONFIRMATORY_DEPENDENCE_REAUDIT.md`, `R5B_V5_CLAIM_INTERPRETATION_MATRIX.md`, `R5B_V5_FEASIBILITY_AUDIT.md`, `R5B_METHOD_NAMING_IMPACT_AUDIT.md`, `R5B_SCALABILITY_PROTOCOL_CORRECTION.md`, `R5B_EXTERNAL_INFERENCE_CORRECTION.md`, `R5B_PRIOR_ART_DECODER_CONTROL_PROTOCOL.md` |
| Earlier hostile rounds | `3/review/FINAL_CN_HOSTILE_REVIEW.md`, `3/review/FINAL_CN_CLAIM_EVIDENCE_AUDIT.md`, `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md`, `3/tifs_reference_rebuild_r3/review/R3_HOSTILE_TIFS_REVIEW.md`, `3/chinese_rewrite_r4/phase2/review/R4_HOSTILE_TIFS_REVIEW.md`, `3/chinese_rewrite_r4/phase3/review/R4_FIGURE_AUDIT.md` |
| Final consolidation | `out/multi_bridge_expansion/tifs_final_consolidation/FINAL_HOSTILE_REVIEW.md`, `FINAL_TIFS_SUBMISSION_DECISION.md`, `FINAL_TITLE_DECISION.md`, `TIFS_SUPPLEMENT_CHECKLIST.md` |
| Final author decision | `3/final/FINAL_AUTHOR_DECISION_REPORT.md`, `3/final/FINAL_AUTHOR_BLOCKERS_V2.md` |

**Status vocabulary.** `resolved` = document says fixed/closed · `mitigated` = partially addressed / downgraded / disclosure-only · `open` = still outstanding · `disputed` = authors pushed back · `accepted-risk` = author decided to accept it.

---

## Consolidated risk register

| # | 问题 | 严重度 | 来源文档 | 当前状态 | 关闭证据 |
|---|---|---|---|---|---|
| R-01 | **外部方法性能完全缺失**：三次外部尝试（v3 永久执行失败、v4 G=7<8 adequacy FAIL、v5 双重 FAIL）后，方法从未在真实非一对一标注语料上产生任何性能数字（`method_predictions=0`, `method_runs=0`）。论文标题/摘要关键词为 AML / suspicious fund flow，但全部正向结果来自半合成模板。 | **blocker** | `3/chinese_rewrite_r5/audit/R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §A（"BLOCKING — largest open scientific question"）；`R5_HOSTILE_REVIEW_B.md` B-M1"Blocking Yes"；`R5_HOSTILE_REVIEW_C.md` C-M3/C-M5；`_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-B1（致命） | **accepted-risk** | 定为最终局限（CASE A）。`R5C_FINAL_RETURN.md` item 19/20 明确列为 final limitation 而非待办；`3/final/FINAL_AUTHOR_BLOCKERS_V2.md` 无阻断项。作者决策 AD-5 选 A（永久局限）。 |
| R-02 | **表 4 操作点特征循环性**：`rcuot_q_precision_rerank` 为 LogisticRegression，5 个特征中 3 个来自被比较基线（`conn_score`、`abct_score`，而 `bridge_proxy ≡ abct_score`），却以"UOT-Q"名义与未获同等调参预算的基线比精度。 | **blocker** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §5 A.1–A.3、§8 C-1（CRITICAL）；`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` §1 "NOT calibration-parity-symmetric"；`scripts/run_phase29_robust_rcuot_superiority.py:164,178` | **mitigated** | 改名 `Diagnostic reranking configuration` + 强制披露三项循环特征 + 加"不可作优势证据"禁令 + 补 ABCTracer 四项劣势（`_review/CHAPTER5_REVISION_R5.md` §5.4(b) 表 4 表注 1–5；`R5B_FINAL_RETURN.md` item 5 OPTION B 已落地 CN+EN）。彻底解决需 RE-2（特征消融重训），未执行。 |
| R-03 | **表 7（ETH-BNB protocol generalization）的 0.978/0.959/0.010/0.833 无源工件**，仅存于汇总表 `论文实验部分结果与数据表格汇总.md` 第 79–81 行；冻结 bundle 无法定位。 | **blocker** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §7 C.1 来源③、§8 C-2（CRITICAL）、AD-4；§8.3 列为"最高优先级单项建议" | **open** | 无。AD-4 要求"补齐源工件路径或删除该行"，文档明示"须作者判定该工件是否存在"。**报告撰写时未见任何后续文档宣告该行已删除或已溯源** —— 最后落点仍是 5.4(c) 的数值陈述。 |
| R-04 | **PolyNetwork 数字跨口径/跨来源混用**：来源① F1=0.7085（交易对口径，冻结 bundle 内）与来源② F1=0.8330（逐桥覆盖口径，位于未冻结的 `out/`）出现在同一比较语境。 | **blocker** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §7 C.1–C.4、§8 C-3；`_review/CHAPTER5_REVISION_R5.md` §5.5(c) 末"核对提示（提交前须解决）"；D6（更正） | **mitigated** | 已加"二者不可互换、不可并列"禁令与表注；主表口径优先（AD-1 建议 A）。但 D6 原文要求"提交前必须由作者以冻结工件为准裁决"——见下表 AD-1。 |
| R-05 | **一对一基线"0"的解释错误**：原稿称分流/合流均"结构上不可能、上限 ≤ 0.5"；代码级审计显示约束是**源侧**出度 ≤1，故 2→1 合流在数据结构上可表示，其零恢复是**经验结果**。 | **blocker** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 C-4、MUST-11；`capability_tables/capability_table.md` 更正 1；`unmatched_semantics.md` | **resolved** | `_review/CHAPTER5_REVISION_R5.md` §A.2/A.3、Part B §5.2(b) 已收回原说法并改写为"分流结构上不可能 / 合流为经验性零"，表 2 表注同步（"须修正的简化"段）。 |
| R-06 | **开发期 amount-free 核解码上限 0.3172 > 保留集条件解码 0.3104**（recovery_fraction=1.0686>1），原稿完全未交代，会被读作"发布了一个不如直接核排序的管线"。 | **blocker** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 C-5、MUST-6（"选择性报告风险"）；`FINAL_TRANSPORT_DIAGNOSIS.md` 第 6/7/20 条；`_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-A1（致命） | **resolved** | `_review/CHAPTER5_REVISION_R5.md` §5.3(a) 前置交代、表 3 独立诊断行、5.3(c) 边界三、摘要与结论各补一句；`3/final/FINAL_AUTHOR_DECISION_REPORT.md` §1–§3 全文禁用词零命中。 |
| R-07 | **宏平均掩盖桥间异质性**：全覆盖操作点 F1 为 Celer 0.5233 / **Multichain 0.0450** / PolyNetwork 0.8193；在三分之一的评测桥上方法实质无效。 | **blocker** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-C1（致命）；`R5_HOSTILE_REVIEW_B.md` B-M3；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 C-6、MUST-7 | **resolved** | `_review/CHAPTER5_REVISION_R5.md` §5.5(c) 新增表 6 + 表 6b（Multichain 阈值权衡 0.00–0.99）+ 三条"必须写明"的推论；5.4(e) 适用边界表。 |
| R-08 | **主 CI 的随机化单元过细（伪重复）**：720 实例按桥内模板实例重采样，未按模板族/种子聚类；族聚类 CI 宽 1.77×。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §G（R4-M1）；`R5_CONFIRMATORY_RESAMPLING_AUDIT.md` §1–§5；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §6、§8 M-1、MUST-8 | **mitigated** | 族聚类 CI [0.0676,0.0849]、seed→模板 [0.0677,0.0831]、基锚点簇 [0.0708,0.0803] 三者在 5.1/5.3 并列 co-report；冻结 CI 仍为主（AD-3 选 A）。 |
| R-09 | **模板族粒度不足**：每桥仅 12 个 Q\|D 族、5 个种子，无法达到 generator-level uncertainty；B-M6 认为"所有 5 种敏感性方案都排除零，部分是同构成所致的构造性结果"。 | **major** | `R5_HOSTILE_REVIEW_B.md` B-M6；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K5；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` S3（BLOCKING→已修） | **accepted-risk** | 已把 generator-bound 条款提升到摘要；Supplement S.2 改名 "robustness ladder" + 显式 no-nominal-coverage warning。彻底解决需 RE-3（扩充独立模板族后重跑），未执行。 |
| R-10 | **真实 N→1 合流证据极弱**：85（v4）/ 76（v5）个合流单元，占真值边 0.48% / 0.19%，开发语料合流占比仅 0.17%；合流结论几乎完全依赖注入式半合成模板。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §H；`R5_REAL_MERGE_EVIDENCE_AUDIT.md` §1–§2；`_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-B3 | **mitigated** | 在 5.1 A、5.2(e)、5.6(b) 第 2 条三处披露；`R5_REAL_MERGE_EVIDENCE_AUDIT.md` §3 要求"合流结论来自半合成压力测试"的定性句已落地。需 RE-4（扩样）才能消除。 |
| R-11 | **代价与边际的已知规格缺陷未修复**：边际 a/b 与流 USD 金额 corr=1.00（双重编码）；成对金额相似性与分流/合流结构结构性不兼容（真分流子边承受 ≈0.50–0.52 金额代价，全额混淆者仅 ≈0.000–0.043）；条件解码**不修复**该问题。 | **major** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-A4；`FINAL_COST_TRANSPORT_DIAGNOSIS.md` 第 2 条；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 M-4、W-2 | **mitigated** | `_review/CHAPTER5_REVISION_R5.md` §5.6(b) 第 8 条如实写入并声明条件解码不修复。彻底解决需 RE-5（amount-free 成对代价 + 重跑保留集），未执行。 |
| R-12 | **方法命名学理不准确**：风险以 0.15 代价权重 + 边际重加权 `a0·(1+λ·AML/100)` 进入，**不存在硬约束**；"Risk-Constrained"为数学上的 misnomer；"Unbalanced"在 headline 指标上无实测收益（Δ_bot CI 含零）。 | **major** | `R5_HOSTILE_REVIEW_SYNTHESIS.md` K4；`R5_HOSTILE_REVIEW_A.md` A-M3/A-M4；`R5B_METHOD_NAMING_IMPACT_AUDIT.md` §1 | **resolved** | `R5B_METHOD_NAMING_IMPACT_AUDIT.md` 方案 B 落地：正文改中性 "UOT formulation"，标识符 RC-UOT 保留并加一句"R = risk weighting, not a hard constraint"；`R5C_FINAL_RETURN.md` item 8 两种语言全稿扫过（legacy 标识符仅出现一次）；`R5B_SCOPED_HOSTILE_CHECK_RESULTS.md` §2(b) 8 处 "risk-constrained" 全部改写。 |
| R-13 | **风险项与证据项的独立贡献未量化**：标题/贡献曾依赖 "Risk-Constrained" / "Evidence-Constrained"，但无逐项消融。 | **minor** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §J（MINOR-to-MAJOR）；`R5_EVIDENCE_CONSTRAINT_ABLATION_AUDIT.md` §1–§3 | **mitigated** | 采用 claim contraction（option 1）：风险/证据项明确定位为 interface/constraint terms，不主张独立贡献；选定最小五臂有界消融仅在 v5 内可选执行，未执行。 |
| R-14 | **命题 1（双重消除）为两行代数约简**，只用到分解式与非负性、从不使用 Sinkhorn 收敛性；带 proof environment 的编号命题会被指为过度信号化。 | **major** | `R5_HOSTILE_REVIEW_A.md` A-M1；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K3；`FINAL_HOSTILE_REVIEW.md` R1-m2 | **mitigated** | 全文改为"直接推论 / 供完整性陈述的 modest corollary"，摘要去掉过度声张；作者未采纳 A-M1 的"降为 inline derivation、删除 proof environment"建议。 |
| R-15 | **解码器即既有 prior art**：S_row=P/c、S_col=P/r + 互选 top-5 即 NC-Net/LoFTR/SuperGlue 的 dual-softmax + MNN 应用到传输计划；两个确认性面（301–305 与 v5）**都只含 transport-family 解码器**，无法区分"有用适配"与"改名已有算子"。 | **major** | `R5_HOSTILE_REVIEW_A.md` A-M2（Critical）；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K3；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` N1 | **mitigated** | prior-art 控制臂已冻结入 v5 roster（Control A = 既有 `AMOUNT_FREE_COST_D4`），但**从未执行**；且 R5C 发现 Control B **规格写错**（softmax 直接作用于 K 而非 S=log K，且省略 LoFTR 置信阈值）→ 已标 MIS-SPECIFIED / NEVER EXECUTED / **RETIRED**（`R5C_PRIOR_ART_CORRECTION_NOTE.md` §1–§2）。 |
| R-16 | **确认性面受生成器约束**：720 实例共享**同一注入结构构成**（每模板固定 1 分流对 + 1 合流对 + 1 未匹配源 + 2 噪声对），v5 守卫已关闭 → 无法测 generator invariance。 | **major** | `R5_HOSTILE_REVIEW_SYNTHESIS.md` K5；`R5_HOSTILE_REVIEW_B.md` B-M1/B-M6；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` S5 | **accepted-risk** | 5.1 C.1、5.2(e)、5.3(e)、表 3 表注、5.6(b) 第 9 条多处披露；摘要级 generator-bound 条款。 |
| R-17 | **可扩展性无法测量**：原稿仅有定性 O(nm)，无 runtime/memory 表，而对比论文 EPSD-HOT / BlockAthena 均提供实测曲线。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §I；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K8；`R5_HOSTILE_REVIEW_B.md` B-M8 | **resolved** | `R5B_FINAL_RETURN.md` item 16/17（AUTHOR_APPROVED_R5_SCALABILITY_ONLY，10 网格 × 4 臂 × 3 重复，120/120 成功）：288×288 下 UOT ≈3.18 s / BOT 10.85 s / 峰值 RSS 325 MB；`R5C_FINAL_RETURN.md` item 9 措辞锁定 BOUNDED，禁用"高度可扩展/近常数复杂度/生产规模"。 |
| R-18 | **122 对商空间"完美 P/R/F1=1.000"是同义反复**：label 与评分规则共用同一 transferId 键，非循环性判决 FAIL；n=122 过小。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §M；`R5_122_PAIR_RECHECK.md` §2；`R5_HOSTILE_REVIEW_B.md` B-M9 | **resolved** | 作者决定 A：表移入补充 S.1（EN Table 3 / CN 表 S1），正文仅留一句定义一致性检查；`R5B_FINAL_RETURN.md` item 12；`R5C_FINAL_RETURN.md` item 18 悬挂引用 0。禁用词零命中（`3/review/FINAL_CN_CLAIM_EVIDENCE_AUDIT.md` §6）。 |
| R-19 | **参考文献与渲染缺陷（P0）**：5 条作者列表丢失（[10][12][16][36][39]）、LaTeX 转义泄漏（C\'edric、Peyr\'e×2、S\&P×3）、CONNECTOR DOI 无效、[28] venue 为空、多条缺页码/DOI 冲突。 | **blocker（packaging P0）** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §K；`R5_REFERENCE_FIX_REPORT.md` §2–§3；`R5_FIRST_ROUND_RETURN.md` item 9 | **resolved** | 48 条全部重核：CONNECTOR DOI → `10.1109/TIFS.2025.3588249`；[28] → RAID 2024 四作者（pp.298–316, DOI 10.1145/3678890.3678894）；[42] FATF 具体 URL；bib 同步 hash `CB16BFA4…`。`R5B_RENDER_QC_REPORT` 程序化 26/26 PASS；`R5C_FINAL_RETURN.md` item 18 REFERENCE_QC = PASS。 |
| R-20 | **参考文献页需人工视觉复核**（ér、ï、č、š、ņ、ā 变音字形、DOI/URL 折行、字段缺失）。 | **minor** | `R5_FIRST_ROUND_RETURN.md` item 9；`R5_REFERENCE_FIX_REPORT.md` §6；`R5B_FINAL_RETURN.md` item 15；`R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` | **open** | 无 renderer（无 Word/LibreOffice/pandoc），只能做全文重抽取；checklist 明确要求作者在真实渲染上逐项确认。 |
| R-21 | **参考文献条目集合前后不一致**：先前审计报告描述的 48 条是 `references.bib`，与 R4/R5 DOCX 实际 48 条**不是同一集合**（DOCX 删 7 增 7）。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §L；`R5_REFERENCE_FIX_REPORT.md` §1/§5；`R5_HOSTILE_REVIEW_A.md` A-m3 | **resolved** | 确立 DOCX 为 reference of record，bib 同步至核验后的 48 条集合并记录 pre/post hash（`07044E81…` → `CB16BFA4…`）。 |
| R-22 | **正式参考文献条目总数再次变动**：作者决定轮将 47 → **41** 条（6 条无可靠来源者移除）；英文稿为 41 条。 | **minor** | `3/final/FINAL_AUTHOR_DECISION_REPORT.md` §4；`3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R5-M2 | **resolved** | 正文 [1]–[41] 连续零缺口；正式列表 0 内部标签（UNVERIFIABLE/FAILED/TODO/AUTHOR_REVIEW_REQUIRED 全清），QA marker 计数 = 0；AUTHOR_INPUT_REQUIRED = 0。 |
| R-23 | **残留 PARTIALLY_VERIFIED 文献（9 条）与 qin2022rise 引用意图**需投稿时在出版方页面终确认。 | **minor** | `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R5-M2；`3/references/REFERENCE_AUTHOR_ACTIONS.md`；`FINAL_AUTHOR_BLOCKERS_V2.md` 随附项 2 | **mitigated** | qin2022rise 语义核查 PASS（`3/review/FINAL_CN_CLAIM_EVIDENCE_AUDIT.md` §7）；其余 9 条列为投稿时终确认，非阻断。 |
| R-24 | **断句级编辑缺陷**：摘要整句重复、§5.5 整句重复、表 1 括号错乱（"(一对一)" 重复）。 | **minor** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` §B.3；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` MUST-1/2/3、m-1..m-3 | **resolved** | `_review/CHAPTER5_REVISION_R5.md` Part C.1 给出摘要重复句合并处理；Part D D13 记为"其余待排版时清理"。 |
| R-25 | **全稿视觉 QC 未执行**：Fig.1 标签重叠、Fig.2 六阶段标签不可读、Fig.4 红字注拥挤、表 1 跨页断行——四项已程序化修复，但**无任何页面级渲染验证**。 | **blocker（submission gate）** | `R5C_FINAL_RETURN.md` items 12–16/20；`R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` §"Submission gate"；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` E1（BLOCKING） | **open** | 修复已做（Fig.1 画布 2.35→2.9 in；Fig.2 两行蛇形编号；Fig.4 红字移至图底边距；表 1 全部行 `w:cantSplit` + 表头 `w:tblHeader`），但 READY_FOR_TIFS_SUBMISSION 明确保持 **NO** 直到真实渲染 checklist 通过。 |
| R-26 | **§5.10 release URL 与开源许可证占位**。 | **minor** | `R5C_FINAL_RETURN.md` item 20(2)；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` E-cross；`FINAL_HOSTILE_REVIEW.md` R5-M1 | **mitigated** | 采用安全措辞（"将在投稿包中补充"），正文 0 假 URL / TODO / TBD（`FINAL_AUTHOR_BLOCKERS_V2.md` 随附项 3）；投稿/接收时由作者提供。 |
| R-27 | **英文稿超出 TIFS 常规页数**：编译 PDF 21 页 vs 常规约 14 页。 | **minor** | `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R5-m1；`3/tifs_reference_rebuild_r3/review/R3_HOSTILE_TIFS_REVIEW.md` R5-m1（8 页过短，需回填至 11.5–12.5 页） | **open** | 压缩策略留待作者决定；`FINAL_ENGLISH_HOSTILE_REVIEW_LENGTH_COMPLIANT.md` 确认压缩未损坏任何论证链、不可协商的负面结果与 claim boundary 均保留。 |
| R-28 | **冻结 CI 的紧致度本身可疑**：C-M2 指出"即使放宽 1.77×，[0.0706,0.0803] 对三桥聚类保留集而言仍窄得不合理，本身即信号 underestimate dependence"。 | **major** | `R5_HOSTILE_REVIEW_C.md` C-M2 | **disputed** | `R5B_CONFIRMATORY_DEPENDENCE_REAUDIT.md` 重新审计生成器血缘后**推翻**该依赖结构论断：Q\|D 单元是**分层因子**而非依赖簇；真实依赖单元是 base anchor，锚点簇 CI [0.0708,0.0803] 与冻结 CI 相差 <0.1%，故"单元选择基本无害"；"effective n is 12" 已**撤回**。 |
| R-29 | **依赖单元术语一度错误**："paired hierarchical" 被写成层级重采样、"template-family dependence clusters"、"effective n is 12"（R5 审计的自述）。 | **major** | `R5_CONFIRMATORY_RESAMPLING_AUDIT.md` §8 更正；`R5B_CONFIRMATORY_DEPENDENCE_REAUDIT.md` §4；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` S2（BLOCKING→已修） | **resolved** | R5B 更正已记录（非静默）：seed = 顶层随机复制，base anchor = 依赖单元，Q\|D = 分层，两阶段 seed→模板 CI [0.0677,0.0831] 为尊重层级的敏感性；三条禁用措辞已写入 §6；S2 已把 within-bridge 实例重采样与锚点区间 [0.0708,0.0803] 前置。 |
| R-30 | **外部统计协议两次出错**：①把 sign-flip 零分布的 2.5/97.5 百分位当作效应量 CI；②"direction > 0" 的单/双侧关系未决；③studentized-BASIC 混合写法；④seed 矛盾（v4 20240520 vs v5 20250515）。 | **major** | `R5B_EXTERNAL_INFERENCE_CORRECTION.md` §1–§3；`R5B_SCOPED_HOSTILE_CHECK_RESULTS.md` §1(b) | **resolved** | 已冻结分离：sign-flip 仅作 p 值（双侧，8≤G≤20 穷举 2^G 无 +1；G>20 MC R=100,000 +1，seed 20250515）；效应量 CI 用 residual-centered BASIC wild-cluster bootstrap（cluster=primary-address cluster, B=4000, seed 20250515, plain BASIC 无 studentization）；"direction > 0" 降为方向性复现判据；small-G（8≤G≤12）脆弱性 caveat 已加。 |
| R-31 | **外部端点选择被质疑为 endpoint shopping / task dilution**：v5 主端点是 within-method 对比（同一冻结计划的两种解码器），绝对边 F1 与严格精确恢复都被显式排除为主端点。 | **major** | `R5_HOSTILE_REVIEW_A.md` A-M6；`R5_HOSTILE_REVIEW_C.md` C-M7；`R5_EXTERNAL_ENDPOINT_JUSTIFICATION.md` §2 被拒候选 | **accepted-risk** | 以四条冻结理由辩护（论文主确认主张即 decoder repair；保留取证效用；与冻结 holdout 主端点镜像；严格恢复因 degree>5 构造性失败）。九个端点全部共同发表（`R5B_V5_CLAIM_INTERPRETATION_MATRIX.md` §3），全部端点均 descriptive。 |
| R-32 | **v5 设计陷入 catch-22**：同一 Celer ETH→BSC 车道、同一四道 gates；持久聚合器地址必然 OVERLAP_FAMILIAR，排除后保守 G 从 0 起算；v4 实测新簇到达率 ≈2/年 vs 需要的 8。 | **major** | `R5_HOSTILE_REVIEW_B.md` B-M5；`R5_HOSTILE_REVIEW_C.md` C-M5；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K6；`R5B_V5_FEASIBILITY_AUDIT.md` §1–§4 | **resolved** | 先验可行性审计给出 UNLIKELY 判定；R5B 实际收集 v5（data-only）后 **adequacy FAIL 双重确定**（保守 G=2<8；full max-share 0.6200>0.5），CASE A 逐字适用。`R5B_FINAL_RETURN.md` item 10/18。 |
| R-33 | **v5 的创建推翻了先前 "DO NOT CREATE v5" 人类裁决**，且未附新证据；C 要求把 reopen 理由保持可见。 | **minor** | `R5_HOSTILE_REVIEW_C.md` C-M5；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K6（"requires the reopening rationale to stay visible"） | **accepted-risk** | 设计冻结文档链完整（`V5_DESIGN_OVERVIEW` / `V5_CORPUS_SPEC` / `V5_DISJOINTNESS_PROTOCOL` / `V5_DATA_ACCRUAL_PREREGISTRATION`），重开理由分散记录于各 R5 文档，未见单一"reopen rationale"章节。 |
| R-34 | **fan-out 前提可能是聚合假象**：27.51% 分流份额与 2,451 个 v4 Tier-A 单元可能只是 1800 s 滚动窗 + primary-address 聚类切出来的普通交易所/聚合器行为；v4 前 5 地址占 2,441/2,451 = 99.6%。 | **major** | `R5_HOSTILE_REVIEW_C.md` C-M1（Blocking Yes）；`R5_HOSTILE_REVIEW_SYNTHESIS.md` §4；`R5B_V5_FEASIBILITY_AUDIT.md` §1 | **mitigated** | 措辞层已修（贡献一区分 pairwise 结算 vs 流级聚合；禁用"真实桥结算天然多对多"；`R5_REAL_MERGE_EVIDENCE_AUDIT.md` §2）。C-M1 要求的"七簇行为归因（交易所/资金图分析）"被明确列为**超出 data-only 审计范围的开域问题**，未做。 |
| R-35 | **abstention gate 本身从未验证**：弃权在难题上弃权、在易题上认证，但弃权决策的准确率与泄漏未做 held-out 验证；也缺 prevalence-weighted decision analysis（真实 prevalence：合流 0.17%、分流 27.51%）。 | **major** | `R5_HOSTILE_REVIEW_C.md` C-M4（Blocking Yes）；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K2（"C additionally demands a prevalence-weighted decision analysis and a held-out validation of the abstention gate"） | **open** | 无。已被定义为需新研究（prevalence-weighted 决策分析 + 弃权 gate 的 held-out 验证），作者决定轮明确"不批准任何新统计/新实验"。 |
| R-36 | **可操作精度缺失（R3-M1 FATAL）**：最优报告操作点上 ~81% 被提升的流边为假（精度 0.192，F1 0.316）；0.889 精度只存在于一对一区间，而该区间内闭集 Connector 为 0.9736。 | **blocker（曾被判 FATAL）** | `FINAL_HOSTILE_REVIEW.md` R3-M1（**唯一 FATAL**）、§3 FATAL ledger；`R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §E | **mitigated** | §5.3 改名 "Precision Trade-offs vs. Recall-Oriented Retrieval"，正文写明 ≈4/5 promoted edges false、0.889 仅在一对一区间且闭集 Connector 0.9736 更强、不主张任何 investigator-grade 操作点；结论描述词由 "precision-oriented" 改为 "coverage-qualified"；`3/final/FINAL_AUTHOR_DECISION_REPORT.md` §3 作者接受并强化。底层事实不可由措辞消除，仍未闭合。 |
| R-37 | **严格精确拓扑恢复全为 0**：5 方法 × 3 桥 15 个单元全 0；且 degree>5 的扇出单元对 top-5 解码器**构造性不可能**。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §F；`R5_HOSTILE_REVIEW_A.md` A-m5；`R5_HOSTILE_REVIEW_C.md` C-M7 | **mitigated** | 严格 0 与 0.946/0.967 永远同块出现（`R5_STRUCTURAL_CLAIM_AUDIT.md` §1 双稿 PASS）；阈值支配进正文；degree>5 界已在方法节陈述（A-m5 已落地）。 |
| R-38 | **0.946/0.967 被当作"恢复"的头条误读**（3 秒误读风险）：图内大号粗体数字旁无限定词，严格 0 注为灰色 8 pt。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §D；`R5_STRUCTURAL_CLAIM_AUDIT.md` §3；`R5_HOSTILE_REVIEW_B.md` B-M4 | **resolved** | 六关键词类全稿扫描 + 限定词紧邻强制；Fig.4 注记修复（"0.946 (edge-inclusion)" / 深色 8.6 pt 粗体严格 0 注）；英文两个节标题去 "recovery"；`R5B_FINAL_RETURN.md` item 15 Fig.4 已替换为注记修复版 PNG。B-M4 的"把隐含假阳性负担数量化"（~90 假边覆盖 ~2 真边）已写入 5.2/5.3。 |
| R-39 | **闭集原生 0.9736 与流级 0.192 并置的印象风险**。 | **major** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-B4；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 M-7、MUST-9 | **resolved** | 5.4(a) 加显式禁令"二者不可相互排名，也不构成比较基准"，5.5(a) 表注 2 复述。 |
| R-40 | **Threshold-MM 的 τ\*=0.05 落在预注册 19 点网格下边界，基线未充分调优**；而该基线在边 F1 上仍 3–4 倍高于传输族（0.120–0.168 vs ≤0.040）。 | **minor** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` R-C4；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 m-4、W-3；`R5_HOSTILE_REVIEW_B.md` B-M2 | **mitigated** | 在 5.1 参照清单与 5.4(d) 表注 2 两处写明"落在预注册网格下边界、未被充分调优；即便如此其边 F1 仍高于本管线"。B-M2 要求的"量化传输法在何处超过阈值规则"被正面回答为：仅边精度一侧小幅存在（0.0148–0.0221 vs 0.0095–0.0159）。 |
| R-41 | **公平预算基线比较在本稿中不存在**：适配基线为零预算常量启发式，所提操作点来自无界开发史；即便 v5 成功（within-method），也不能建立对任何基线的优势。 | **major** | `R5_HOSTILE_REVIEW_SYNTHESIS.md` K7；`R5_HOSTILE_REVIEW_B.md` B-M7；`R5_BASELINE_FAIRNESS_PROTOCOL.md` §2 表 | **mitigated** | OPTION B 已批准并落地：Connector-style / ABCTracer-style 正式降级为 representation-capability controls，禁用 "outperforms" 三类措辞；"zero trials" 不再被称为 fair-budget proof；信息访问定义更正为"同一 RAW 可观测量宇宙"；并写入局限"no fair-budget baseline comparison exists in this submission"。**但公平预算审计（option A）只在已关闭的 v5 内存在。** |
| R-42 | **适配基线常量的作者书面来源声明缺失**：代码显示 `calibration="none"`，但仓库内无作者声明证明常量未在 Phase 10S→29 序列中被隐式调整。 | **minor** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 M-8、AD-7；`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` §5 AUTHOR_INPUT_NEEDED①；`3/final/FINAL_AUTHOR_DECISION_REPORT.md` §2 | **mitigated** | 作者未提供声明；正文改用**最保守措辞**（明示"仓库亦未包含作者书面来源声明待补"），未虚构来源。后续 `FINAL_BASELINE_CONSTANT_PROVENANCE.md` 给出 C/C/C/B 分类并记为**非阻断**随附项（`FINAL_AUTHOR_BLOCKERS_V2.md` 随附项 1）。 |
| R-43 | **v4 / v5 充分性门槛失败原因不同，摘要曾只写半句**：v4 行为来源多样性不足（G=7<8）；v5 G_full=8 但主地址集中度过高（max-share 0.6200>0.5）且保守 G=2。 | **major** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` §4.6 结论②；`_review/CHAPTER5_REVISION_R5.md` Part C.1 原句 5 | **resolved** | 摘要与正文均补齐两种失败原因；`R5C_FINAL_RETURN.md` item 1 合并 12 行表 4 并更新图 6 图注与摘要一句话。 |
| R-44 | **保守 max-share 分母口径易误读**：0.0009 = 最大未见簇(5) / **全部** Tier-A 扇出单元(5,516)；而集内占比 5/6 = 0.8333 才是直觉读法但**不是门槛量**。 | **minor** | `R5C_V5_ADEQUACY_FORMULA_AUDIT.md` §2–§4；`R5C_FINAL_RETURN.md` item 5 | **resolved** | 公式重推导无代码/报告错误；命名要求锁定为"largest unfamiliar-cluster size / total Tier-A fan-out units"，并写入表注与 Supplement S.3；0.8333 记为 NOT USED / NOT REPORTED as gate quantity。 |
| R-45 | **CN / EN 主张一致性**：曾担心核心主张只在单语版本出现。 | **major** | `R5C_CN_EN_CLAIM_PARITY_AUDIT.md`；`3/stage2/review/ENGLISH_CHINESE_CLAIM_PARITY_AUDIT.md`；`3/tifs_reference_rebuild_r3/review/EN_R3_CLAIM_PARITY.md` | **resolved** | 23/23 claims 双语齐备（`R5C_FINAL_RETURN.md` item 17 PASS）；EN 主文 Table 3 悬挂引用 = 0；Supplements S.1–S.4 全部解析到文件。 |
| R-46 | **legacy 字段方向颠倒**：`many_to_one` = 1→N fan-out、`one_to_many` = N→1 merge，与规范拓扑相反，属数据完整性陷阱。 | **minor** | `R5_HOSTILE_REVIEW_C.md` C-m5；`R5_REAL_MERGE_EVIDENCE_AUDIT.md` §3(3) | **mitigated** | 表 1 脚注载明历史字段方向颠倒、全文统一"分流=1→N、合流=N→1"（`3/review/FINAL_CN_CLAIM_EVIDENCE_AUDIT.md` §1 PASS）；C-m5 要求"重新审计每个下游使用"仅部分可见。 |
| R-47 | **Tier-A 真值的纯度是承重的**：pairwise transferId 关联是协议原生的，但组装 fan-out 单元的 primary-address 聚类是**启发式层**；v5 主端点完全跑在 Tier-A 单元上。 | **major** | `R5_HOSTILE_REVIEW_C.md` C-M6（Blocking Yes） | **mitigated** | `R5B_SCOPED_HOSTILE_CHECK_RESULTS.md` §4(a) 用 `V5_COUNTING_DEFINITIONS.md` 钉死聚类链接判据（cluster key = source-flow primary address，单地址簇构造）、Tier-A 阈值（degree ≥ 2，全部边协议原生）与 familiarity 范围，并纳入 hash lock。C-M6 要求的"证明每个 Tier-A 单元可由结算事件 + 地址身份单独推导"未完整给出。 |
| R-48 | **条件解码公式 v5 adequacy 相关公式正确性**：需逐项重推导 G_full / G_conservative / max_share_full / max_share_cons 以防代码或报告错误。 | **blocker** | `R5C_V5_ADEQUACY_FORMULA_AUDIT.md`；`R5C_FINAL_RETURN.md` item 4 | **resolved** | 全部量从 hash-locked collector 与冻结 v5 工件重推：G_full=8（sizes [3420,1171,495,306,112,6,5,1]）、G_conservative=2（[5,1]）、max_share_full=3420/5516=0.6200、max_share_cons=5/5516=0.0009。**无代码或报告错误**；FAIL 决策双重确定。 |
| R-49 | **遗留旧 numbering / 陈旧文档引用**：Table 3 / 表 7 等编号在 CN 与 EN 之间迁移造成悬挂引用。 | **minor** | `R5B_SCOPED_HOSTILE_CHECK_RESULTS.md` §3(d)；`R5C_CN_EN_CLAIM_PARITY_AUDIT.md` §2 | **resolved** | 三处悬挂 "Table 3" 引用修复（→Supplement S.1）；CN Supplement S.1 目标文件已建（`CN_补充S1_覆盖商空间一致性核对.md`）。 |
| R-50 | **v5 可复现打包缺陷**：12 个 per-block anchor 文件共用同一 bundle 路径（解压互相覆盖），manifest 缺少 availability 节承诺的条目。 | **blocker（packaging）** | `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` E2（BLOCKING）；`R5C_FINAL_RETURN.md` item 11 | **resolved** | bundle 重建：per-block anchors 置于 `v5_corpus/blocks/bNN/`，68 条目、无重复路径（duplicate-path guard: NONE），补入 Cross AML prototype / confirmatory statistics+verifier / one-shot runner / independent verifier / v3 failure record；§5.10 与 manifest 一致。 |
| R-51 | **正文残留修订史叙述**（"the earlier statement … imprecise"、"Old headline metrics"、"An earlier Route A v1 run was rejected"、"Reading precision on hierarchical"）。 | **minor** | `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` E3（BLOCKING→已修）；`R5C_FINAL_RETURN.md` item 10 | **resolved** | 五处叙述全部改写为陈述句；§5.8 压缩为四句；种子图/字面 sha256/验证脚本名/v3 调试年表/quarantine 细节移入 Supplement S.3。 |
| R-52 | **"independent verifier" 一度只是声称**（未提供独立代码路径与复算范围）。 | **minor** | `FINAL_HOSTILE_REVIEW.md` R4-m3/R5-m8；`R5_BASELINE_FAIRNESS_PROTOCOL.md` 相关 | **resolved** | §4.3.5/§5.8 说明独立代码路径 `verify_locked_temporal_external_validation.py` 从原始 per-cell 工件复算，1e-9 内一致；`verify_locked_external_validation` 脚本纳入 bundle。 |
| R-53 | **recovery_fraction = 1.0686 无 CI**，且 >1 的含义未被解释。 | **minor** | `FINAL_HOSTILE_REVIEW.md` R4-m8；`_review/TIFS_EXPERIMENT_REVIEW_R5.md` §B.2 | **resolved** | §4.3.5 明确其为无 CI 的点估计，并说明 >1 表示"超过参考上限"。 |
| R-54 | **单次置换对照**（permuted-label control 只有一个固定置换）被指证据力不足。 | **minor** | `FINAL_HOSTILE_REVIEW.md` R4-m6 | **resolved** | Table 5 脚注声明该对照为单一固定置换，按"确定性 sanity control"报告。 |
| R-55 | **"Mathematically equivalent" 求解器修复仅为断言**。 | **minor** | `FINAL_HOSTILE_REVIEW.md` R5-M3 | **resolved** | §5.8 给出单行数学论据（KL-UOT 目标在零参考质量上无穷）并降级为"在正支撑上经验等价（15/15 + 8/8）"。 |
| R-56 | **v3 永久执行失败（零边际 NaN + 实现缺陷）、不重跑的决策**需在正文完整披露。 | **major** | `R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §A；`FINAL_HOSTILE_REVIEW.md` R5-m4；`R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` F5 | **resolved** | §5.6/§5.8 完整披露原因与 no-rerun 决策；quarantined b04–b06 已入 §5.8；`v3_record/V3_EXECUTION_FAILURE_ADJUDICATION.md` 入 bundle。 |
| R-57 | **摘要/结论的"可疑/illicit"措辞可能被读作方法产出台非法性判定**。 | **major** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` §四项重点检查、§B.2；`_review/CHAPTER5_REVISION_R5.md` Part C.1 原句 1 | **resolved** | 摘要动机句后紧接显式否定"本文不产出台非法性判定，只输出证据限定的对应假设与弃权"；`_review/CHAPTER5_REVISION_R5.md` §A.2 禁用"method solves AML"→"supports AML investigation by recovering evidence-constrained … correspondence"。 |
| R-58 | **triage / "precision-oriented" 定位过强**。 | **minor** | `FINAL_HOSTILE_REVIEW.md` R3-M1；`_review/CHAPTER5_REVISION_R5.md` §5.5(a) 表注 3 | **resolved** | 结论描述词改为 "coverage-qualified"；正文写明"流级压力操作点精度 0.192，本文不作分诊级操作点声明"。 |
| R-59 | **标题/新颖性定位风险**：Option 0 前景化了**最未获确认**的 "Transport Representation"与**最暴露于 prior art** 的 "Conditional Decoding"，却省略了 abstention（可能最具领域新颖性）。 | **major** | `R5_HOSTILE_REVIEW_A.md` A-M9；`R5_TITLE_NOVELTY_DECISION.md` §1；`R5_HOSTILE_REVIEW_SYNTHESIS.md` §3 | **disputed** | harness 保留 Option 0 为常设标题；A-M9 要求"以 A-M2/A-M7 为显式判据重跑标题决策"未执行。见下表 AD-3。 |
| R-60 | **"Non-One-to-One" 标题词与严格恢复 = 0 的张力**。 | **minor** | `R5_HOSTILE_REVIEW_A.md` A-M9 | **mitigated** | 摘要/贡献三逐层标注"确立了什么/未确立什么"；严格 0 与 0.946/0.967 同块出现；`R5_TITLE_NOVELTY_DECISION.md` §3 记录保留理由。 |
| R-61 | **S_row / S_col 命名易被读者反向理解**（分别除以列质量/行质量）。 | **minor** | `R5_HOSTILE_REVIEW_A.md` A-m1；`R5_HOSTILE_REVIEW_SYNTHESIS.md` §5(6) | **open** | 建议改为 destination-conditional / source-conditional，文档仅记为 minor checklist 项，未见落地证据。 |
| R-62 | **微基准外推禁令**：曲线在 50×50…288×288 近乎平坦源于求解器固定开销，不反映候选格增长。 | **minor** | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 m-5、W-5；`R5C_FINAL_RETURN.md` item 9 | **resolved** | 表 8 表注写明平坦原因并明令"禁止高度可扩展 / 近常数复杂度 / 生产规模"表述；`R5C_FINAL_RETURN.md` item 9 确认禁用措辞缺席。 |
| R-63 | **Recall@3 分母（弃权源被移出）与并列打破规则未在正文说明**。 | **minor** | `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R4-M2；`3/review/FINAL_CN_HOSTILE_REVIEW.md` R4-m2 | **mitigated** | 表 2 脚注已载明分母；并列规则入脚注与 Supplement Appendix D；分母含弃权源的变体属新计算，冻结期不执行。见下表 AD-7。 |
| R-64 | **表 4 同为单划分点估计、无 CI**，审稿人可能要求 bootstrap 显著性。 | **minor** | `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R2-M2；`3/review/FINAL_CN_HOSTILE_REVIEW.md` R2-M2；`FINAL_HOSTILE_REVIEW.md` R4-m2 | **accepted-risk** | 正文已标注探索性、单划分、无 CI；作者决定轮明确"统计冻结：零新增统计"，不批准新 bootstrap。 |
| R-65 | **Cross AML 原型的 candidate discovery 依赖后端，dry-run 最小输入未说明**，复现门槛偏高。 | **minor** | `3/review/FINAL_CN_HOSTILE_REVIEW.md` R3-m2；`3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R3-m1 | **open** | 建议补充"源哈希即可"的最小 dry-run 说明；未见落地证据。 |
| R-66 | **框架图（Fig.2）信息密度不足**，与三篇参考 TIFS 论文的主框架图相比为单栏小图；Fig.3/4 未做多面板重组。 | **minor** | `3/tifs_reference_rebuild_r3/review/R3_HOSTILE_TIFS_REVIEW.md` R6-M1 | **mitigated** | R5C 已把 Fig.2 重做为两行蛇形 + 短英文阶段标签（科学内容不变）；"更高信息密度/更宽主框架图"的重绘未做（需作者授权）。 |
| R-67 | **表 2 的 Recall@3 0.481 定义来源缺失（AUTHOR_INPUT_NEEDED）**。 | **minor** | `TIFS_SUPPLEMENT_CHECKLIST.md` §6 末项、§10.3 | **open** | 需定位 `FINAL_TOP3_METRIC_DEFINITION.md` 并加一行定义；文档明示 AUTHOR_INPUT 未决。 |
| R-68 | **Table III 主表被压缩为 11 行**（一对一基线行移入 Supplement S3），审稿人可能要求主表完整。 | **minor** | `3/tifs_reference_rebuild_r3/review/R3_HOSTILE_TIFS_REVIEW.md` R2-M1 | **mitigated** | 严格 0 的披露载体保留在主文；全 15 行表在补充 A.4；作者可决定是否恢复。 |
| R-69 | **R1 公式级 prior-art 对照缺失**：`S_row` 与 dual-softmax 列归一化形式只有谱系引用，无逐点公式对照。 | **minor** | `3/review/FINAL_CN_HOSTILE_REVIEW.md` R1-M1；`3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R1-M1 | **mitigated** | 正文已有"谱系对照基于摘要级证据、公式级归属待定"的声明（部分存在）；逐点对照段未补。 |
| R-70 | **命题严格正性假设（u_i>0, v_j>0, K_ij>0）在真实零边际（v3 失败根因之一）不成立**；零边际存在时 S_row 的数值行为未说明。 | **minor** | `3/review/FINAL_CN_HOSTILE_REVIEW.md` R1-M2；`3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R1-M2 | **mitigated** | §5.8 已有"支撑归约后零边际侧不进入解码分母"的等价性论述，可交叉引用；R1-M2 要求的显式交叉引用句未确认落地。 |

---

## Blockers that remain open

| # | 问题 | 为什么仍是阻断项 | 需要什么才能关闭 |
|---|---|---|---|
| B-1 | **全稿视觉 QC 未执行（E1）** —— `VISUAL_QC = NOT_EXECUTED`；Fig.1/Fig.2/Fig.4 与表 1 四项修复仅为程序化/文本验证。 | `R5C_FINAL_RETURN.md` item 20 明确 READY_FOR_TIFS_SUBMISSION = **NO**，并把 E1 列为唯一仍在的作者侧阻断项。子代理环境无 Word/LibreOffice/pandoc/wkhtmltopdf/docx2pdf，无法产生页面级渲染。 | 作者在真实渲染器（Word/WPS）上逐项通过 `R5C_AUTHOR_VISUAL_QA_CHECKLIST.md` 的 8 项：Fig.1 无重叠、Fig.2 六阶段标签 100% 可读、Fig.4 红字位于图底边距且不遮挡、表 1 不跨页且表头重复、参考页 48 条变音字形/DOI/折行完整、全部公式渲染、图表交叉引用正确（表 4 为合并后 12 行 v4/v5 审计表）、无字体替换。 |
| B-2 | **§5.10 release URL 与开源许可证占位（E2 类）**。 | `R5C_FINAL_RETURN.md` item 20(2) 与 `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` cross-review synthesis 均列为 author-assigned 投稿前阻断项（"READY can flip to YES" 的条件之一）。 | 作者提供真实 release URL 与 license 类型并替换 §5.10 安全措辞。 |
| B-3 | **表 7（ETH-BNB protocol generalization）的 0.978/0.959/0.010/0.833 无源工件**。 | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8.3 把 AD-4 列为**最高优先级单项建议**：四个数字既在稿件中承担实质论证（近一对一锚定基线占优），又无冻结工件支撑；且 C-2（溯源）与 C-3（口径混用来源③）是文档自述"**唯二必须通过删除或裁决才能闭合**"的 Critical 项。 | 作者补齐冻结工件路径并纳入 bundle；或删除该行。删除可一并闭合 C-3 中来源③ 的混用风险。 |
| B-4 | **PolyNetwork 数字的定稿裁决（AD-1）**。 | D6/AD-1 原文要求"**提交前必须由作者以冻结工件为准裁决**"；当前仅以"核对提示"形式标注未擅自裁决。 | 作者在选项 A（主表口径 0.8894/0.7085/0.8453，删除或改写 0.833 相关表述）与选项 B（逐桥覆盖口径 0.8893/0.8330/0.8810）之间书面裁决并存档。 |
| B-5 | **abstention gate 未做 held-out 验证 + 缺 prevalence-weighted decision analysis**。 | C-M4 判为 Blocking Yes：弃权决策本身的准确率与泄漏从未验证，"在难题上弃权、在易题上认证"是设计性拒答；且半合成压力配比（1 分流 + 1 合流 + 1 未匹配 + 2 噪声）与真实 prevalence（合流 0.17%）不匹配。 | 需要新研究：prevalence-weighted 决策分析（含假归属与漏边成本）+ 弃权 gate 的 held-out 验证 + 每桥种子数 >5。作者决定轮已明确不批准新统计/新实验 → 在当前冻结范围内**不可闭合**。 |
| B-6 | **公平预算基线比较在本稿中不存在（K7）**。 | 适配基线为零预算常量启发式，所提操作点来自无界开发史且特征含基线分数；v5 主端点为 within-method 对比，即便成功也不能建立对任何基线的优势。本稿**任何地方都不存在公平预算比较**。 | 需 RE-2（移除 `conn_score`/`abct_score`/`bridge_proxy` 三特征后重训重排器并在同一保留集 292–311 评估）或 v5 内的 option-A 公平预算审计。文档自述 RE-2 是"唯一成本可控且收益明确的项"（脚本与保留集均已存在），但**未执行**。 |
| B-7 | **真实非一对一标注数据上的方法性能缺失（RE-1）**。 | 最根本的科学缺口；`R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §A 定级 BLOCKING（"largest open scientific question"）。当前以 CASE A 永久局限承载，但 v5 双重 FAIL（保守 G=2<8；full max-share 0.6200>0.5）与 v3 永久失败意味着**在冻结范围内无路径闭合**。 | 在通过预注册充分性标准的独立语料上执行一次方法评估（需新语料收集 + 方法执行 + 标注验证）。作者选择 AD-5 选项 A（作为永久局限），因此**按作者决定不再是待办**；但从审稿风险角度仍是最大拒绝诱因。 |

---

## Author decisions that deviate from reviewer advice

| 决策 | 审查方建议 | 作者决定 | 出处 |
|---|---|---|---|
| **表 4 保留 vs 删除** | 审查方给出三个层级的建议：`_review/TIFS_EXPERIMENT_REVIEW_R5.md` 附 2 与 `_review/CHAPTER5_REVISION_R5.md` 附 1 均称"**删除表 4**从证据纪律看最干净（该校准对比不成立）"；PRE_SUBMISSION 审计 AD-2 推荐选项 A（保留 + 披露）但要求作者"确认接受'含基线分数的有监督重排器'这一公开表述"；A-M2/K3 要求移除基线特征重跑（RE-2）。 | 保留表 4，改名 `Diagnostic reranking configuration` + 强制披露三项循环特征 + 加"不可作优势证据"禁令 + 补 ABCTracer 四项劣势。**未执行 RE-2 特征消融重训。** | `_review/CHAPTER5_REVISION_R5.md` Part B 5.4(b) 与"附：本次修复未采用的备选方案"第 1 条；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §5 A.5 方案 5 与"本轮建议"；`R5B_FINAL_RETURN.md` item 5（OPTION B）；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §4 RE-2（记录为未执行） |
| **主 CI 口径** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` 附录 1 明言"**更保守也更正确**（F13），但会与已冻结补充 B.3 口径不一致"；AD-3 选 B 的条件是"若审稿人已质疑伪重复，则应改 B"；PRE_SUBMISSION 审计 §6 预测"TIFS 统计倾向审稿人会要求把族聚类 CI 作为**并列或替代主口径**"。 | 保留冻结 CI [0.070557, 0.080312] 为主口径（与已冻结补充 B.3 一致），三种敏感性 CI 并列 co-report。 | `_review/CHAPTER5_REVISION_R5.md` 附 1 与 Part D D2 缓解路径列①；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` AD-3 "建议 A" |
| **表 7 的溯源裁决** | AD-4 要求"补齐源工件路径并纳入冻结 bundle"或"**删除该行**"；§8.3 建议"若无法溯源，**建议删除**（不可解释的数字比缺失数字更危险）"。 | **未见作者裁决记录**；表 7 数值仍在 5.4(c) 作为"适配基线优于本管线"的证据陈述。 | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8 C-2、AD-4、§8.3；`_review/CHAPTER5_REVISION_R5.md` §5.4(c) |
| **"Unbalanced" 命名** | A-M3 给出明确解决测试：**"要么在 v5 加未匹配质量端点，要么改名为 RC-OT 并把 KL 边际降为实现细节"**；N 项（gap matrix）记录审稿人攻击原话"why does the method carry Unbalanced when it has no measured benefit"。 | 未改名、未加未匹配质量端点；改以中性散文 "UOT formulation" + 保留实现标识符 RC-UOT + 一句"R = risk weighting, not a hard constraint"应对。 | `R5_HOSTILE_REVIEW_A.md` A-M3；`R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §N；`R5B_METHOD_NAMING_IMPACT_AUDIT.md` §2（方案 A/B/C）与 §4；`R5C_FINAL_RETURN.md` item 8 |
| **"Risk-Constrained" 命名** | A-M4 建议"**全文改名 risk-weighted / risk-shaped，或跑最小有界 full-vs-minus-risk 消融**"；K4 建议"rename to RC-OT"。 | 采方案 B：正文改中性 UOT 表述，标识符 RC-UOT 保留并加解释句；**未跑风险项消融**。 | `R5_HOSTILE_REVIEW_A.md` A-M4；`R5_HOSTILE_REVIEW_SYNTHESIS.md` K4；`R5B_METHOD_NAMING_IMPACT_AUDIT.md` §3 推荐 Option B |
| **标题 Option 0 vs Option 2** | A-M9 要求"**以 A-M2 与 A-M7 为显式判据重跑标题决策**；Option 2 或 task-plus-abstention 变体更安全，贡献列表应以已确认的诊断领衔"；§3（synthesis）要求"must re-run the decision with A-M2/A-M7 as explicit criteria"。 | 保留 Option 0（"Non-One-to-One Cross-Chain Forensic Fund Flow Correspondence: Transport Representation and Conditional Decoding"），以两部正文调整替代改标题；**未按 A-M2/A-M7 判据重跑决策**。 | `R5_HOSTILE_REVIEW_A.md` A-M9；`R5_HOSTILE_REVIEW_SYNTHESIS.md` §3 与 §6(3)；`R5_TITLE_NOVELTY_DECISION.md` §3 |
| **命题 1 的表述形式** | A-M1 解决测试：**"降为 inline derivation 或 Remark、删除 proof environment、从结论移除 dual-cancellation Proposition 措辞"**。 | 保留编号命题 + proof environment，仅把措辞降为 "direct corollary / modest corollary"。 | `R5_HOSTILE_REVIEW_A.md` A-M1；`FINAL_HOSTILE_REVIEW.md` R1-m2；`R5_TITLE_NOVELTY_DECISION.md` §4 |
| **prior-art 解码器控制臂** | A-M2 解决测试：**"把预注册的 LoFTR-style dual-softmax-on-K 与 NC-Net-style mutual-topk-on-K 臂加入 v5，且制定与 novelty 相关的预指定判决规则"**。 | 控制臂设计冻结但**从未执行**；R5C 进一步发现 Control B 规格写错 → 标 MIS-SPECIFIED / RETIRED（历史 hash 保留）。 | `R5_HOSTILE_REVIEW_A.md` A-M2；`R5B_PRIOR_ART_DECODER_CONTROL_PROTOCOL.md` §2–§3；`R5C_PRIOR_ART_CORRECTION_NOTE.md` §1–§2 |
| **核解码上限纳入表 3 作为正式基线** | CHAPTER5 Part D D4 缓解路径：**"若审稿人要求，可把 0.3172 的核解码本身作为一条解码基线正式纳入表 3（数字已冻结，仅需重新表述，不需新实验）"**。 | 仅作为**独立诊断行**（标注"非保留集、不参与任何对比"）附于表 3 末行，未纳入主对比。 | `_review/CHAPTER5_REVISION_R5.md` Part D D4；`_review/CHAPTER5_REVISION_R5.md` §5.3 表 3 表注 3 |
| **5.5 节（仅数据审计）去留** | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` 附 3 与 `_review/CHAPTER5_REVISION_R5.md` 附 2 建议"**若审稿人对'未运行方法的审计'容忍度低，可整体移入补充 S.3**，正文仅留一句问题存在性陈述与失败记录"。 | 保留在正文（理由是 F17 把它定为最终外部有效性状态、摘要需要它支撑诚实性）。 | `_review/TIFS_EXPERIMENT_REVIEW_R5.md` 附 3；`_review/CHAPTER5_REVISION_R5.md` 附 2；`R5C_FINAL_RETURN.md` item 1/2（正文保留） |
| **122 对表的处置** | 多个审查方建议移入补充或删除（`R5_HOSTILE_REVIEW_A.md` A-m2"即使'定义一致性检查'框架也应移入补充或删除"；B-M9；C-m1；K9"must leave the main text"）。 | **采纳**：移入补充 S.1 / 表 S1，正文仅留一句（作者决定 A）。此为**与建议一致**的记录，列入本表以标明该争议已按审查方意见闭合。 | `R5B_FINAL_RETURN.md` item 12；`3/final/FINAL_AUTHOR_DECISION_REPORT.md` §1；`R5C_FINAL_RETURN.md` item 18 |
| **公平预算审计（option A）** | `R5_BASELINE_FAIRNESS_PROTOCOL.md` §4 harness 建议"**option B now + option A only inside the approved v5 execution**"；B-M7 要求把 option B 措辞逐字整合并加局限句。 | Option B 已整合（representation-capability controls + 禁用 superiority 措辞 + "no fair-budget baseline comparison exists"）；**option A 随 v5 关闭而永久不可达**。 | `R5_BASELINE_FAIRNESS_PROTOCOL.md` §4 与 R5B UPDATE；`R5B_FINAL_RETURN.md` item 5；`R5B_V5_CLAIM_INTERPRETATION_MATRIX.md` CASE A |
| **新增统计 / 敏感性补充** | 多个审稿方要求：种子簇级 bootstrap 敏感性（R4-M1）、表 4 的 bootstrap 显著性（R2-M2）、Recall@3 分母含弃权源的变体（R4-M2）、prevalence-weighted 决策分析（C-M2/C-M4）、模板族扩充后重跑（RE-3）、amount-free 成对代价重跑（RE-5）。 | 作者决定轮明确：**"统计冻结——作者已决定不批准，关闭"**（零新增统计）。 | `3/final/FINAL_AUTHOR_DECISION_REPORT.md` §6 与 §8.5；`3/review/FINAL_CN_HOSTILE_REVIEW.md` §6 共识关注②；`_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §4 RE-1..RE-5 |
| **v5 的创建（推翻 DO NOT CREATE v5 裁决）** | C-M5 指出"先前裁决说 DO NOT CREATE v5（2026-09-05 人类决定），本轮在**无新证据**下推翻"，且要求 reopen 理由保持可见。 | 创建 v5 设计并实际收集（data-only），结果 adequacy FAIL 双重确定，CASE A 逐字适用；重开理由分散记录，未集中成单一章节。 | `R5_HOSTILE_REVIEW_C.md` C-M5；`R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §A（"DO NOT CREATE v5"原文）；`R5B_FINAL_RETURN.md` items 7–10 |
| **"外部性能未建立"的最终定位** | AD-5 建议选项 A：作为**永久局限**声明（当前 F17 立场，不列为待办）。 | **采纳选项 A**（记为永久局限而非待办）——与建议一致，列出以示该争议闭合。 | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §3 AD-5；`R5C_FINAL_RETURN.md` item 19/20 |
| **方法命名体系（工程标识符）** | 审查方建议把标题与正文命名统一去掉 "Unbalanced / Risk-Constrained"；`R5B_METHOD_NAMING_IMPACT_AUDIT.md` 方案 C 提出新散文名 RW-UOT。 | 采方案 B（中性散文 + 保留标识符）；`R5C_FINAL_RETURN.md` item 8 进一步把 paper-facing 名统一为 "UOT formulation / conditional UOT decoding / UOT-Q pipeline"，legacy `RC-UOT(-Q)` 仅出现一次于实现注。代码标识符未改。 | `R5B_METHOD_NAMING_IMPACT_AUDIT.md` §2 方案 B vs C、§4；`R5C_FINAL_RETURN.md` item 8 |
| **作者书面来源声明（基线常量）** | AD-7 要求作者声明"跨 Phase 10S→29 从未隐式调优"或承认存在跨轮次调整（`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` §5 AUTHOR_INPUT_NEEDED①）。 | 作者**未提供书面声明**；正文改用最保守措辞（"仓库亦未包含作者书面来源声明待补"），未虚构来源；后续以 C/C/C/B 分类记为**非阻断**随附项。 | `_review/PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §5 A.4 §5、§3 AD-7；`3/final/FINAL_AUTHOR_DECISION_REPORT.md` §2；`FINAL_AUTHOR_BLOCKERS_V2.md` 随附项 1 |
| **命题正性假设 vs 零边际（v3 根因）的交叉引用** | `3/review/FINAL_CN_HOSTILE_REVIEW.md` R1-M2 与 `3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R1-M2 均要求补一句交叉引用（支撑归约后零边际侧不进入解码分母）。 | 未新增交叉引用句（§5.8 已有等价性论述可被引用）。 | `3/review/FINAL_CN_HOSTILE_REVIEW.md` R1-M2；`3/stage2/review/FINAL_ENGLISH_HOSTILE_REVIEW.md` R1-M2 |

---

## 统计汇总

**按状态（70 项）**

| 状态 | 数量 | 占比 |
|---|---|---|
| resolved（已修复/已闭合） | 31 | 44.3% |
| mitigated（部分处理 / 降级 / 仅披露） | 27 | 38.6% |
| open（仍悬置） | 8 | 11.4% |
| accepted-risk（作者决定接受） | 5 | 7.1% |
| disputed（作者未采纳审查方意见） | 2 | 2.9% |
| **合计** | **70** | 100% |

open 8 项：R-03（表 7 无源工件）、R-20（参考页人工渲染复核）、R-25（全稿视觉 QC）、R-35（abstention gate 未验证）、R-61（S_row/S_col 命名）、R-65（Cross AML dry-run 说明）、R-67（Recall@3 定义来源）、R-27（页数）。

**按严重度**

| 严重度 | 数量 | 其中 resolved | 其中 mitigated | 其中 open | 其中 accepted-risk / disputed |
|---|---|---|---|---|---|
| **blocker** | 12 | 5 | 5 | 1 | 1 |
| **major** | 27 | 10 | 12 | 3 | 2 |
| **minor** | 31 | 16 | 10 | 4 | 1 |
| **合计** | **70** | **31** | **27** | **8** | **4** |

**早期轮次的历史级联（供趋势判断）**

| 轮次 | 文档 | 最高判级 | 数量 |
|---|---|---|---|
| R3（8 页稿） | `3/tifs_reference_rebuild_r3/review/R3_HOSTILE_TIFS_REVIEW.md` | 无 FATAL、无 Blocking | 6 项（全部 Major/Minor 作者决策） |
| R4 阶段 2/3 | `3/chinese_rewrite_r4/phase2/review/R4_HOSTILE_TIFS_REVIEW.md` | 无 FATAL、无 Blocking | 5 项（1 可选强化 + 2 可辩护现状 + 2 可选润色） |
| 打包轮（五审） | `out/multi_bridge_expansion/tifs_final_consolidation/FINAL_HOSTILE_REVIEW.md` | **FATAL 1**（R3-M1 精度框架）| FATAL 1 / MAJOR 15 / MINOR 29；随后 **PARTIALLY ADDRESSED** |
| R5（三审） | `3/chinese_rewrite_r5/audit/R5_HOSTILE_REVIEW_SYNTHESIS.md` | 共识阻断 K1–K9 | 9 项阻断关注 + 4 项非阻断共识 |
| R5B（四审定向） | `R5B_SCOPED_HOSTILE_CHECK_RESULTS.md` | 全部 CLOSED 或 PARTIAL→FIXED | 4 轴全闭合 |
| R5C（四审） | `R5C_FINAL_SUBMISSION_HOSTILE_REVIEW.md` | BLOCKING 8 项 + KNOWN LIMITATION 11 项 | 8 项中 7 项在本轮修复，仅 **E1（视觉渲染）** 留作作者侧；11 项 limitation 全部保留 |
| 最终作者轮 | `3/final/FINAL_AUTHOR_BLOCKERS_V2.md` | **无阻断项** | 0 阻断 + 3 非阻断作者事实输入 |

**级联结论**：整个链条的判级从「R3/R4 无阻断」→「打包轮 1 项 FATAL」→「R5 九项共识阻断」→「R5C 八项 BLOCKING 全部为打包/措辞级」→「最终 V2 零阻断项」。**唯一跨轮次未收敛的类别是"证据缺口"（外部性能、公平预算基线、真实合流、弃权验证），它被系统性转化为已披露局限而非被消除。**

---

## 5 项最严重剩余风险（评估）

> 判据：结合（a）多轮独立审稿方的最高判级、（b）是否可用措辞消除、（c）审稿人自行发现的杀伤力、（d）当前是否有任何闭合路径。

### 1. 外部方法性能完全缺失 + 真实非一对一标注数据上零性能证据（R-01 / B-7，blocker, accepted-risk）
所有第三方审稿在这一点上独立一致：`R5_TIFS_PRE_REVIEW_GAP_MATRIX.md` §A 判 **BLOCKING**（"largest open scientific question"），B-M1 判 **Blocking Yes**（"a design document is not a result"），C-M3/C-M5 判 **Blocking Yes**，C 的总结姿态是**"reject in present form"**；`R5_HOSTILE_REVIEW_A.md` 的第三条 attack verdict 判 **OPEN**。
杀伤力在于它打在论文的**标题词**上——标题与摘要承诺 "forensic fund flow correspondence / suspicious fund flow correspondence"，而三次外部尝试给出的是 v3 永久失败 + v4 G=7 失败 + v5 双重失败，`method_predictions = 0`。这不是"证据弱"，而是"零外部证据"。
不可闭合性：v5 是**最后一条合法外部通道**，其 adequacy 已 FAIL（保守 G=2<8；full max-share 0.6200>0.5），且 B-M5 预测的 catch-22 已被实证；RE-1 需要新语料 + 方法执行 + 标注验证。当前唯一防线是把"外部性能未建立"作为永久局限（AD-5 选 A），即 **accepted-risk**。审稿人仍可能据此要求大修或拒稿——这是本稿最大的单一拒绝诱因。

### 2. 表 4 操作点的特征循环性（R-02 / B-6，blocker, mitigated）
`rcuot_q_precision_rerank` 的 5 个特征中 3 个直接来自被比较的基线（`conn_score`、`abct_score`，而 `bridge_proxy` 是 `abct_score` 的逐行复制，脚本 `run_phase29_robust_rcuot_superiority.py:164,178`），且该操作点以 "UOT-Q" 名义与**零调参预算**的基线比精度。`FINAL_TABLE4_BASELINE_FAIRNESS_AUDIT.md` §1 原文即 "The Table 4 comparison is NOT calibration-parity-symmetric"，判定 `PARTIALLY_DOCUMENTED`。
缓解已做得相当彻底（改名 + 逐项披露特征 + 禁令 + 补 ABCTracer 四项优势 + 适用边界表），但**特征是循环的这一点无法由措辞消除**：它属于"把基线的输出当作自己方法的能力"。三位审查方给出的独立结论一致：彻底解决需 RE-2（移除三项基线特征后重训重排器并在同一保留集 292–311 评估）——而 `PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §4 明确指出这是**唯一成本可控且收益明确的项**（脚本、保留集、评估器均已存在），却**未执行**。
风险等级被压到"已披露的诊断性对照"，但审稿人一旦自行打开 `run_phase29_robust_rcuot_superiority.py`（其路径已写入论文的可复现包），披露与不披露的差别就被抹平。这是**残余杀伤力最高的一项**——因为它是"作者已知道但选择只披露"的类型。

### 3. 可操作精度缺失（R-36，blocker, mitigated；历史唯一 FATAL）
`FINAL_HOSTILE_REVIEW.md` §3 的 FATAL ledger 只记录**一项** FATAL：R3-M1 "No operationally usable operating point"。事实是：流级压力操作点精度 0.192（约每 5 条提升边 4 条为假）、F1 0.316；严格精确恢复 5 方法 × 3 桥全 0；而唯一漂亮的 0.889 精度只存在于一对一区间，**该区间内闭集 Connector 原生就是 0.9736**。
三方独立强化了这一判断：A-M7 判 Critical/Blocking（"the multi-mapping advantage is potential rather than demonstrated"）；B-M3 判 High（"a tool that promotes mostly false edges creates investigative cost"）；C-M4 判 High/Blocking（"不是 SAR 级可操作"）。歧义点是：应用型审稿人一定会问"那么调查者现在能用它做什么"，而论文的答案是"表示能力 + 排序卫生 + 弃权安全层"——这是一个**诚实但很弱的实操定位**。
已做到的缓解是把它提升到摘要级（"gain at decoded precision ≈0.19, not a triage-ready operating point"）并把结论描述词改为 coverage-qualified；但底层数字不可改，且**弃权 gate 本身从未验证**（见 R-35 / B-5），使"安全层"这个卖点也缺乏自身证据。

### 4. 公平预算基线比较在本稿中不存在（R-41 / B-6 相关，major, mitigated）
`R5_HOSTILE_REVIEW_SYNTHESIS.md` K7 的表述最准确："**zero trials is not fairness**——the method arrives pre-tuned from an unbounded development history on the same generator family while only the control is searched"。B-M7 的推论更致命："**if v5 never runs, no fair comparison exists anywhere in the submission**"——而 v5 确实没跑成。
同预算内部对照只有 Balanced-OT（同代价同边际，Δ_bot CI 含零）与 Threshold-MM，而 **Threshold-MM 在边 F1 上 3–4 倍高于传输族**（0.1195–0.1683 vs 0.0273–0.0404），且 τ\*=0.05 落在预注册网格下边界（基线"未被充分调优"却仍然赢）。这两项叠加给出一个非常直观的审稿叙述："**一个未调优的阈值规则在论文自己的基准上以 3–4 倍击败了所提方法族**"。
已落地的是 OPTION B 降级（style 适配基线 → representation-capability controls + 禁用 superiority 措辞 + 写入"本稿不存在公平预算基线比较"）。但这是一次**声明性的自我限定**，不是一个对照实验；配合 R-02 的特征循环性，公平性这条线的整体说服力仍然脆弱。

### 5. 表 7 的不可溯源数字（R-03，blocker, open）
这是当前清单上**唯一仍然完全未闭合的 blocker**。0.978 / 0.959 / 0.010 / 0.833 四个数字既在正文承担实质论证（"近一对一锚定口径上适配基线优于本管线"），又**在冻结 bundle 中无法定位源工件**——唯一在册记录是汇总表自身的第 79–81 行。
`PRE_SUBMISSION_AUTHOR_DECISION_AUDIT.md` §8.3 把它列为"**最高优先级的单项建议**"，理由写得非常克制且致命："它是唯一目前状态不明的项……若 AD-4 无法溯源，删除该行同时会消除 C-3 中来源③ 的混用风险，一举闭合两项 Critical"。**而后续所有轮次（R5B / R5C / 最终作者轮）都没有记录该项的裁决。**
为什么它比"外部性能缺失"更值得优先处理：外部性能缺失是**已被承认的局限**，作者、审稿人、编辑三方对它的定位一致；而不可溯源的数字是**可被直接判定为学术规范问题的类型**（"不可解释的数字比缺失数字更危险"），且审稿人只需把四个数字与 0.833 的 PolyNetwork 来源②（位于**未冻结的 `out/` 目录**）一对，就会形成一个不需要任何专业知识就能提出的质疑。相对而言它的闭合成本最低（一个路径或一次删除），**风险收益比最差**。

---

*本文件为只读取证的汇总产物，未修改 `<REPO>` 内任何其他文件。*
