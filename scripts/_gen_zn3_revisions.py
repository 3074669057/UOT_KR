import os, shutil, win32com.client as win32
src = r"D:\Edge\1\ZN_2.docx"
dst = r"<REPO>\out\ZN_3.docx"
if os.path.exists(dst):
    os.remove(dst)
shutil.copy2(src, dst)

word = win32.Dispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(dst, ReadOnly=False, ConfirmConversions=False)
doc.TrackRevisions = True

def find_para(unique):
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs(i)
        if unique in p.Range.Text:
            return p
    return None

def replace_para(unique, new):
    p = find_para(unique)
    if p is None:
        print("MISSING PARA:", unique[:40])
        return False
    r = p.Range
    if r.End > r.Start:
        r.End = r.End - 1
    r.Text = new
    return True

def insert_caption_after(unique, caption):
    p = find_para(unique)
    if p is None:
        print("MISSING INSERT AFTER:", unique[:40])
        return None
    rng = doc.Range(p.Range.End, p.Range.End)
    rng.InsertAfter("\r" + caption)
    return find_para(caption)

def insert_table_after_caption(caption, data):
    cp = find_para(caption)
    if cp is None:
        print("MISSING CAPTION:", caption[:40])
        return None
    rng = doc.Range(cp.Range.End, cp.Range.End)
    rows = len(data)
    cols = len(data[0])
    tbl = doc.Tables.Add(rng, rows, cols)
    try:
        tbl.Borders.Enable = True
    except Exception:
        pass
    for ri, row in enumerate(data, start=1):
        for ci, val in enumerate(row, start=1):
            tbl.Cell(ri, ci).Range.Text = str(val)
    return tbl

abstract = "跨链桥与去中心化金融的深度融合使反洗钱（AML）取证面临三重结构性挑战：桥特定语义字段并非在所有调查场景下可用；开放候选池规模远超真实匹配对数；资金活动呈现分流、合流与延迟释放等非一对一模式。本文提出证据约束非平衡最优传输（EC-UOT-Q），一种将跨链资金流对应建模为可弃权软传输的方法论框架。该方法将金额、时间、代币类型与合约路径等公开可解码特征统一编码为复合传输代价，通过商空间分组约束候选关系空间，并在证据不足或证据冲突时执行选择性弃权以控制错误升级风险。为检验跨协议适用性，本文在 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥上，以可复现的冻结协议报告原始交易对全量保留集结果，数据全部来自缓存产物，不调用新的链上 API。Celer 的 216 配置系统搜索揭示了公开特征约束下的精度-覆盖边界：无弃权条件下，开放候选池精度上限约为 0.19，全量 F1 约为 0.09；达到精度不低于 0.90 的唯一途径是接近全弃权，覆盖率仅为 0.2%。三桥结果进一步显示，EC-UOT-Q 在证据充分的 PolyNetwork 上达到 precision=0.889、F1=0.833；在证据稀薄的 Multichain 上则以 98.45% 弃权主动拒绝不可靠判断，体现了选择性弃权作为取证安全机制的价值。Connector-style 与 ABCTracer-style 的高分依赖完整桥语义或风格化适配，不等同于原始系统。本文的核心贡献不在于宣称方法优势，而在于通过严谨实验协议为跨链资金流匹配建立可复现、可弃权、可审计的评估边界。"

paras = [
    ("随着区块链生态由单链账本逐渐演化为多链互操作网络", abstract),
    ("为支撑评估，本文基于公开 Celer 跨链桥锚定信号构建了资金流级监督基准", "为支撑评估，本文基于 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定信号构建资金流级监督基准，并从结构恢复、证据覆盖、多桥泛化与独立保留测试四个层面验证所提框架。实验表明，EC-UOT 能够恢复交易级硬匹配无法表达的分流与合流结构，EC-UOT-Q 能在证据覆盖范围内输出可靠且可弃权的高置信对应；在证据稀薄的 Multichain 上，EC-UOT-Q 以高弃权主动拒绝不可靠判断，体现了取证安全边界。需要强调的是，该基准提供的是跨链对应真值而非可疑性标签。"),
    ("（3）基准与系统： 本文基于公开 Celer 跨链桥锚定信号构建了资金流级评测基准", "（3）基准与系统：本文基于 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定信号构建资金流级评测基准，并实现 CrossFlow-Audit 原型系统。通过“非锚定泄漏审计”与“对称掩码退化实验”，系统刻画了该方法在桥语义字段逐级缺失下的渐进退化规律与可适用性边界。"),
    ("在此背景下，开展面向反洗钱调查", "在此背景下，开展面向反洗钱调查的“资金流对应”研究兼具现实意义与迫切必要性。随着跨链异构生态的演进，多跳混币、多协议拆分与金额时间异步释放已成为洗钱犯罪分子阻断链上审计的常态化手段。传统基于单账本的交易级硬匹配方法在面对上述分流、合流与不完整观测时会陷入结构性失效，导致执法机关的取证证据链出现严重断裂。因此，亟需一种能够跨越协议边界、在证据不完整条件下显式刻画流级对应关系与不确定性边界的取证分析框架，以支持跨账本洗钱证据的重组。需要说明的是，带有经核验分流/合流标签的交易级公开轨迹仍然稀缺；因此，本文在定量评估中采用事件锚定的 Celer、Multichain、PolyNetwork 三条 ETH→BNB 交易对作为可复验监督信号[51]，并将受控压力测试中的分流、合流参数与上述真实扇出与归集形态对齐，以降低外推风险并避免过度声明。"),
    ("因此，现有公开资源尚难以同时满足真实跨链交易", "因此，现有公开资源尚难以同时满足真实跨链交易、资金流级对应真值、AML语义标签与可复验性这四项要求。正如引言中所强调的，本文的评测目标是资金流级对应能力，而非全链空间中的非法性分类；这里进一步说明该定位在数据层面的具体含义：后续实验采用 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定事件构建资金流级监督基准，其作用是提供可复验的对应真值，可疑性则被视为实际AML工作流中的上游输入或调查上下文，不作为本文的直接监督目标。这一区分也决定了本文实验应被解释为对应恢复评估，而非可疑性检测评估。"),
    ("据此，本文明确区分已评估分量与预留接口", "据此，本文明确区分已评估分量与预留接口：金额、时间因果、路径、图上下文与证据质量代价，以及证据质量加权边际，均在 Celer、Multichain、PolyNetwork 三条 ETH→BNB 锚定基准上参与评估；风险一致性代价与风险加权边际因缺乏风险标签而默认关闭，仅作为面向真实 AML 工作流的扩展接口保留，其有效性留待具备风险标签的数据上验证。"),
    ("当前公开数据资源尚难以直接支持真实跨链 AML 正负标签评测", "当前公开数据资源尚难以直接支持真实跨链 AML 正负标签评测。已有公开 AML 数据集主要集中在单链 Bitcoin 或单链交易图场景，通常提供 licit/illicit 交易标签、账户风险标签或洗钱子图标签；已有公开跨链数据则更多服务于桥交易测量、跨链活动统计、桥异常检测或攻击分析，通常不提供 AML 语义下的可疑/非可疑资金流标签。因此，本文不将实验设计为跨链非法性分类任务，而是将其限定为跨链资金流对应任务。具体而言，本文使用 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定事件构建可复验的源链与目标链对应真值，并在该基准上评估 EC-UOT 与 EC-UOT-Q 是否能够恢复资金流级对应、表达分流/合流结构，并在证据不足时执行低置信输出或弃权。"),
    ("为避免过度声明，本文将每一类实验结论限定在相应的证据作用域内", "为避免过度声明，本文将每一类实验结论限定在相应的证据作用域内。真实 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定信号用于构建可复验的资金流级监督基准；半合成结构压力测试用于隔离验证分流、合流、未匹配质量和延迟噪声下的结构恢复能力；覆盖限定推断用于检验 EC-UOT-Q 在事件证据充分子集上的可靠性；联合时间可采解码用于给出本文最终主结果；leave-anchor-out 泄漏审计用于排除桥锚定证据泄漏导致的循环评估；对称掩码退化实验则用于说明原版 Connector 与 EC-UOT-Q 在桥语义逐级缺失条件下的可适用性边界。"),
    ("由于未获得 ABCTracer 的官方模型检查点", "由于未获得 ABCTracer 官方检查点和可复现的官方推理实现，也因原始 Connector 与开放候选池输入格式不兼容，本文在 Multi/Poly 主结果中使用可复现的 Connector-style adapted 与 ABCTracer-style adapted 基线；Celer 上仍保留原始 Connector 的闭集原生诊断作为上界参照。适配基线不等同于原始系统，本文不以风格化重实现替代官方结果。"),
    ("本文基于 Celer 跨链桥构建资金流级监督基准", "本文基于 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥构建资金流级监督基准。监督信号来自三组桥锚定交易对，每个锚定对连接一笔 Ethereum 源链交易和一笔 BNB Smart Chain 目标链交易。本文不对锚定关系进行启发式重标注，而是将其作为构造资金流级标签的基础证据。"),
    ("在资金流构建阶段，系统按照主地址、链标识、资产上下文和滚动时间窗口聚合桥相关活动", "为检验构建过程能否稳定产生流级监督信号，表 3 汇总三条桥的锚定交易对、源侧样本数、保留测试集源交易数以及时间窗口下的分割鲁棒性。"),
    ("表 3显示，7,296 个 Celer 锚定交易对", "表 3 显示，Celer、Multichain、PolyNetwork 分别提供 7,296、8,349、5,460 个锚定交易对；按时间顺序分割后，Celer、Multi、Poly 的保留测试集分别包含 5,107、2,444、1,630 个源交易，候选池召回率分别为 0.991、1.000、1.000。不同桥的源侧与目标侧资金流数量并不一一等长，说明非一对一结构在三条桥上普遍存在。"),
    ("从对应模式看，一对一关系占 72.32%", "从对应模式看，Celer 基准中一对一关系占 72.32%，多对一占 27.51%，一对多占 0.17%；Multi/Poly 同样呈现显著的合流结构。这说明仅以交易哈希为单位的一对一硬匹配不足以表达全部监督关系，多对一结构在监督标签中占有不可忽略的比例。该现象为本文将跨链追踪问题提升为资金流级对应问题提供了直接实验动机。"),
    ("需要强调的是，Celer 锚定对提供的是跨链对应真值", "需要强调的是，三桥锚定对提供的是跨链对应真值，而不是非法性或可疑性标签。因此，本节评估的是 correspondence recovery，而不是 suspiciousness detection。"),
    ("因此，RQ1 的答案是肯定的", "因此，RQ1 的答案是肯定的：在公开跨链 AML 正负标签缺乏的条件下，Celer、Multichain、PolyNetwork 三条 ETH→BNB 锚定信号能够支持可复验的资金流级对应基准，并为后续结构恢复、覆盖限定推断和基线比较提供监督基础。"),
    ("RQ1 得到肯定回答:Celer 跨链锚定信号", "RQ1 得到肯定回答：Celer、Multichain、PolyNetwork 三条 ETH→BNB 锚定信号能够支持构建可复验的资金流级 CSFFC 基准，并暴露出非一对一合流结构（表 3）；三桥的候选池召回率接近或等于 1.0，表明低召回并非候选检索缺失所致。"),
    ("RQ2 得到肯定回答:EC-UOT 能够在受控", "RQ2 得到肯定回答：EC-UOT 能够在受控的半合成压力测试中恢复分流、合流等非一对一资金流结构（表 4），但排序层面的精确定位仍受候选相似性与诱饵噪声的影响；真实桥数据上的固定解码器消融（表 5）进一步证明，这一恢复能力并非任意简单匹配器或代价排序方法所能替代，而是确实来自 EC-UOT 传输机制本身。"),
    ("RQ3 得到肯定回答:EC-UOT-Q 的证据限定机制", "RQ3 得到肯定回答：EC-UOT-Q 的证据限定机制既能在事件覆盖的证据充分子集内给出可靠对应（表 6），也能在 Celer 全部 7,296 个锚定对上形成高 precision 主取证操作点（表 7、表 8）；在 Multi/Poly 原始交易对全量保留集上，覆盖与弃权随证据充分程度变化（表 13）。"),
    ("RQ5 得到肯定回答:对称掩码实验", "RQ5 得到肯定回答：对称掩码实验（表 10、表 11）表明 EC-UOT-Q 在桥语义缺失时展现出原版 Connector 所不具备的可适用性边界与优雅退化特征；三桥主表（表 13）进一步表明，在证据稀薄的 Multichain 上 EC-UOT-Q 以高弃权拒绝不可靠判断，体现取证安全边界。"),
    ("总体而言,本章的实验结果表明", "总体而言，本章的实验结果表明：CSFFC 基准能够支持流级跨链 AML 对应关系的评估；EC-UOT 能够在受控条件下有效建模分流、合流与未匹配质量；EC-UOT-Q 能够在证据覆盖范围内输出可靠对应，并在证据不足时通过弃权控制错误升级；三条 ETH→BNB 桥的主表进一步显示，选择性弃权在证据稀薄的 Multichain 上形成了可解释的取证安全边界。"),
    ("本文提出 EC-UOT/EC-UOT-Q 框架，将跨链 AML 调查中的资金流对应问题", "本文提出 EC-UOT/EC-UOT-Q 框架，将跨链 AML 调查中的资金流对应问题表述为证据约束、可弃权的非平衡最优传输。实验贡献包括：(1) 通过预注册的 216 配置系统搜索，首次系统刻画了公开特征约束下的精度-覆盖前沿；(2) 消融实验揭示了运输机制本身的判别力与分组策略的信息瓶颈之间的张力；(3) 在 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥上建立可复现的多协议主基准，并明确基线适配与原版系统的差异；(4) 严格的禁止字段审计与原子冻结协议确保了全部结果的可复现性。核心实证发现表明，公开特征下精度与覆盖严格互斥，EC-UOT-Q 的差异化价值在于证据不足时主动弃权，而非在桥语义完备时声称超越交易级匹配方法。"),
    ("基于 Celer 资金流级基准与独立保留测试集的实验表明", "在 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥的联合主基准上，EC-UOT 能够恢复分流与合流结构，EC-UOT-Q 能够在证据覆盖范围内输出可靠对应；在证据充分的 PolyNetwork 上，EC-UOT-Q 的原始交易对全量保留集 precision 为 0.889、F1 为 0.833，而在证据稀薄的 Multichain 上以 98.45% 弃权主动拒绝不可靠判断。Celer 上的 leave-anchor-out 泄漏审计表明，对应恢复能力来自标签结构本身，而非桥锚定证据泄漏。Connector-style 与 ABCTracer-style 的高分依赖完整桥语义或风格化适配，不能等同于原始系统；EC-UOT-Q 的差异化价值，在于桥语义缺失时依然可评估、可运行、可弃权。"),
    ("总体而言，本文为跨链 AML 调查提供了一种证据限定、可审计且可弃权的资金流对应框架", "总体而言，本文为跨链 AML 调查提供了一种证据限定、可审计且可弃权的资金流对应框架。当事件证据不足、候选池不完整，或金额/时间约束发生冲突时，框架不会强行输出确定性结论，而是将高置信声明严格限制在链上证据可支撑的范围之内；Multichain 上接近全弃权的行为正是这一取证安全机制的直接体现。未来工作将扩展到更多跨链桥协议与链上数据后端，并在真实调查流程中评估该框架的泛化能力与实用价值。"),
    ("本文方法的适用范围", "本文方法的适用范围，首先由监督信号本身的性质所决定：训练与评测所使用的真值来自 Celer、Multichain、PolyNetwork 三条 ETH→BNB 桥锚定对，而非司法确认结果或机构内部的洗钱案件标签。因此，本文所验证的是桥事件监督下的资金流对应能力及其在 AML 分诊场景中的适用性，而不是端到端的洗钱检测准确率。"),
]

for unique, new in paras:
    ok = replace_para(unique, new)
    print("PARA", ok, unique[:25])

cap3 = "表 3（可编辑版）三桥监督基准统计"
insert_caption_after("为检验构建过程能否稳定产生流级监督信号", cap3)
data3 = [
    ["协议", "锚定交易对", "源侧样本数", "保留测试集源交易", "候选池召回率"],
    ["Celer", "7296", "7296", "5107", "0.991"],
    ["Multichain", "8349", "8349", "2444", "1.000"],
    ["PolyNetwork", "5460", "5460", "1630", "1.000"],
]
insert_table_after_caption(cap3, data3)

cap13 = "表 13 三桥原始交易对全量保留集结果"
insert_caption_after("因此，RQ1 的答案是肯定的", cap13)
data13 = [
    ["协议", "方法", "n_test", "候选池召回率", "precision", "recall", "full-set F1", "coverage", "abstention"],
    ["Celer", "RC-UOT-Q", "5107", "0.991", "0.124", "0.060", "0.081", "0.481", "0.519"],
    ["Multichain", "RC-UOT-Q", "2444", "1.000", "0.316", "0.005", "0.010", "0.016", "0.984"],
    ["PolyNetwork", "RC-UOT-Q", "1630", "1.000", "0.889", "0.783", "0.833", "0.881", "0.119"],
    ["Multichain", "Connector-style adapted", "2444", "1.000", "0.993", "0.964", "0.978", "0.971", "0.029"],
    ["PolyNetwork", "Connector-style adapted", "1630", "1.000", "0.999", "0.998", "0.998", "0.999", "0.001"],
    ["Multichain", "ABCTracer-style adapted", "2444", "1.000", "0.959", "0.959", "0.959", "1.000", "0.000"],
    ["PolyNetwork", "ABCTracer-style adapted", "1630", "1.000", "0.998", "0.998", "0.998", "1.000", "0.000"],
]
insert_table_after_caption(cap13, data13)

doc.SaveAs2(dst, FileFormat=16)
doc.Close(False)
word.Quit()
print("SAVED", dst)