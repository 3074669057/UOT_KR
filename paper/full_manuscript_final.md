# Non-One-to-One Cross-Chain Forensic Fund Flow Correspondence: Transport Representation and Conditional Decoding


---

# Abstract

Cross-chain laundering fragments suspicious asset trails across bridges, tokens, and blockchains. Existing transaction-level tracing tools struggle with splitting, merging, bridge latency, amount drift, and incomplete event evidence, making one-to-one transaction matching fragile for operational anti-money-laundering (AML) analysis. We introduce **Cross-Chain Suspicious Fund Flow Correspondence (CSFFC)**, a flow-level task in which one source-chain flow may correspond to several destination-chain flows, and several source flows may consolidate into one destination flow, under amount, timing, risk, path, and event constraints. Supervision is built from one-to-one protocol-native bridge anchors with synthetic non-one-to-one topology imposed on real flow features. We propose an **unbalanced optimal transport (UOT) formulation** with risk-weighted inputs that models soft split/merge-aware correspondence with explicit unmatched mass, and **UOT-Q**, an evidence-aware extension with coverage-qualified abstention when evidence is insufficient. Controlled structural experiments show that the transport representation can express non-one-to-one candidate correspondences that one-to-one decoders cannot represent, although strict exact-topology recovery is zero for every tested method at the frozen decode, and a calibrated threshold rule dominates edge F1 (Table 2d).

We further diagnose that raw transport-plan values are not pure correspondence affinities: their rankings are distorted by transport scaling and marginal pressure. Directional conditional decoding cancels the conditioned side's scaling exactly—a dual-cancellation analysis we state for completeness—repairing this ranking distortion; we position the decoder explicitly as an adaptation of standard mutual-matching normalization, not as a new normalization. A **preregistered, hash-locked, one-shot confirmatory holdout on untouched seeds across three bridges** (Celer, Multichain, PolyNetwork) confirms the repair on semi-synthetic structural grids—a within-method contrast between two decoders of the same frozen plan—with conditional dual-cancelled decoding raising macro edge F1 from **0.2350 to 0.3104** (**Δ_primary = +0.075413**; paired 95% CI **[0.070557, 0.080312]** (bridge-stratified template-instance bootstrap, B = 4,000; aggregation hierarchy template → seed → bridge → macro)), positive on every bridge, with all preregistered mechanism directions satisfied and an independent verifier recomputing every result; this gain is measured at decoded edge precision ≈0.19 and is a within-method ranking-repair result, not a triage-ready operating point; the same experiment establishes no unbalanced-relaxation-specific advantage. We release **Cross AML**, an open-source evidence-to-report prototype for investigator triage. Two prediction-blind post-development data audits documented real flow-level fan-out outside the development corpus, concentrated in a few persistent addresses (§4.9), but both failed preregistered behavioral-diversity criteria; consequently, no external method-performance claim is made.

# 1. Introduction

Cross-chain money laundering increasingly appears as **coordinated fund-flow patterns** rather than isolated transactions [@chainalysis2024laundering; @elliptic2025crosschain]. Attackers route assets through bridges, decentralized exchanges, multi-hop paths, and address aggregation, splitting and merging value across Ethereum, BNB Smart Chain, and other networks. Investigators therefore need to know whether a suspicious **source-chain flow** corresponds to a suspicious **destination-chain flow**—not merely whether two transaction hashes look similar.

This paper addresses that need with a flow-level formulation, an unbalanced transport (UOT) solver with risk-weighted inputs, an evidence-aware quotient extension, and a standalone triage tool.

## 1.1 Problem motivation

Single-chain AML tracing assumes that suspicious activity can be followed along contiguous transaction graphs on one ledger [@meiklejohn2013fistful; @reid2011analysis]. Cross-chain laundering breaks that assumption: the same economic movement may surface as several bridge-facing transactions, token swaps, and delayed arrivals on different chains. Regulators now explicitly flag **chain hopping** and cross-bridge typologies as elevated virtual-asset risks [@fatf2023virtualassets]. Analysts must reconstruct **cross-chain fund-flow correspondence** under partial observability, bridge latency, and protocol-specific event formats.

Operational questions are inherently **flow-level**: Did funds leaving address *A* on chain *X* plausibly arrive as related activity around address *B* on chain *Y*? How much mass moved, through which route, and with what timing? Transaction hashes remain essential evidence, but the unit of analysis is the **flow**—a temporally and semantically coherent bundle of bridge-relevant activity.

## 1.2 Why transaction-level tracing is insufficient

Adapted transaction-level tracers such as **CONNECTOR** [@lin2025connector] and **ABCTracer** [@zheng2025abctracer] are valuable references for cross-chain retrieval on DeFi bridges. Nevertheless, several structural mismatches motivate a flow-level formulation:

- **Split and merge.** One source activity may fan out to multiple destinations, or multiple sources may consolidate into one destination. Hard one-to-one tx matching cannot represent soft mass allocation.
- **Delay.** Bridge finality and routing introduce lags that break naive hash-level alignment.
- **Amount drift.** Fees, slippage, and partial transfers change nominal amounts between legs.
- **Incomplete event evidence.** Bridge logs, token transfers, and receipt fields may be missing, mis-parsed, or unavailable from a given RPC tier.
- **Bridge event observability.** Not every bridge episode exposes the same event schema; observability varies by protocol and archive depth.
- **Fragile one-to-one matching.** High tx hit rate on predominantly single-anchor mappings does not imply robust correspondence under split/merge stress or evidence gaps.

CSFFC is designed for these conditions rather than for replacing transaction-level tools outright.

## 1.3 CSFFC formulation

**CSFFC (Cross-Chain Suspicious Fund Flow Correspondence)** formalizes soft, mass-bearing links between source flows \(\mathcal{S}\) and destination flows \(\mathcal{T}\). A **flow** is an AML **observation unit**: transactions aggregated by primary address, chain, asset context, and a rolling time window (canonical: 1800 seconds). Flows are not native on-chain atoms; they reduce per-transaction noise while preserving amount, timing, route, and evidence references needed for correspondence [@weber2019amlbitcoin].

CSFFC seeks a non-negative transport plan \(P\) where \(P_{ij}\) allocates mass from source flow \(s_i\) to destination flow \(t_j\). Supervised labels are directed flow edges with pattern metadata (`one_to_one`; legacy fields `many_to_one` / `one_to_many`, which canonically denote 1→N fan-out and N→1 merge respectively—see the Table 1 topology note). Evaluation separates real Celer supervision, semi-synthetic structural stress, and diagnostic unmatched/decoy analyses.

## 1.4 Method overview

The **UOT formulation** (repository identifier RC-UOT(-Q) retained for backward compatibility; “R” denotes risk weighting, not a hard constraint) — solves CSFFC via entropy-regularized **unbalanced** optimal transport [@cuturi2013sinkhorn; @chizat2018unbalanced; @villani2009optimal]. Pairwise costs combine amount error, temporal causality, bridge/path consistency, AML risk, graph context, and evidence quality. Unbalanced marginals allow **unmatched mass**, modeling partial observation instead of forcing full assignment. Risk and evidence terms shape effective supply and demand. The solver outputs a soft transport plan, decoded into reported edges by a **dual-cancelled conditional decoder** (§3.4) that repairs ranking distortion introduced by the raw plan's dual scalings.

**UOT-Q** extends UOT with **event-backed quotient groups**. When exact flow pairs are ambiguous, bridge-event evidence collapses candidates into quotient groups. **Coverage qualification** assigns tiers (A/B/C/Uncovered): high-confidence inference is permitted only on event-backed covered pairs; uncovered edges **abstain**. This is an AML safety feature—unsupported hypotheses are not promoted to confirmed matches.

## 1.5 Tool overview

We release **Cross AML** (`tools/cross_aml/`), a standalone open-source prototype for evidence-qualified cross-chain AML triage. Given source and optional destination transaction hashes, the tool:

1. Collects RPC evidence (transactions, receipts, logs, ERC-20 transfers, bridge-like events);
2. Constructs AML flows and features;
3. Builds quotient groups and applies coverage gates;
4. Runs UOT-Q-style matching with abstention; and
5. Emits interpretable `report.json` / `report.md` outputs with evidence bundles and investigator guidance.

The tool supports **dry-run** (mock evidence, no RPC) and **live RPC** modes, plus **pair_scoring** and **candidate_discovery** workflows. Candidate discovery without an indexer, archive node, or trace backend is explicitly limited—standard RPC alone cannot guarantee large-scale destination search.

## 1.6 Contributions

We make four contributions:

1. **Non-one-to-one forensic formulation.** We formalize Cross-Chain Suspicious Fund Flow Correspondence (CSFFC): flow-level, mass-bearing correspondence in which one source-chain flow may correspond to several destination-chain flows and vice versa, with explicit unmatched mass, partial observability, and coverage-qualified abstention; protocol-level one-to-one settlement is distinguished from flow-level fan-out/merge structure.

2. **Forensic task formulation, diagnosis, and abstention design.** We formalize CSFFC as a flow-level soft correspondence task, design coverage-qualified abstention as a safety mechanism, and diagnose that raw transport-plan values mix correspondence affinity with transport scaling and marginal pressure. The conditional decoder is an adaptation of standard mutual-matching normalization to transport-plan decoding, validated by a preregistered confirmatory holdout; its contribution is the ranking-distortion diagnosis and the forensic adaptation, not the normalization itself.

3. **Faithful structural-capability evaluation.** Across Celer, Multichain, and PolyNetwork, a code-level capability audit and a unified strict evaluation establish what one-to-one decoders structurally cannot represent, and bound what the transport decode actually achieves (edge-inclusion recall at low decoded precision; strict exact-topology recovery zero for all tested methods; a calibrated threshold rule dominating edge F1).

4. **Preregistered confirmatory decoder-repair evidence.** A hash-locked one-shot holdout on untouched seeds across three bridges confirms the repair (Δ_primary = **+0.0754** macro edge F1; paired 95% CI **[0.0706, 0.0803]** (template-instance bootstrap; aggregation hierarchy template → seed → bridge → macro); positive on every bridge; independent verifier), and a data-only independent temporal audit (§4.9) confirms real flow-level fan-out existence outside the development corpus while deliberately making no external-performance claim.

On a development-sealed holdout, UOT-Q reports a higher **precision/F1 Pareto frontier** than the untuned heuristic and style-adapted representation controls (Table 4)—**not original Connector or ABCTracer system runs**; original-system Connector applicability and native closed-set diagnostic are reported separately (Table 6; Appendix B). We do **not** claim that UOT-Q dominates original Connector in native bridge-semantics mode. Full interpretation appears in §5 and §6.

# 2. Related Work

We organize prior work into five themes aligned with the bibliography (`references.bib`) and clarify how CSFFC, UOT, and UOT-Q differ from transaction-level tracing and standard optimal transport.

## 2.1 Blockchain AML and cryptocurrency forensics

Blockchain AML systems track suspicious activity through address clustering, heuristics, graph propagation, and exchange-facing intelligence [@meiklejohn2013fistful; @reid2011analysis; @androulaki2013evaluating; @phillips2018ransomware]. Econometric and forensic studies further quantify illicit use of cryptocurrencies at ecosystem scale [@foley2019illegal]. Foundational deanonymization studies showed that pseudonymous ledgers still leak structure when transactions are aggregated over time [@meiklejohn2013fistful]. Subsequent work introduced machine-learning pipelines over transaction graphs, notably the **Elliptic** dataset and graph convolutional models for illicit-transaction detection [@weber2019amlbitcoin; @pareja2020evolvegcn]. Recent extensions move from node-level labels to **subgraph-level** laundering patterns [@weber2024elliptic2]. Surveys and taxonomies document cryptocurrency fraud types, forensic workflows, and regulatory responses [@harvey2020cryptocurrency; @kipp2021taint; @monamo2016unsupervised; @nadarajah2018characterizing].

Industry intelligence reports quantify the scale of crypto laundering and the growing share routed through bridges and chain hopping [@chainalysis2024laundering; @elliptic2025crosschain]. These sources motivate cross-chain analysis but do not provide open, flow-level correspondence labels—our Celer-supervised benchmark addresses that experimental gap.

**Positioning.** CSFFC builds on single-chain AML graph learning but shifts the unit of analysis from isolated transactions to **cross-chain flows** with explicit split/merge and partial observability.

## 2.2 Cross-chain tracing and bridge analysis

Cross-chain bridges enable asset portability across heterogeneous ledgers but create observability seams: deposits and withdrawals appear on different chains with delays, fee adjustments, and protocol-specific logs [@loesch2019towards; @li2023demystifying]. Security research documents DeFi bridge exploits, systemic risks, and attack surfaces [@zhou2023sok; @qin2022rise; @apostolaki2019hijacking]. Our benchmark is anchored on **Celer/cBridge**-style bridge traffic [@thibodeau2022celer], a widely deployed route between Ethereum and BNB Smart Chain.

**CONNECTOR** [@lin2025connector] is the closest published DeFi bridge traceability system: it extracts generic features from bridge contract traces, mines execution logs, and associates deposit/withdrawal transaction pairs without relying on centralized internal ledgers. It reports high deposit identification and withdrawal association rates on thousands of real bridge pairs. **ABCTracer** [@zheng2025abctracer] extends the line with bidirectional tracing, NER over bridge-agnostic cues, and information retrieval for bridge-specific implicit cues—achieving strong F1 on multiple mainstream bridges and case studies on attacks and laundering flows.

Earlier cross-chain efforts often target **CeFi** bridges via centralized APIs or rule templates [@yousaf2023crosschain], which do not generalize to permissionless DeFi contracts. CSFFC instead uses **open on-chain evidence** (logs, transfers, receipts) and targets **flow-level soft correspondence** rather than hard tx-pair association alone.

**Positioning.** In Table 4, CONNECTOR and ABCTracer appear as **heuristic and style-adapted diagnostic baselines** mapped to our flow labels—not as original-system runs. **Original-system Connector behavior** is evaluated separately (Table 6; Appendix B). We do not claim to replace CONNECTOR/ABCTracer tx-retrieval strengths, but we address split/merge mass, unmatched observation, and coverage-qualified abstention absent in their native formulations. According to the audited literature, neither CONNECTOR nor ABCTracer provides a direct representation-capable non-one-to-one flow-level output: both emit one-to-one transaction associations, and we found no published cross-chain tracing system with a native flow-level non-one-to-one correspondence output (audit coverage recorded in the reproducibility package; sources assessed at abstract level only are treated as uncertain).

## 2.3 Graph learning, transaction matching, and diagnostic adaptations

Graph neural networks and temporal graph models are standard for blockchain fraud detection [@hamilton2017inductive; @weber2019amlbitcoin; @pareja2020evolvegcn; @khalil2018iotc]. Account-based chains introduce richer internal-call structure; **TRacer** scales personalized PageRank-style search for money-flow tracing on account-based graphs [@wang2022tracer]. Ethereum-focused work detects phishing and scam neighborhoods via graph classifiers [@chen2021phishing]. These methods excel at **ranking suspicious nodes or edges on one chain** but do not natively output cross-chain transport plans.

**CONNECTOR** [@lin2025connector] and **ABCTracer** [@zheng2025abctracer] are the primary **cross-chain** references for our Table 4 diagnostic adaptations. CONNECTOR emphasizes rule- and log-driven association on bridge contracts; ABCTracer adds learned explicit/implicit cues and bidirectional retrieval. In Table 4 they are **mapped to flow IDs** for Pair-F1 comparison on a frozen Celer holdout as Connector-inspired and ABCTracer-inspired diagnostic baselines—not original-system evaluations. They perform strongly where labels are dominated by single-anchor pairs; high diagnostic adaptation metrics therefore bound what tx-level retrieval achieves on adapted flow labels—not what flow-level soft transport with unmatched mass achieves, and not original Connector native performance (Appendix B: F1 = **0.9736** under closed-set native semantics).

**Positioning.** UOT-Q complements graph-learning tracers by modeling **mass allocation** between flow sets, explicit **unmatched** mass, and **coverage tiers** when event evidence is incomplete.

## 2.4 Optimal transport and partial/unbalanced correspondence

Optimal transport (OT) provides a principled framework for matching distributions with soft assignment [@villani2009optimal; @peyre2019computational]. Entropic regularization yields scalable **Sinkhorn** iterations for discrete transport [@cuturi2013sinkhorn]. **Unbalanced OT** relaxes marginal constraints so mass need not be fully conserved—appropriate when destination pools are incomplete, hidden sinks exist, or decoys absorb mass [@chizat2018unbalanced; @chizat2018scaling]. Bregman and iterative scaling schemes further stabilize large-scale transport [@benamou2014iterative]. OT losses have also been used in machine learning for distribution matching [@frogner2015learning; @sejourne2019sinkhorn].

UOT instantiates unbalanced OT on AML **flow** distributions with decomposed, risk-aware costs and decodes transport plans to reported edges. UOT-Q adds a **quotient layer** that groups ambiguous candidates using bridge events before coverage-qualified inference—moving OT from raw pair matching to **evidence-scoped** hypothesis generation. We do **not** claim OT replaces domain-specific bridge parsers; rather, it structures correspondence when multiple destinations compete for source mass.

**Conditional decoding and mutual-matching prior art.** The ingredients of our decoder are standard, and we do not claim them as novel. The factorization \(P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)\) is textbook entropic-OT structure [@cuturi2013sinkhorn; @peyre2019computational]. Bidirectional score normalization with mutual top-*k* selection belongs to the dual-softmax / mutual-nearest-neighbour family of feature matching [@rocco2018neighbourhood; @sun2021loftr], and Sinkhorn transport combined with mutual selection is SuperGlue [@sarlin2020superglue]. Our contribution at this level is the **diagnosis** that raw transport-plan decoding in cross-chain flow correspondence mixes kernel ranking with solver dual scalings, the algebraic cancellation **analysis** (§3.4, stated for completeness as a modest corollary of Sinkhorn structure), and its **preregistered confirmatory validation** (§4.3.5). The prior-art audit's source-level confirmations are partial for some entries (abstract/slide level), so these works are cited as lineage context, not as precise formula-level attributions.

**Positioning.** We found no published blockchain-forensics system coupling OT with cross-chain AML correspondence in the audited literature; our contribution is the **CSFFC** formulation and the unbalanced (UOT) solver with risk-weighted inputs, not a generic OT speed record.

## 2.5 Explainability, auditability, and deployment boundaries

Virtual-asset regulation requires VASPs and jurisdictions to implement risk-based AML/CFT controls for transfers, mixing, and cross-border flows [@fatf2021updatedguidance; @fatf2023virtualassets]. Automated outputs in high-stakes settings should be **interpretable** and auditable rather than opaque scores [@rudin2019stop; @arrieta2020explainable; @adadi2018peeking]. RegTech surveys highlight human review, audit trails, and model governance in production AML stacks [@liu2022compliance].

Cross AML operationalizes these principles via coverage tiers, abstentions, evidence bundles, and explicit claim boundaries in each report. **Coverage qualification** aligns with AML practice: when bridge observability is inadequate, the system withholds high-confidence claims rather than guessing [@fatf2023virtualassets]. Reproducibility norms for security and ML artifacts further motivate frozen benchmarks and open evaluation scripts [@pineau2021improving; @chen2019reproducibility; @lamprecht2019research].

**Positioning.** Our deployment boundary is **investigation triage**, not autonomous adjudication or legal proof. Industry reports [@chainalysis2024laundering; @elliptic2025crosschain] contextualize threat scale but must not be cited as evidence of model superiority.

## 2.6 Positioning summary

| Aspect | Transaction-level tracers (diagnostic adaptations; Table 4) | CSFFC / UOT-Q |
|--------|-----------------------------------|------------------|
| Unit of analysis | Transaction hash / retrieval hit | AML flow |
| Split / merge | Implicit or post-hoc mapping | Soft transport mass |
| Unmatched observation | Often forced assignment | Explicit unmatched mass |
| Evidence gaps | Often silent | Coverage tiers + abstention |
| Primary claim scope | Strong tx retrieval on Celer mapping | Flow correspondence + conditional decoding repair + covered quotient + precision/F1 Pareto difference vs untuned representation controls |

Our contribution is not universal dominance over original Connector, original ABCTracer, or their diagnostic adaptations [@lin2025connector; @zheng2025abctracer], but a **flow-level, evidence-aware, coverage-qualified** formulation with auditable tooling and frozen empirical boundaries reported in §4–§5.

# 3. Method: Evidence-Aware Flow Correspondence with UOT-Q

This section presents the methodological core of our framework: how on-chain evidence is organized into AML flows, how the **UOT formulation** computes unbalanced correspondence with risk-weighted inputs, how **UOT-Q** refines ambiguous cases via event-backed quotient groups, and how **coverage qualification** enforces abstention when evidence is insufficient. The design targets cross-chain AML workflows that require soft split/merge modeling, auditable hypotheses, and explicit withholding of unsupported claims.

## 3.1 Design Goals and System Assumptions

Cross-chain AML correspondence differs from single-chain graph tracing in four structural ways that drive our method design.

**Flow-level modeling.** Investigators reason about coordinated movements—bundles of bridge-facing deposits, transfers, and withdrawals—not isolated transaction hashes. The method therefore operates on **flows** as the primary observation unit while retaining transaction-level provenance.

**Soft correspondence.** A source movement may fan out to several destinations or consolidate from several sources. Hard one-to-one matching cannot represent partial mass allocation. We model correspondence as a **non-negative transport plan** between source and destination flow sets.

**Evidence qualification.** Bridge observability is protocol- and RPC-dependent. Correspondence hypotheses must be conditioned on which events, logs, and receipts were actually observed, not assumed complete.

**Abstention and auditability.** High-stakes AML triage should not promote unsupported links to confirmed matches. When event backing or candidate completeness is inadequate, the method **abstains** or emits diagnostic-only candidates. Every promoted hypothesis should be traceable to cited evidence and investigator-action guidance [@fatf2023virtualassets; @rudin2019stop].

We assume investigators supply seed transaction hashes on known source and destination chains; evidence is collected from standard JSON-RPC interfaces (and optional trace tiers when available). Outputs are **investigation leads** for human review—not legal conclusions or autonomous adjudication.

The end-to-end logical pipeline is summarized below and illustrated in *Figure 1*.

```text
transaction hashes
  -> evidence collection (RPC or controlled replay)
  -> event parsing (token transfers, bridge-like logs)
  -> flow construction and feature extraction
  -> event-backed quotient grouping
  -> coverage qualification
  -> UOT-Q correspondence scoring
  -> interpretable correspondence report
```

*Figure 1: Evidence-aware correspondence pipeline. Hashes are converted into parsed events, AML flows, quotient groups, coverage tiers, transport-based scores, and investigator-facing reports.*

## 3.2 Evidence Representation and Flow Construction

**Primitive evidence.** The lowest layer comprises on-chain artifacts: transaction payloads and receipt status; **ERC-20 Transfer** and related token events; **bridge-like logs** parsed from contract ABIs; block timestamps; and, when available, internal-call or trace data. Together these define addresses, nominal amounts, assets, routes, and temporal order.

**Flows as AML observation units.** A **flow** is not a native on-chain object. It is a constructed unit that aggregates bridge-relevant activity for correspondence analysis. Flows are keyed by primary address (role-specific per chain), chain identifier, asset context (asset group when available, otherwise route identifier), and a rolling time window (canonical: **1800 seconds** in our benchmark). A new flow begins when the key changes or the inter-arrival gap exceeds the window. This aggregation reduces per-transaction noise while preserving split/merge structure: multiple transactions may belong to one flow, and transport mass may link one source flow to several destinations.

**Formal flow record.** We write each flow as

\[
s_i = (A_i,\, c_i,\, g_i,\, \tau_i,\, x_i,\, E_i),
\]

where \(A_i\) is the primary address set, \(c_i\) the chain, \(g_i\) the asset/route context, \(\tau_i\) temporal summaries (e.g., first/last event time), \(x_i\) a feature vector (amounts, risk scores, graph embeddings), and \(E_i\) a multiset of **evidence entries** referencing supporting transactions, logs, and parsed bridge events. Destination flows \(t_j\) are defined analogously on the target chain.

**Correspondence structure without discarding transactions.** Transaction-level hashes and event identifiers remain attached through \(E_i\). Correspondence is scored at flow granularity, but reports can expand any flow edge back to its evidentiary support for audit.

**Evidence collection.** For each supplied hash, the system retrieves transaction and receipt fields, decodes logs, applies bridge parsers, and aligns timestamps across chains. In controlled evaluation settings, evidence may be replayed from frozen exports; in operational use, live RPC endpoints supply the same fields subject to node tier and retention limits.

## 3.3 UOT: Unbalanced Flow Matching with Risk-Weighted Inputs

Given finite source set \(\mathcal{S} = \{s_1,\ldots,s_n\}\) and destination set \(\mathcal{T} = \{t_1,\ldots,t_m\}\), **UOT** seeks a transport plan \(P \in \mathbb{R}_{\ge 0}^{n \times m}\) where \(P_{ij}\) allocates correspondence mass from \(s_i\) to \(t_j\).

**Pairwise cost.** Edge costs combine normalized subterms for amount mismatch, temporal causality and delay feasibility, bridge/route consistency, AML risk alignment, and evidence quality:

\[
C_{ij} = w_{\mathrm{amt}}\, c_{\mathrm{amt}}(i,j)
      + w_{\mathrm{time}}\, c_{\mathrm{time}}(i,j)
      + w_{\mathrm{route}}\, c_{\mathrm{route}}(i,j)
      + w_{\mathrm{risk}}\, c_{\mathrm{risk}}(i,j)
      + w_{\mathrm{evid}}\, c_{\mathrm{evid}}(i,j).
\]

Weights \(w_{\cdot}\) are fixed or dev-tuned on training splits; risk and evidence terms also shape effective supply and demand when marginal masses are adjusted for label confidence or observability.

**Unbalanced optimal transport objective.** Let \(\mu \in \mathbb{R}_{\ge 0}^n\) and \(\nu \in \mathbb{R}_{\ge 0}^m\) denote nonnegative source and destination marginal mass vectors over flows. Source and destination total masses need not be equal (\(\mathbf{1}_n^\top \mu \neq \mathbf{1}_m^\top \nu\) in general). UOT solves

\[
\min_{P \ge 0}\;\; \langle C, P \rangle
  + \varepsilon\, \Omega(P)
  + \lambda_s\, \mathrm{KL}(P\mathbf{1}_m \,\|\, \mu)
  + \lambda_t\, \mathrm{KL}(P^\top \mathbf{1}_n \,\|\, \nu),
\]

where \(\Omega(P) = \sum_{ij} P_{ij}(\log P_{ij} - 1)\) is entropic regularization [@cuturi2013sinkhorn], and \(\mathrm{KL}(u \,\|\, v)\) denotes the Kullback–Leibler divergence between nonnegative mass vectors [@chizat2018unbalanced; @chizat2018scaling]. Because \(P \in \mathbb{R}_{\ge 0}^{n \times m}\), the source marginal is \(P\mathbf{1}_m\) and the destination marginal is \(P^\top \mathbf{1}_n\). The KL penalties relax exact marginal matching, yielding **unmatched mass** on sources and destinations when the visible pool cannot explain full flow activity—appropriate for partial observation and decoy destinations [@peyre2019computational; @villani2009optimal].

**Computation.** We compute approximate solutions via entropy-regularized iterative scaling (Sinkhorn-type updates on the regularized dual). Decoding maps mass in \(P\) to reported directed edges subject to threshold and sparsity rules used in evaluation (§4). Complexity: dense cost/kernel storage is \(O(nm)\); each Sinkhorn scaling iteration is \(O(nm)\), for \(O(Tnm)\) total over \(T\) iterations; the decoder's per-row and per-column ranking pass is also \(O(nm)\). In the tested \(\le 288\times288\) candidate regime (a bounded computational microbenchmark on the frozen benchmark cell; full table in Supplement S.4), solver-plus-decode runtime was approximately 3 s for the UOT pipeline; this is a bounded runtime/memory characterization of that regime and is not extrapolated to corpus-scale candidate spaces.

Component ablations on semi-synthetic stress tests (§4.3) indicate **temporal causality** and **unbalanced mass** are the most sensitive terms among those isolated in our evaluation setup; we report them as sensitivity evidence rather than a complete ranking of all cost components. No isolated risk-term ablation is reported: the identifier "RC" refers to the frozen risk-weighted cost and marginal design (risk enters as a cost weight and a marginal reweighting, not as a hard constraint), and the demonstrated sensitivity ordering names temporal causality and unbalanced mass as the most sensitive isolated terms — the risk term's isolated contribution is not separately quantified.

## 3.4 Conditional Transport-Plan Decoding: Dual-Cancellation Proposition

**Raw-plan decoding problem.** An entropic transport plan factorizes as \(P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)\) with \(K_{ij} = \exp(-C_{ij}/\varepsilon)\). Decoding \(P\) directly—for example, mutual top-*k* selection by the row and column ranks of \(P\)—therefore ranks row \(i\) by \(u_i K_{ij} v_j\): the destination-side dual scaling \(v_j\) enters the row ranking multiplicatively, and symmetrically \(u_i\) enters the column ranking. These scalings are solver artifacts of marginal competition, not pairwise affinity, so raw-plan ranking can drop kernel-strong edges when the solver assigns them small realized mass.

**Directional conditional scores.** Let \(r_i = \sum_j P_{ij}\) and \(c_j = \sum_i P_{ij}\) denote the realized row and column masses. Define the row-direction score \(S^{\mathrm{row}}_{ij} = P_{ij}/c_j\) for \(c_j > 0\) (used for within-row ranking) and the column-direction score \(S^{\mathrm{col}}_{ij} = P_{ij}/r_i\) for \(r_i > 0\) (used for within-column ranking). Cells whose conditioning mass is zero carry no transport support and are ranked last. The decoder emits edge \((i,j)\) if and only if both directional ranks are at most \(k\) (mutual top-\(k\); frozen \(k = 5\), stable index tie-break).

**Proposition 1 (directional dual-cancellation).** Let \(P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)\) with \(u_i > 0\), \(v_j > 0\), and \(K_{ij} > 0\). Under these strict-positivity hypotheses every realized mass \(c_j, r_i\) is positive, and for every cell,

\[
S^{\mathrm{row}}_{ij} \;=\; \frac{u_i K_{ij}}{\sum_l u_l K_{lj}},
\]

which is independent of \(v\). Symmetrically,

\[
S^{\mathrm{col}}_{ij} \;=\; \frac{K_{ij} v_j}{\sum_l K_{il} v_l},
\]

which is independent of \(u\).

*Proof.* \(c_j = \sum_l P_{lj} = \sum_l u_l K_{lj} v_j = v_j \sum_l u_l K_{lj}\), and \(c_j > 0\) implies \(v_j \sum_l u_l K_{lj} > 0\), so \(S^{\mathrm{row}}_{ij} = P_{ij}/c_j = u_i K_{ij} v_j / (v_j \sum_l u_l K_{lj}) = u_i K_{ij} / \sum_l u_l K_{lj}\). The column case is symmetric: \(r_i = u_i \sum_l K_{il} v_l\). ∎

The strict-positivity hypotheses cover dense plans; degenerate plans with zero-mass rows or columns lie outside the identity, and the decoder's "ranked last, no transport support" handling of such cells is an empirical ranking rule, not a consequence of the Proposition.

**What the cancellation means.** Within a fixed row, \(u_i\) is a common positive factor, so the row ranking equals the ranking of \(K_{ij} / D_j\) with \(D_j = \sum_l u_l K_{lj}\): the destination-side scaling \(v_j\) is removed **exactly**, while destination-side source competition \(D_j\)—which carries the source scalings of all competitors—remains active. The column direction is symmetric. The conditional decoder is therefore **dual-cancelled, not kernel-restored**: it does not recover \(K\) exactly, and it retains opposite-side competition. The scores are also gauge-invariant: any positive rescaling of \(v\) leaves \(S^{\mathrm{row}}\) unchanged (and symmetrically for \(u\) and \(S^{\mathrm{col}}\)), whereas \(P\)-based ranking is not invariant under independent rescaling of the ranked-against side.

**Scope of the Proposition.** The identity uses only the factorization, nonnegativity, and division at positive-mass endpoints. It holds for every plan of the factorized form: balanced entropic OT admits \(P = \mathrm{diag}(a)\,K\,\mathrm{diag}(b)\) [@cuturi2013sinkhorn; @peyre2019computational], and entropic unbalanced OT with KL marginal penalties admits the generalized Sinkhorn form \(P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)\) [@frogner2015learning; @chizat2018unbalanced]. The Proposition therefore applies structurally beyond UOT alone and says nothing about which solver family produces a better plan—it neither implies nor predicts a UOT–BOT difference. Interpreting the conditional score as a kernel ranking requires \(K\) to be the solver's kernel \(K = \exp(-C/\varepsilon_{\mathrm{solver}})\) with the decoder's reading of the plan using the same \(\varepsilon\) (both frozen at \(\varepsilon = 0.05\) in our system), and the frozen plans satisfy the factorization to solver tolerance (final errors \(\le 10^{-11}\)), as verified by the per-cell cancellation check. The Proposition establishes the algebraic **channel** of the repair; whether the repair improves correspondence ranking is an empirical question answered by the preregistered confirmatory holdout (§4.3.5). We state the Proposition for completeness of the algebraic justification; it is a modest corollary of standard Sinkhorn structure, and we do not present it as a new normalization or a new theoretical result.

## 3.5 UOT-Q: Event-Backed Quotient-Level Correspondence

Exact **flow-pair attribution** may be **underdetermined by the available on-chain evidence** when multiple destination flows exhibit similar amount–time profiles, when bridge events are shared across candidates, or when event keys are missing. Promoting a single \((s_i, t_j)\) match in such cases risks **over-attribution**.

**UOT-Q** extends UOT from exact flow-pair scoring to **event-backed quotient-level correspondence**. Candidates that share bridge-event incidence structure are grouped into **quotient groups** (event-backed quotient groups) before high-confidence inference. Each group is defined as

\[
q = \bigl([S_q],\, [T_q],\, E_q\bigr),
\]

where \([S_q]\) and \([T_q]\) are multisets of source and destination flows collapsed under a shared bridge-event key (e.g., transfer identifier or event-backed incidence), and \(E_q\) is the qualifying event evidence set. Transport and scoring may be applied within or across quotient boundaries depending on coverage; the intent is to align correspondence hypotheses with **observable bridge episodes** rather than spurious pairwise symmetry.

Quotient grouping does not discard flow-level mass modeling; it constrains **where** hard correspondence claims are admissible. When \(E_q\) is incomplete, pairs remain at diagnostic tier or are withheld (§3.6).

## 3.6 Coverage Qualification and Abstention

**Coverage tiers** qualify whether a candidate pair or quotient group may enter high-confidence output:

| Tier | Role |
|------|------|
| **A / B** | Event-backed with sufficient bridge observability; eligible for high-confidence correspondence if transport and rule scores exceed thresholds |
| **C** | Partial or weak evidence; **diagnostic only**—surfaced for investigator review, not promoted as high-confidence correspondence |
| **Uncovered** | Missing event backing or parser coverage; **abstain** from high-confidence claims |

**Abstention as an AML safety feature.** Withholding a match when evidence is insufficient is preferable to inflating recall with unsupported links [@fatf2023virtualassets]. Uncovered and Tier-C cases may still appear in reports with explicit abstention reasons and suggested follow-up (additional hashes, archive/trace queries, bridge-internal records), but they do not receive the same confidence semantics as Tier A/B event-backed hypotheses.

Scoped evaluation metrics in §4.5 apply only to the **event-backed covered subset**. The interpretation of covered-subset and full-pool metrics is discussed in §5.

## 3.7 Interpretable Correspondence Reports

Each analysis emits a structured **correspondence report** intended for investigator triage. Reports summarize:

- **Correspondence** — proposed source–destination flow or quotient-level link and allocated mass;
- **Confidence** — transport-derived score and tier-qualified decision;
- **Coverage tier** — A, B, C, or Uncovered;
- **Evidence used** — transactions, logs, and bridge events supporting the hypothesis;
- **Missing evidence** — parsers, chains, or event types not observed;
- **Abstention reason** — when the system withholds high-confidence promotion;
- **Investigator action** — suggested next steps (supply destination hashes, widen time window, request trace tier, etc.).

Implementation emits machine-readable and human-readable report forms (JSON and Markdown). The design embeds an explicit **claim boundary**: outputs require human review and do not assert legal proof or full-population precision/recall; detailed deployment and artifact instructions appear in the released tool package.

## 3.8 Prototype Implementation

We provide **Cross AML**, a research prototype that instantiates the pipeline in §3.1 for operational triage. Given investigator-supplied transaction hashes, it (i) collects RPC or dry-run replay evidence, (ii) constructs flows and features, (iii) forms event-backed quotient groups and coverage tiers, (iv) runs coverage-qualified UOT-Q-style matching, and (v) emits interpretable correspondence reports.

The prototype supports **pair scoring** when both source and destination seeds are known and limited **candidate discovery** when only source hashes are supplied; large-scale discovery without indexer, archive, or trace backends remains explicitly out of scope for standard RPC-only deployment.

Cross AML is a **research artifact** for reproducibility and investigator-facing demonstration—not production compliance software. Implementation details, configuration schemas, security notes, and artifact instructions are provided in the released tool package.

# 4. Experiments

We evaluate CSFFC, UOT, UOT-Q, and the Cross AML prototype along five research questions (RQs). Metrics are reported on fixed benchmark splits and held-out test partitions that were not used for model or threshold selection unless noted.

## 4.1 Experimental Setup and Questions

| ID | Research question |
|----|-------------------|
| **RQ1** | Can the Celer signals support a reproducible CSFFC flow benchmark? |
| **RQ2** | Can UOT recover split/merge structures under controlled stress? |
| **RQ3** | Can UOT-Q provide reliable inference on event-backed covered quotient pairs? |
| **RQ4** | Does UOT-Q report higher Precision/F1 than the untuned heuristic and style-adapted representation controls (Table 4; not original-system runs)? The historical calibration budget was asymmetric and favored the proposed pipeline (§4.6). |
| **RQ5** | Can the Cross AML prototype execute the evidence-to-report pipeline? |

**Evaluation settings.** Real Celer supervision supports benchmark construction (RQ1). Semi-synthetic structural stress tests isolate split/merge recovery (RQ2). **Full-anchor admissible decoding** on the frozen transport plan evaluates ranked correspondence and temporal admissibility without re-solving UOT (§4.4). Event-backed **covered quotient** holdout evaluation tests UOT-Q reliability under coverage qualification (RQ3). An **independent held-out test set** (disjoint from development and prior burned evaluation seeds) compares UOT-Q against **heuristic and style-adapted diagnostic baselines** (Table 4; not original-system Connector or ABCTracer runs) on flow-stress Precision, recall, F1, and calibration (RQ4). **Original-system Connector applicability and native diagnostic are reported separately** (Table 6; Appendix B). Prototype validation confirms end-to-end execution without re-running training pipelines (RQ5). An **independent temporal data audit** (§4.9; data-only, no method execution) additionally supports RQ1 by documenting real protocol-grounded flow-level fan-out structure outside the development corpus; it is problem-validity evidence, not performance-validation evidence.

## 4.2 Celer-Supervised Flow Benchmark

**Supervision.** Labels derive from official **Celer cross-chain anchor pairs**: each anchor links one Ethereum transaction hash to one BNB Smart Chain transaction hash. We do not relabel anchors heuristically.

**Flow segmentation (canonical).** Flows aggregate bridge-relevant activity by primary address, chain, asset context (asset group when available, otherwise route identifier), and a rolling **1800 s** window. A new flow starts when the segmentation key changes or the inter-arrival gap exceeds the window.

**Table 1 — Celer-supervised CSFFC benchmark statistics**

| Quantity | Value |
|----------|-------|
| Celer anchor transaction pairs | **7,296** |
| ETH source flows | **5,735** |
| BNB destination flows | **7,122** |
| Supervised flow label rows | **7,128** |
| Sum of support transaction-pair counts | **7,296** |
| Pattern: one-to-one (edge share) | **72.32%** |
| Pattern: one-to-many / fan-out (legacy field `many_to_one`) | **27.51%** |
| Pattern: many-to-one / merge (legacy field `one_to_many`) | **0.17%** |
| Segmentation robustness (600 s / 3600 s vs 1800 s canonical) | Flow-label count change ≤ **±1.1%**; support sum **7,296** |

*Source: frozen paper artifact.*

*Topology direction note: the frozen label layer's legacy pattern names are inverted relative to canonical source→destination topology — legacy `many_to_one` denotes one source flow with multiple destination flows (1→N fan-out), and legacy `one_to_many` denotes multiple source flows with one destination flow (N→1 merge), as verified in code (`flow_label_builder.py` degree rules) and data. Edge identity and all numbers are unchanged; the correction is to the human-readable pattern wording only. See the v2 preregistration package, REAL_TOPOLOGY_TERMINOLOGY_CORRECTION.md.*

Table 1 summarizes benchmark scale, edge-level pattern mix, and segmentation stability. The support transaction-pair sum matches the anchor-pair count (**7,296**), and alternative rolling windows (600 s and 3600 s) perturb flow-label counts by at most **±1.1%** without changing that sum—supporting a reproducible flow-level CSFFC evaluation setting (RQ1).

## 4.3 Structural representation and strict-recovery limits of non-one-to-one bridge flows

We stress-test UOT on **semi-synthetic** episodes with known split, merge, unmatched, and decoy structure cloned from real subgraphs: **48** hierarchical templates × **5** random seeds.

**Table 2 — Edge-inclusion recall under semi-synthetic split/merge stress (frozen reference; NOT exact topology recovery)**

| Metric | Mean | 95% CI |
|--------|------|--------|
| Split edge-inclusion recall | **0.946** | [0.907, 0.985] |
| Merge edge-inclusion recall | **0.967** | [0.913, 1.000] |
| Top-3 destination hit rate per source (recall@3) | 0.481 | [0.418, 0.544] |

*Source: frozen paper artifact. Forty-eight templates × five seeds (42–46). The split/merge metric counts a ground-truth edge as recovered whenever the decoded transport plan assigns it positive mass (≥1e-9); an independent audit (§4.3.2) shows it is edge-inclusion recall over a dense plan, NOT exact topology recovery. The Top-3 row is the macro-averaged fraction of source flows — among those with at least one ground-truth destination and at least one decoded edge with transport mass > 1e-9 — for which at least one ground-truth destination appears among the top-3 destinations ranked by descending decoded transport mass, averaged over seeds 42–46 (rank-based inclusion, not strict recovery; per-source macro; the metric predates the later per-template strict evaluator of Table 2d).*

![Figure 5. Structural capability (edge-inclusion) under semi-synthetic split/merge stress.](figures/ch4/fig5_structural_recovery.png)

*Figure 5: Structural capability under semi-synthetic split/merge stress. Bars show mean edge-inclusion recall with 95% confidence intervals over 48 templates and 5 seeds; the metric is inclusion of ground-truth edges in the dense decoded plan, not exact topology recovery (Table 2d).*

**Table 2b — Edge-inclusion split/merge recall across three bridges (frozen faithful run, 48 templates × 5 seeds, mean ± s.d.)**

| Bridge | Method | Split edge-inclusion recall | Merge edge-inclusion recall |
| --- | --- | ---: | ---: |
| Celer | UOT-Q | 0.950 ± 0.032 | 0.967 ± 0.043 |
| Multichain | UOT-Q | 0.971 ± 0.035 | 0.979 ± 0.026 |
| PolyNetwork | UOT-Q | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Celer / Multichain / PolyNetwork | Connector-style | 0.000 (all seeds) | 0.000 (all seeds) |
| Celer / Multichain / PolyNetwork | ABCTracer-style | 0.000 (all seeds) | 0.000 (all seeds) |

*Source: `out/multi_bridge_expansion/faithful_flow_structural_three_bridges/` (frozen; not re-run in this study). The recovery metric is edge inclusion over the dense decoded plan (positive transport mass), not exact topology recovery; the strict evaluation of the same operating point is Table 2d. Cross-bridge amount caveats apply: Multi/Poly amounts use a current external price snapshot (not transaction-time prices) and PolyNetwork inherits its lock==unlock amount structure (no fee asymmetry) — see §4.3.4 Limitations.*

### 4.3.1 Representation capability

Before any fitting, the five methods compared here differ in what their decoded output data structures can represent. Four properties are audited at the code level and verified with minimal constructed tests (TEST A–F, `out/multi_bridge_expansion/structural_baseline_mechanism_study/capability_tests/`):

- **Connector-style / ABCTracer-style** decoders enforce source out-degree ≤ 1 (`WithdrawLocator.search_withdraw`: per-source `groupby(txhash).idxmin()` + `drop_duplicates`; `predict_abctracer_style`: per-source argmax). A 1→2 split is therefore structurally unrepresentable; TEST B verifies that only one of the two split edges can be emitted. Target-side uniqueness is **not** enforced, so a 2→1 merge *is* representable in their output data structure (TEST C), although pairwise amount/time selection scores zero merge edges on this benchmark.
- **Threshold-MM** keeps every edge whose calibrated cost is below a threshold, decided independently per cell; 0/1/many edges per source and empty rows are admissible, and its decisions are invariant to other nodes' mass (TEST F).
- **Balanced-OT** solves the joint entropic plan with strictly balanced marginals and no dustbin/dummy node; it emits both edges of split and merge (TEST B/C) but cannot abstain — the unmatched source's mass is transported and the unmatched target receives mass (TEST D/E).
- **UOT-Q** relaxes the marginals (unbalanced KL, `allow_unmatched`): mass destruction/creation is the first-class abstention signal (frozen Celer seed-42 plan transports 0.782 of 1.0 unit mass).

**Table 2c — Method capability table (code-level audit + constructed unit tests)**

| Method | Split (1→2) | Merge (2→1) | Global allocation | Unmatched |
| --- | --- | --- | --- | --- |
| Connector-style | NO | PARTIAL | NO | PARTIAL |
| ABCTracer-style | NO | PARTIAL | NO | PARTIAL |
| Threshold-MM | YES | YES | NO | YES |
| Balanced-OT | YES | YES | YES | NO |
| UOT-Q | YES | YES | YES | YES |

*"Unmatched = PARTIAL" for the one-to-one methods means rejection-level abstention only (all filter stages reject, or the best score is non-positive), never a first-class per-node decision, and an unmatched destination is never represented (`capability_tests/unmatched_semantics.md`). UOT-Q's "YES" is a mass-level property; at the literal 1e-9 decode threshold the decoded edge set can still contain a low-mass edge for the unmatched node. For the one-to-one methods an unmatched destination is never represented as a first-class decision.*

![Figure 5c. Method capability ladder.](figures/ch4/fig5c_capability_ladder.png)

*Figure 5c: Method capability ladder from the constructed unit tests (TEST A–F).*

### 4.3.2 Three-bridge structural representation and strict evaluation

**Protocol.** For each bridge we build flow-segment exports from on-chain data (USD-normalized amounts preserving real source/destination asymmetry; real event timestamps; a rule-derived risk proxy; a per-flow evidence proxy; multi-address sets; provenance in `feature_provenance.md`), clone **48** real one-to-one templates per seed into split (1→2), merge (2→1), unmatched, and decoy structures (decoy timestamps perturbed +60 s / +120 s), and evaluate **five methods on the identical grids** (seeds 42–46): the frozen UOT-Q plan decoded at 1e-9; Balanced-OT on the same cost matrix and marginals with strictly balanced constraints (unit-normalized totals, reg=0.05) and the same 1e-9 mass-threshold decode; Threshold-MM on the same cost matrix with one **global** threshold calibrated on disjoint seeds (101–103) by edge F1 (τ* = 0.05 quantile of the calibration cost ECDF, cutoff 0.478; never re-tuned on the test seeds); and the project's existing Connector-style / ABCTracer-style per-source top-1 rules on the shared cost decomposition, scoped per template as in the existing structural-baseline implementation. All methods share one unified evaluator that reports strict exact recovery (the predicted edge set of a template must equal its structural ground truth exactly; one extra edge forfeits the template), edge precision/recall/F1, degree accuracy, and false-positive counts.

**Table 2d — Strict structural evaluation, three bridges × five methods (48 templates × 5 seeds; per-seed = mean over templates; mean ± s.d. over seeds)**

| Bridge | Method | Split exact | Merge exact | Edge P | Edge R | Edge F1 | Degree acc | FP edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Celer | Connector-style | 0.000 | 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 | 5.84 ± 0.04 |
| Celer | ABCTracer-style | 0.000 | 0.000 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.026 ± 0.007 | 0.000 | 5.84 ± 0.04 |
| Celer | Threshold-MM | 0.000 | 0.000 | 0.094 ± 0.009 | 0.971 ± 0.006 | 0.168 ± 0.016 | 0.004 ± 0.064 | 67.7 ± 9.7 |
| Celer | Balanced-OT | 0.000 | 0.000 | 0.016 ± 0.002 | 0.968 ± 0.036 | 0.030 ± 0.005 | 0.002 ± 0.046 | 680.8 ± 128.9 |
| Celer | UOT-Q (frozen) | 0.000 | 0.000 | 0.022 ± 0.005 | 0.961 ± 0.035 | 0.040 ± 0.009 | 0.002 ± 0.046 | 539.0 ± 135.3 |
| Multichain | Connector-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| Multichain | ABCTracer-style | 0.000 | 0.000 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.034 ± 0.006 | 0.000 | 5.80 ± 0.04 |
| Multichain | Threshold-MM | 0.000 | 0.000 | 0.075 ± 0.005 | 0.844 ± 0.030 | 0.136 ± 0.009 | 0.000 | 71.0 ± 2.9 |
| Multichain | Balanced-OT | 0.000 | 0.000 | 0.010 ± 0.003 | 0.994 ± 0.008 | 0.018 ± 0.005 | 0.000 | 1069.6 ± 142.4 |
| Multichain | UOT-Q (frozen) | 0.000 | 0.000 | 0.015 ± 0.004 | 0.983 ± 0.020 | 0.027 ± 0.008 | 0.000 | 925.4 ± 140.7 |
| PolyNetwork | Connector-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| PolyNetwork | ABCTracer-style | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.00 |
| PolyNetwork | Threshold-MM | 0.000 | 0.000 | 0.065 ± 0.006 | 0.997 ± 0.006 | 0.120 ± 0.010 | 0.000 | 107.5 ± 8.1 |
| PolyNetwork | Balanced-OT | 0.000 | 0.000 | 0.013 ± 0.001 | 1.000 | 0.026 ± 0.002 | 0.000 | 852.8 ± 55.4 |
| PolyNetwork | UOT-Q (frozen) | 0.000 | 0.000 | 0.018 ± 0.004 | 1.000 | 0.034 ± 0.007 | 0.000 | 743.6 ± 63.4 |

*Source: `out/multi_bridge_expansion/structural_baseline_mechanism_study/aggregated/` (independent verification recomputes every cell from the raw per-template artifacts). The frozen UOT-Q split/merge "recovery" of Table 2/2b (0.946–1.000) is edge-inclusion recall; the strict exact column here is 0.000 for every method on every bridge.*

**Three audited causes of the all-zero exact column** (not evaluator artifacts — see `diagnostics/diagnostics.md`): (i) the frozen decode threshold (1e-9) is a "positive mass" rule and both entropic plans are dense — the frozen UOT-Q plan emits 18,776 edges on the Celer seed-42 grid and the balanced plan ~30,000, so strict equality with a 6-edge truth is unattainable; (ii) the true structural edges are **not** the locally cheapest cells — their per-source cost rank is ≈3–4 (Top-k-MM edge F1 rises from 0.019 at Top-2 to 0.248 at Top-3), because the split source's cheapest target is the full-amount merge destination and the merge sources' cheapest targets are the half-amount split destinations; any local threshold that keeps the truth edges also keeps cheaper confusers, so exact recovery is unattainable for local rules, and the transport methods include the truth edges only through global mass allocation; (iii) the legacy metric counts inclusion, not exactness.

![Figure 5d. Three-bridge strict structural comparison.](figures/ch4/fig5d_three_bridge_structural.png)

*Figure 5d: Strict structural evaluation on three bridges (48 templates × 5 seeds, mean ± 95% CI). Top: exact split recovery — 0.000 for all five methods (decode-density effect; see text). Bottom: per-template edge precision, recall, and F1. The calibrated threshold heuristic attains the highest edge F1; UOT-Q sits above strictly balanced OT on every bridge, and both transport decodes are far denser than the pairwise threshold rule.*

**Interpretation.** The capability difference is real and bridge-invariant: only the many-match methods can emit a 1→2 edge set at all (verified by unit tests and by the existence of decoded multi-destination sources for Threshold-MM, Balanced-OT, and UOT-Q on every bridge/seed, while Connector/ABCTracer never emit one). But at the frozen operating point no method achieves exact topology recovery, and on edge-level metrics the globally calibrated per-cell threshold rule (F1 0.120–0.168) dominates both transport methods (F1 ≤ 0.040). UOT-Q's unbalanced relaxation keeps it above strictly balanced OT on every bridge (precision 0.015–0.022 vs 0.010–0.016), but the frozen dense decode means its edge precision remains low. The earlier claim that "UOT-Q recovers the non-one-to-one structure while one-to-one baselines remain at zero" holds only for the edge-inclusion metric; under the strict evaluator all methods score zero and the ranking on edge metrics is Threshold-MM > UOT-Q > Balanced-OT > one-to-one baselines.

### 4.3.3 Why unbalanced transport?

To isolate the mechanism behind the unbalanced relaxation, we run four seed-controlled stress ladders (seeds 101–103, three bridges, all methods on the identical cost matrices, no re-tuning): **mass mismatch** (destination amounts × (1+m), m ∈ {0, 0.05, 0.10, 0.20, 0.40}); **unmatched ratio** (extra full-mass sources without ground-truth edges, realized ratios ≈ {0, 0.10, 0.20, 0.30, 0.40}); **decoy density** (k ∈ {1, 2, 4, 8} decoy pairs per template); and **timestamp noise** (true-leg offsets ~U(0, 60·(s−1)) s, s ∈ {1, 2, 4}).

**Clean conditions.** At 0% mismatch, Balanced-OT is not close to UOT-Q (the H3 prediction): edge F1 is 0.070 vs 0.110 (Celer), 0.032 vs 0.047 (Multichain), 0.033 vs 0.045 (PolyNetwork). The unbalanced solver's ability to shed mass on confusers already matters at the balanced operating point.

**Mass mismatch.** Both transport methods degrade as destination mass inflates (Celer precision: Balanced-OT 0.043→0.024; UOT-Q 0.068→0.037; F1 0.070→0.044 vs 0.110→0.064), while Threshold-MM is flat or improves (0.142→0.155) because inflating destination amounts amplifies the pairwise amount signal that the local rule exploits. UOT-Q remains above Balanced-OT at every level, but the *relative* degradation is comparable; the mechanism advantage of unbalanced mass relaxation is a level offset, not a slope difference, in this regime.

**Unmatched ratio.** Precision declines for all three methods (forced transport into hidden targets for Balanced-OT; additional pairwise edges for Threshold-MM; Celer: 0.142→0.086 / 0.043→0.029 / 0.068→0.035 at 40% unmatched). UOT-Q keeps its edge over Balanced-OT at every level while destroying the excess mass (its transported mass falls with the unmatched ratio, the explicit abstention signal).

### 4.3.4 Robustness to unmatched mass and decoys

**Decoy density.** Threshold-MM's false-positive count grows fastest, as its pairwise independence predicts: Celer FP edges per template 26.3 (k=1) → 223.6 (k=8), a 750% increase, with precision falling 0.182 → 0.061; the joint methods grow more slowly (Balanced-OT 266→1006, +278%; UOT-Q 211→714, +239%). The global mass competition of the transport methods does damp FP growth, but their decodes are dense by construction, so their absolute FP counts stay above the threshold rule at every density.

**Timestamp noise.** All three methods are essentially flat across 1–4× noise (Celer F1: 0.242→0.241 threshold; 0.070→0.065 balanced; 0.110→0.101 UOT-Q): the tested perturbations (≤180 s) barely move costs against the 21,600 s window at time weight 0.25. This ladder does not discriminate the methods at the tested scales.

![Figure 5e. Mechanism stress.](figures/ch4/fig5e_mechanism_stress.png)

*Figure 5e: Mechanism stress ladders pooled over Celer / Multichain / PolyNetwork (mean ± 95% CI). Exact recovery is 0.000 at every level for every method (not shown; see text). Top-left: mass mismatch — Threshold-MM flat/improving, both transport methods degrade, UOT-Q above Balanced-OT at every level. Top-right: unmatched ratio — all methods lose precision, UOT-Q retains its level offset. Bottom-left: decoy density — Threshold-MM precision falls fastest (pairwise FP growth). Bottom-right: timestamp noise — no method discriminates at the tested scales.*

**Limitations.** The structural test remains semi-synthetic (real flows supply the feature anchors; the 1→2 / 2→1 patterns are imposed because the published Validation labels are transaction-pairwise one-to-one anchors and supply no non-one-to-one flow ground truth; §4.9 reports a later protocol-native data-only audit establishing that real flow-level fan-out structure does occur outside the development corpus). The frozen decode threshold (1e-9) is a mass-positivity rule that yields dense plans on these 288-flow grids; the strict exact-recovery metric is therefore zero for all methods, and the mechanism conclusions above rest on edge-level precision/recall/F1 and FP counts (pre-registered in `PRE_REGISTERED_HYPOTHESES.md`). The calibrated global τ sits at the lower boundary of the pre-registered grid (edge F1 is monotone decreasing in τ on the calibration set); per-bridge τ values are reported as supplementary. PolyNetwork inherits its lock==unlock amount structure (no fee asymmetry in decoded amounts), and Multi/Poly amounts use a current external price snapshot, not transaction-time prices.

**Summary of the mechanism evidence.** The data support the weakest of the three pre-registered conclusion levels: **UOT-Q removes the representational limitation of one-to-one hard matching** — it can emit 1→2 and 2→1 edge sets, abstain via mass destruction, and allocate globally, which Connector-style and ABCTracer-style structurally cannot (split) or empirically do not (merge). They do not support the stronger claims: the globally calibrated per-cell threshold rule dominates the transport methods on edge F1 (so "global transport improves structural consistency over independent pairwise expansion" is not established in this benchmark), and while UOT-Q beats strictly balanced OT at every stress level, exact recovery is zero for both, so "unbalanced mass relaxation provides robustness" is supported only as a consistent level offset on edge metrics, not as exact-topology robustness.

### 4.3.5 Confirmatory holdout: conditional-plan decoding repairs raw-plan correspondence ranking

**The raw-plan ranking problem.** The frozen UOT-Q plan factorizes as P = diag(u)·K·diag(v) with K = exp(−C/ε), so direct decoding of P mixes the kernel ranking with the solver's dual scalings. On the development seeds (201–205), amount-free cost decoding alone attained macro edge F1 0.3172 while the raw-plan mutual-top-5 decoder (RAW_UOT_PLAN_D4) attained only 0.2374, and the kernel-level diagnosis traced the degradation to marginal competition and destination-side dual scaling: cost-retained ground-truth edges were dropped by raw decoding at high rates, and a uniform-marginal intervention recovered the cost ceiling. We therefore pre-registered a **dual-cancelled conditional-plan decoder** — the row-direction score S_row = P_ij / c_j (column-mass-normalized) and the column-direction score S_col = P_ij / r_i (row-mass-normalized), decoded with the same mutual-top-5 rule (k = 5) — together with a fixed five-gate decision tree, before any holdout data existed. The dual-cancellation Proposition (§3.4) provides the algebraic justification: S_row cancels the destination-side dual scaling exactly and S_col cancels the source-side dual scaling exactly, while retaining opposite-side competition.

**Frozen confirmatory test.** All components — the amount-free renormalized cost, the risk/evidence-weighted marginals, reg = 0.05, reg_m = 0.5, the decoder, the paired bootstrap (B = 4,000, `RandomState(20240101)`), and the decision rules — were hash-locked before the untouched holdout seeds 301–305 (Celer, Multichain, PolyNetwork; 48 templates per seed; 720 paired template instances) were generated or read. The one-shot runner verifies every hash and refuses to execute otherwise, and an independent verification pass (a separate code path that recomputes all results from the raw per-cell artifacts rather than trusting the runner's outputs) matched the runner within 1e-9.

**Primary result.** RAW_UOT_PLAN_D4 attains macro edge F1 0.2350 and the conditional decoder attains 0.3104; **Δ_primary = +0.075413 with a paired 95% CI [0.070557, 0.080312]**, entirely above zero (gate A; Table 2e; Figure 5f). The frozen bootstrap resamples paired template-level contrasts within each bridge with replacement and takes the replicate statistic as the unweighted mean over the three bridges' per-bridge paired means (B = 4,000, `RandomState(20240101)`; per-template metrics are aggregated to per-seed means over 48 templates, then per-bridge means over 5 seeds, then the macro over 3 bridges; edge-level pooling across templates is never used for the CI). The seed level enters the point estimate (per-template metrics → per-seed means → per-bridge means → macro) but not the resampling, which is a within-bridge paired template-instance resample; the CI is therefore a paired within-bridge resampling CI, not a seed-cluster bootstrap. The per-bridge paired s.d. over seed-means are Celer 0.0704, Multichain 0.0881, PolyNetwork 0.0247; the primary effect's sign is robust to seed-level variance inflation (all three per-bridge CIs exclude zero).

**Three-bridge consistency.** The improvement is positive on every bridge — Celer +0.0826 [0.0743, 0.0918], Multichain +0.0820 [0.0709, 0.0933], PolyNetwork +0.0616 [0.0587, 0.0649] — so the primary effect is not driven by a single bridge (gate D). PolyNetwork's comparatively narrow CI is consistent with its disclosed lock==unlock amount structure (no fee asymmetry), which makes PolyNetwork templates near-deterministic; the three bridges are not treated as exchangeable variance sources.

**Mechanism gate.** The gain is accompanied by the exact preregistered directional signature expected if conditional normalization repairs ranking damage associated with dual scaling and marginal pressure in the raw plan: harmful flip rates decrease by −0.2595 (row), −0.2303 (column), −0.5360 (split-child), and −0.5095 (merge-dst), and GT mutual-top5 retention increases by +0.2035 (gate B). A **harmful flip** is a cost-retained ground-truth edge (inside the kernel's mutual top-5) that a decoder drops, measured per direction (row, column) and separately for split-child and merge-destination edges; **GT mutual-top5 retention** is the fraction of ground-truth edges the decoder keeps in its mutual top-5. These observational mechanism metrics are **consistent with the preregistered ranking-repair mechanism** and support it, but they do not alone establish a complete causal decomposition of the OT solver.

**Recovery.** With D = F1(AMOUNT_FREE_COST_D4) − F1(RAW_UOT_PLAN_D4) = +0.070569, the preregistered recovery metric gives **recovery_fraction = 1.0686** (a point estimate without a CI; a value above 1 means the conditional decoder exceeded the amount-free cost reference, which was preregistered as a reference, not a mathematical ceiling), satisfying gate C.

**Transport-plan contribution.** Under the frozen macro rule, conditional decoding of the transport plan exceeds the support-plus-kernel decoder (Δ_support = +0.0104 [0.0062, 0.0152]) and the amount-free cost decoder (Δ_cost = +0.0048 [0.0013, 0.0089]); the preregistered holdout therefore provides **evidence for a positive transport-plan-level contribution beyond direct kernel/support decoding at the macro level** (TYPE A). This is not a claim about the unbalanced relaxation: CONDITIONAL_BOT_D4 (0.3106) and CONDITIONAL_UOT_D4 (0.3104) differ by Δ_bot ≈ −0.0003 (unrounded macro mean −0.0002688, 95% CI [−0.0018, +0.0012], includes zero; the 4-dp Table 2e F1 values imply −0.0002 through rounding), so **the confirmatory experiment does not establish an additional benefit attributable specifically to unbalanced relaxation**. The decoder-repair effect replicated on the preregistered holdout, while the transport-value classification strengthened from TYPE B during development to TYPE A under the frozen holdout rule; the criteria were fixed before seeds 301–305 were accessed, and development and holdout data were not pooled.

**Heterogeneity and integrity.** The primary repair effect is positive with per-bridge 95% CIs that exclude zero on all three bridges, but the transport-value secondary contrasts are heterogeneous: PolyNetwork's secondary contrasts are negative (Δ_cost −0.0043 [−0.0065, −0.0024]; Δ_support −0.0043 [−0.0065, −0.0024]), so **TYPE A is a three-bridge macro classification, not a bridge-uniform effect**. Integrity checks of the one-shot execution: FP/template ratio 1.0511 and FN/template ratio 0.4945 (conditional/raw, both within the preregistered 3× bound), UOT max final error 9.9e-12, BOT max residual 4.4e-12, zero convergence failures, zero missing or duplicate templates, and zero NaNs in any evaluation field.

**Limitations.** The confirmatory endpoint is template-level edge F1 on semi-synthetic split/merge structure (strict exact-topology recovery remains zero for all methods on this benchmark family; Table 2d); the decoder is fixed at k = 5; and the mechanism indicators are observational diagnostics, not a complete solver-level causal decomposition.

**Table 2e — Preregistered confirmatory holdout (seeds 301–305), five frozen decoding methods**

| Method | Edge precision (macro) | Edge recall (macro) | Edge F1 (macro) |
| --- | ---: | ---: | ---: |
| **RAW_UOT_PLAN_D4** | 0.1476 | 0.5975 | 0.2350 |
| **CONDITIONAL_UOT_D4** | 0.1926 | 0.8009 | 0.3104 |
| AMOUNT_FREE_COST_D4 | 0.1896 | 0.7882 | 0.3055 |
| CONDITIONAL_BOT_D4 | 0.1927 | 0.8016 | 0.3106 |
| SUPPORT_PLUS_K_D4 | 0.1862 | 0.7734 | 0.2999 |

*Table 2e: Macro edge precision/recall/F1 on the untouched confirmatory holdout (seeds 301–305; Celer, Multichain, PolyNetwork; 48 templates per seed; identical grids across methods). The primary confirmatory contrast was preregistered as CONDITIONAL_UOT_D4 − RAW_UOT_PLAN_D4 (both rows in bold; no five-way winner is implied). Primary paired effect Δ_primary = +0.075413, paired 95% CI [0.070557, 0.080312] (bridge-stratified template-instance bootstrap, B = 4,000, RandomState(20240101); the point estimate is aggregated template → seed → bridge → macro). Sensitivity intervals recomputed from frozen per-instance results without any method rerun: two-stage seed → template bootstrap [0.0677, 0.0831], amount × delay stratification (Q|D) bootstrap [0.0676, 0.0849], and base-anchor cluster bootstrap (true dependence unit) [0.0708, 0.0803]; all exclude zero (Supplement S.2). CONDITIONAL_BOT_D4 vs CONDITIONAL_UOT_D4: −0.0003, CI includes 0. Source: `out/multi_bridge_expansion/conditional_plan_holdout_results/` (frozen one-shot execution; independent verification within 1e-9).*

![Figure 5f. Preregistered confirmatory holdout: conditional-plan decoding repairs raw-plan correspondence ranking.](figures/ch4/fig5f_conditional_plan_confirmatory.png)

*Figure 5f: Preregistered confirmatory holdout (seeds 301–305; Celer, Multichain, PolyNetwork; 48 templates per seed; 720 paired template instances). (a) Macro edge F1 of the five frozen decoding methods; the primary comparison is CONDITIONAL_UOT_D4 versus RAW_UOT_PLAN_D4 (Δ_primary = +0.0754 [0.0706, 0.0803]). (b) Paired Δ_primary per bridge and macro with 95% CIs (paired template-instance bootstrap, B = 4,000, RandomState(20240101)); all positive. (c) The five preregistered mechanism-direction changes (conditional minus raw): all four harmful flip rates decrease and GT mutual-top5 retention increases (gate B); all five changes point in the preregistered repair direction — negative values indicate reductions for the four harmful-flip metrics, whereas the positive value indicates increased GT mutual-top5 retention. CONDITIONAL_BOT_D4 and CONDITIONAL_UOT_D4 differ by −0.0003 with a CI including 0; the figure does not imply any UOT-over-BOT advantage.*

## 4.4 Fixed-Delay UOT-Q on Full Anchor Pairs

The final model uses **`tx_if_available_else_flow_representative`** delay policy, replacing the legacy flow-boundary delay. On all **7,296** Celer anchor pairs, UOT is evaluated as a **ranked flow-correspondence model**—not a raw tx-pair oracle. The frozen transport plan is **not re-solved**; admissibility strategies apply **decode-only** temporal masks and optional top-*k* rescue on the exported ranking.

The raw tx-level argmax projection reaches pair F1 **0.589** and top-3 recall **0.658**, but is reported only as a **compatibility projection**, not as a forensic admissibility claim (tx-level CVR **0.338**). The formal **UOT-Q high-coverage decoding** is **positive_delay_top3_rescue**: pair F1 **0.590**, top-3 recall **0.658**, tx-level CVR **0.005**, coverage **0.996**. The formal **forensic high-confidence subset** is **joint_time_admissible_filter**: precision **0.889**, recall **0.589**, pair F1 **0.708**, tx-level CVR **0**, coverage **0.845**. A **permuted-label control** collapses pair precision/recall/F1 to near zero, confirming label-structured recovery.

**Coverage** is the fraction of source flows with at least one non-abstained tx projection (denominator: source flows in the transport plan). **Abstention rate** is the fraction of labeled anchor pairs for which decoding abstains (denominator: **7,296**). These denominators differ by design.

Pair F1 ≈ 0.321 figures from the legacy delay / pre-admissible pipeline appear for **diagnostic comparison only** and are not headline claims.

**Table 5 — Fixed-delay UOT-Q main results (full anchor pairs; decode-only; transport frozen)**

| decoding | pair_precision | pair_recall | pair_f1 | top3_recall | tx_level_cvr | coverage | abstention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw tx projection (compatibility) | 0.589 | 0.589 | 0.589 | 0.658 | 0.338 | 1.000 | 0.000 |
| UOT-Q top-3 admissible rescue (high coverage) | 0.590 | 0.589 | 0.590 | 0.658 | 0.005 | 0.996 | 0.002 |
| Joint time-admissible filter (high-confidence forensic) | 0.889 | 0.589 | 0.708 | 0.658 | 0.000 | 0.845 | 0.338 |
| Permuted-label control | 0.000 | 0.000 | 0.000 | 0.002 | — | — | — |

*Source: frozen fixed-delay paper artifact. Top-3 recall reflects transport ranking quality and is unchanged by admissibility filters. Coverage and abstention denominators differ by design (coverage: source flows in the transport plan with at least one non-abstained tx projection; abstention: all 7,296 labeled anchor pairs; see §4.4). For the permuted-label control, tx CVR, coverage, and abstention are not interpretable as method performance; cells are shown as —. The permuted-label control is a single fixed permutation reported as a deterministic sanity control, not a distribution over permutations.*

### 4.4.1 Leakage audit (fixed-delay pipeline)

We audit the **fixed-delay + admissible decoding** pipeline with leave-key-out and leave-anchor-out-strict masking (Appendix Table A.1). Under strict masking, **forbidden_features_remaining = 0** and fake-oracle anchor injection is fully neutralized (fake-anchor probe invariant). The invariance of outputs across masking conditions reflects that the audited features are not consumed by the model (forbidden_features_remaining = 0); the audit therefore functions as a **feature-consumption check** rather than a stress test of leakage sensitivity. **The fixed-delay UOT-Q gains are not explained by bridge-key or bridge-evidence leakage.** Baseline, leave-key-out, and strict agree within tolerance **0.02** on pair F1, recall, and top-3 recall for raw projection, top-3 rescue, and joint filter decodings. This audit supports the new fixed-delay headline; the legacy leave-anchor-out result (pair F1 ≈ 0.321) is **not** used to substantiate current claims.

### 4.4.2 Symmetric masking degradation and baseline applicability

To compare **original-system** behavior beyond a single operating point, we evaluate Connector (original `WithdrawLocator` raw top-1) and UOT-Q (`joint_time_admissible_filter` decoding) under a **symmetric masking ladder** on the shared **7,296** tx-pair ground truth and BNB candidate universe (`candidate_bnb_universe_all_txs.csv`). **Table 6 is a degradation and applicability curve, not a simple leaderboard.** It reports how each method responds as bridge-matching evidence is progressively removed.

**Under full native bridge semantics and the closed-set candidate pool**, Connector achieves substantially stronger raw tx-pair matching (F1 = **0.9736**) than UOT-Q (joint F1 = **0.7085**). **This Connector score is an upper-bound native-feature diagnostic**, not an open-world retrieval result: the candidate pool contains exactly the **7,296** ground-truth BNB destination hashes. **Removing ID-anchor/event-like fields does not affect Connector** because those fields are not consumed by `WithdrawLocator`'s core matching columns; the `id_anchor_masked` row reproduces the full-native Connector outcome. **Removing receiver or amount semantics blocks or collapses Connector**: receiver clearing yields `BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS` (N/A, not F1 = 0); zeroing amount yields `ZERO_PREDICTIONS_AFTER_AMOUNT_MASK` (F1 = **0.0000** with zero predictions). **UOT-Q degrades under stronger masking but remains evaluable** at all ladder levels and preserves **tx-CVR = 0** under the joint admissible filter.

**The contribution is not that UOT-Q dominates Connector in native bridge-semantics mode.** UOT-Q provides a **ranked flow-correspondence model** and **admissible abstaining decoder** that remains applicable when bridge-specific matching semantics are absent or partially unavailable. Results are from the Route A v2 symmetric masking run (`out/baseline_compare/routeA_symmetric_masking_v2/`); Appendix B provides audit traceability. **ABCTracer remains blocked** because no official checkpoint was available.

**Table 6 — Symmetric masking degradation and baseline applicability**

| mask_level | masked fields / retained evidence | Connector raw top-1 F1 | Connector status | UOT-Q joint F1 | UOT-Q tx-CVR | UOT-Q coverage | interpretation |
|------------|-----------------------------------|------------------------:|------------------|------------------:|----------------:|------------------:|----------------|
| full_native | none masked; receiver, amount, asset, chains, timestamp retained | 0.9736 | ACCEPTED | 0.7085 | 0 | 0.8453 | Closed-set upper-bound native diagnostic; UOT-Q ranked flow correspondence |
| id_anchor_masked | event, bridge, sender masked; WL matching fields retained | 0.9736 | ACCEPTED | 0.7086 | 0 | 0.8453 | Non-WL ID fields masked; Connector unchanged; UOT-Q re-solved |
| no_receiver | receiver cleared; amount, asset, chains, timestamp retained | N/A | BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS | 0.7001 | 0 | 0.8407 | Connector blocked (receiver required); UOT-Q evaluable |
| no_amount | amount zeroed; receiver, asset, chains, timestamp retained | 0.0000 | ZERO_PREDICTIONS_AFTER_AMOUNT_MASK | 0.3714 | 0 | 0.7320 | Connector zero predictions after amount mask; UOT-Q degrades |
| no_receiver_no_amount | receiver cleared and amount zeroed; asset, chains, timestamp retained | N/A | BLOCKED_BY_REQUIRED_RECEIVER_SEMANTICS | 0.3752 | 0 | 0.8300 | Connector blocked; UOT-Q evaluable at reduced F1 |

*Caption: Symmetric masking degradation and baseline applicability on **7,296** Celer anchor tx pairs. Connector uses original `WithdrawLocator` raw top-1 only; **Connector top-3 and UOT-Q joint parity are unavailable for Connector** because original `WithdrawLocator` does not expose a ranked top-*k* candidate list. **The Connector candidate pool is closed-set** (**7,296** candidate BNB tx hashes equal the GT destination hash set); therefore the full_native Connector score is an **upper-bound native-feature diagnostic**, not an open-world retrieval result. **ABCTracer remains blocked** because no official checkpoint was available. **This table is a degradation/applicability curve, not a simple leaderboard.** Blocked Connector rows are applicability outcomes (N/A), not F1 = 0; the no_amount row reports zero predictions after amount masking. Audit traceability: Appendix B. Source: `out/baseline_compare/routeA_symmetric_masking_v2/degradation_curve_v2.json`. **Table 6 does not replace Table 5.***

## 4.5 Coverage-Qualified Quotient Inference

**UOT-Q** applies event-backed quotient grouping and coverage tiers before promoting correspondence hypotheses. Holdout evaluation uses a development-frozen quotient rule keyed on bridge-event incidence.

**Coverage-qualified quotient check.** The development-frozen bridge-transfer-key rule reproduces the transferId-consistent quotient grouping on the 122 covered holdout pairs. Because the covered-pair label and the rule share the same bridge-event transferId key, this is a definitional consistency check on the quotient machinery, not an independent predictive validation. The full table is reported in Supplement S.1.

![Figure 6. Coverage-qualified inference scope.](figures/ch4/fig6_coverage_scope.png)

*Figure 6: Coverage-qualified inference scope. Event-backed projection coverage is approximately 0.792; uncovered edges are abstained. Covered P/R/F1 is reported only on the 122 covered quotient holdout pairs.*

On the **122** covered holdout pairs (44 positive / 78 negative; holdout seeds 212–231), the development-frozen bridge-transfer-key rule reproduces the transferId-consistent quotient grouping exactly (Supplement S.1; Figure 6). **Interpretation discipline:** both the covered-pair label and the rule are keyed on the same bridge-event transferId incidence, so this result is a **definitional consistency check on the event-backed quotient projection** — it verifies that the frozen rule and the covered-scope grouping agree — and it carries **no independent correspondence-accuracy interpretation**; it is not evidence that the method recovers cross-chain correspondence beyond what the bridge-event key already encodes. Event-backed projection coverage is approximately **0.792** (computed on the gate-reference seeds 52–57; the full-scope claim gate fails on this value), so metrics on the 122 covered pairs must not be read as full-population recovery. Scope boundaries are discussed in Section 5.

## 4.6 Comparison with Cross-Chain Tracing Baselines

We compare UOT-Q (precision-oriented reranking configuration, frozen at development time) against **style-adapted representation-capability controls**—heuristic adaptations to our flow labels, **not original Connector or ABCTracer system runs**—on an **independent held-out test set** (the frozen pipeline's development-sealed flow-stress holdout, not used for model selection). **Calibration asymmetry disclosed.** UOT-Q's Table 4 operating point received a development tuning budget: a logistic-regression precision reranker (balanced class weights) trained on development seeds 52–71 using ground-truth pair labels, a decision threshold selected on the same development labels over a fixed grid (frozen τ = 0.7796), and decoder selection on development data—all frozen before the holdout (292–311) was touched. The two adapted baselines received **no calibration budget**: their weights are fixed heuristic constants from the frozen config catalog (`calibration="none"`, threshold 0.0, top-k = 50), with no development-selection step. The comparison holds the candidate pool (K = 50) and the evaluator fixed across methods, but the tuning budget is **asymmetric and favors the proposed method**; Table 4 is therefore reported as a fixed-scope diagnostic with untuned heuristic controls, not as evidence of calibrated superiority over tuned baselines. **Original-system comparison is reported in Table 6 and Appendix B.**

**Table 4 — Flow-stress comparison on the independent held-out test set**

| Method | Precision | Recall | F1 | ECE |
|--------|-----------|--------|-----|-----|
| **UOT-Q** | **0.192** | 0.900 | **0.316** | 0.028 |
| Heuristic adapted baseline (Connector-inspired) | 0.113 | 0.718 | 0.195 | 0.060 |
| Style-adapted diagnostic baseline (ABCTracer-inspired) | 0.100 | 0.975 | 0.181 | 0.018 |

*Source: frozen paper artifact. **These rows are not original-system runs** — they serve as untuned representation-capability controls — and are superseded for original-baseline comparison by Route A v2 (Table 6) and the Connector native audit (Appendix B). They must not be used to support claims about original Connector or ABCTracer performance (e.g., native Connector F1 = 0.9736).*

![Figure 7. Flow-stress comparison on the independent held-out test set.](figures/ch4/fig7_baseline_grouped_metrics.png)

*Figure 7: Flow-stress comparison on the independent held-out test set. UOT-Q reports higher Precision and F1 than the untuned style-adapted representation controls on this held-out partition; the recall-oriented style-adapted control retains higher Recall. Original-system applicability is reported separately (Table 6; Appendix B).*

**UOT-Q reports higher Precision and F1 than both untuned style-adapted representation controls** on this held-out partition (Table 4; Figure 7); these are point estimates on a single partition with no attached CI, and the comparison is exploratory (RQ4). **The style-adapted representation control (ABCTracer-inspired) retains higher Recall** (0.975 vs 0.900). Recall-oriented and calibration trade-offs are analyzed in Section 5.

### 4.6.1 ETH-BNB Protocol Generalization

To address the concern that the canonical benchmark is Celer-specific, we extend the same ETH-BNB route to Multichain and PolyNetwork using the published Validation labels under `data/Validation/ETH-BNB`. No live NodeReal calls are made: source features and BNB candidate pools are read from cached RPC logs under `out/multi_bridge_expansion`. The split follows the same chronological development/test protocol as the Multi/Poly cache, and all methods use identical per-protocol candidate pools.

**Table 7 — ETH-BNB protocol generalization, held-out raw transaction-pair full-set**

| Protocol | Method | Pairs | Candidate recall | Precision | Recall | F1 | Coverage | Abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multichain | UOT-Q | 2444 | 1.000 | 0.316 | 0.005 | 0.010 | 0.016 | 0.984 |
| Multichain | Connector-style adapted | 2444 | 1.000 | 0.993 | 0.964 | 0.978 | 0.971 | 0.029 |
| Multichain | ABCTracer-style adapted | 2444 | 1.000 | 0.959 | 0.959 | 0.959 | 1.000 | 0.000 |
| PolyNetwork | UOT-Q | 1630 | 1.000 | 0.889 | 0.783 | 0.833 | 0.881 | 0.119 |
| PolyNetwork | Connector-style adapted | 1630 | 1.000 | 0.999 | 0.998 | 0.998 | 0.999 | 0.001 |
| PolyNetwork | ABCTracer-style adapted | 1630 | 1.000 | 0.998 | 0.998 | 0.998 | 1.000 | 0.000 |

*Source: `out/multi_bridge_expansion/summary.csv` and per-protocol `rc_uot_q_test_metrics.json`, `connector_test_metrics.json`, `abctracer_test_metrics.json`. Cached BNB logs only; no live RPC. ABCTracer original checkpoint remains unavailable, so the ABCTracer-style adapted diagnostic is reported.*

The extension shows two distinct regimes. PolyNetwork is separable under raw transaction-pair evaluation: UOT-Q reaches F1 **0.833** with coverage **0.881**, while both adapted baselines remain strong on the near one-to-one anchors. On Multichain, UOT-Q does **not** produce usable raw transaction-pair output: it essentially abstains (abstention **0.984**, raw pair F1 **0.010**), while the adapted baselines reach 0.978/0.959. We state this plainly rather than as a strength: near-total abstention is the designed fail-safe of a coverage-qualified flow-level model, not a retrieval result, and it is exactly the behavior specified for evidence-insufficient regimes. Connector-style and ABCTracer-style adapted baselines retain high raw pair F1 on both protocols, mirroring their Celer behavior. Celer remains the canonical benchmark; its Table 4 flow-stress and Table 5/6 decode-only results use different evaluation scopes and are not inserted into this raw full-set table.

## 4.7 Prototype Validation

**Cross AML** implements the evidence-to-report pipeline described in §3. The released artifact supports **dry-run validation** without live RPC credentials by replaying bundled evidence. In this mode it executes input parsing, evidence replay, flow construction, coverage qualification, UOT-Q-style scoring, and interpretable report generation. The artifact package includes unit tests and reproducibility instructions for reviewers; implementation and security details are not repeated here. This validation shows that the proposed pipeline is not only an offline evaluation protocol, but can also be executed as a standalone evidence-to-report workflow (RQ5).

## 4.8 Summary of Findings

The experiments support six main conclusions:

1. **Benchmark.** The Celer-supervised corpus provides a reproducible flow-level CSFFC evaluation setting at scale (Table 1: 7,296 anchor pairs; 7,128 supervised flow labels).

2. **Structural transport.** UOT covers split/merge structure under controlled semi-synthetic stress in the edge-inclusion sense (Table 2; Figure 5: split edge-inclusion recall **0.946**, merge edge-inclusion recall **0.967**; Table 2b: 0.950–1.000 across three bridges). A strict exact-topology evaluator applied to the same frozen operating point gives 0.000 for every method on every bridge, and on edge-level metrics the calibrated threshold heuristic dominates while UOT-Q stays above strictly balanced OT (Table 2d). The capability audit (Table 2c; TEST A–F) shows the many-match methods can represent 1→2 / 2→1 and abstention that the one-to-one decoders structurally cannot; the mechanism ladders show UOT-Q retaining a consistent edge-metric offset over balanced OT under mass mismatch, unmatched flows, and decoys, while pairwise threshold false positives grow fastest with decoy density (§4.3.3–4.3.4). Separately from the split/merge representation-capability result, a preregistered confirmatory holdout on untouched seeds (301–305, three bridges) confirmed that conditional-plan decoding repairs correspondence ranking relative to direct decoding of the raw transport plan: dual-cancelled conditional decoding raises macro edge F1 over raw-plan decoding by **+0.0754 [0.0706, 0.0803]**, with the preregistered mechanism directions on all five indicators (§4.3.5; Table 2e; Figure 5f; sensitivity intervals in Supplement S.2).

3. **Full-anchor fixed-delay admissible decoding.** On all **7,296** anchor pairs, recommended top-3 admissible rescue preserves nearly full coverage and recovery while sharply reducing tx-level temporal violations; a stricter joint filter yields high-precision forensic subsets (Table 5; §4.4). Fixed-delay leave-anchor-out audit confirms gains are not explained by bridge-key or bridge-evidence leakage (Appendix Table A.1).

4. **Covered quotient and baselines.** UOT-Q's coverage machinery reproduces its own event-backed quotient grouping exactly on the **122** covered holdout pairs (a definitional consistency check keyed on the same bridge-event transferId incidence; Supplement S.1; Figure 6; §4.5), reports original-system symmetric masking degradation (Table 6; Appendix B), and reports higher **Precision and F1** than the untuned style-adapted representation controls on the independent held-out test set (Table 4; Figure 7; the calibration budget was asymmetric and favored the proposed pipeline — §4.6), with broader scope and trade-off interpretation in Section 5.

5. **ETH-BNB protocol generalization.** The cached Multi/Poly extension shows Connector-style and ABCTracer-style adapted baselines remain strong on near one-to-one anchors, while UOT-Q on PolyNetwork reaches raw pair F1 **0.833** at coverage **0.881**; Multichain remains a high-abstention regime for UOT-Q (Table 7).

6. **Post-development data-only audits (problem validity only).** Two disjoint, prediction-blind post-development Celer windows, assembled data-only under the frozen protocol (§4.9), contain **2,451** and **5,516** Tier-A source-level fan-out units, confirming that real protocol-grounded flow-level fan-out structure exists outside the development corpus. Both failed the preregistered adequacy criteria (v4: G = 7 < 8; v5: max cluster share 0.6200 > 0.5 and conservative G = 2); no method was executed on either corpus and no external-performance claim is made.

## 4.9 Post-Development Data-Only Audits (Problem Validity; No Method Performance)

To assess whether non-one-to-one flow structure occurs outside the development corpus, we conducted two prediction-blind, data-only audits on disjoint post-development Celer windows, both assembled under the same frozen protocol: the protocol-native linkage `Send.transferId == Relay.srcTransferId` (Ethereum to BNB Smart Chain; `Send.dstChainId == 56`, `Relay.srcChainId == 1`), per-block contract-continuity checks, independent ground-truth verification (anchor-set and transaction-identity), and disjointness against all historical development and prior-window transaction identities (zero intersections in every block). No solver was run, no cost or plan was constructed, and no method prediction was generated on either corpus.

Both windows contain large numbers of protocol-grounded flow-level fan-out units (Table 8). Protocol-level Celer settlement remains transaction-pairwise: each anchor links one source transaction to one destination transaction. The non-one-to-one structure arises at the forensic **flow** level, when pairwise settlements aggregate around one source or destination flow; the fan-out units are therefore flow-level aggregations, and they do not imply that the bridge protocol natively splits a single settlement into multiple destination settlements.

**Table 8 — Post-development data-only audits (prediction-blind; no method execution)**

| Quantity | v4 window | v5 window |
|---|---|---|
| Window | 2024-05-20 → 2025-05-19 (UTC) | 2025-05-14 → 2026-05-09 (UTC) |
| Protocol-native anchors | **32,905** | **42,114** |
| Unique flow-level GT edges | **25,621** | **39,595** |
| Tier-A source-level fan-out units | **2,451** | **5,516** |
| Merge units | **85** | **76** |
| Temporal span | **359.98 days** | **359.97 days** |
| G_full (independent primary-address clusters) | **7** | **8** |
| max-share_full (largest cluster / total fan-out units) | **0.371277** | **0.6200** |
| G_conservative (clusters with addresses unseen in development / prior corpora) | **2** | **2** |
| Adequacy (preregistered joint criteria) | **FAIL** | **FAIL** |
| Method executed? | **NO** | **NO** |

*Source: frozen data-only artifacts (v4 and v5 temporal packages; prediction-blind; adequacy per the preregistered joint criteria — G ≥ 8, max share < 0.5, Tier-A fan-out ≥ 30, spread ≥ 2 calendar months; the conservative count applies the frozen OVERLAP_FAMILIAR rule, Supplement S.3). Conservative max-share is reported under the frozen denominator convention — largest unfamiliar-cluster size divided by total fan-out units: 0.0245 (v4) and 0.0009 (v5).*

**Interpretation.** (1) Both independent post-development windows contain large amounts of protocol-grounded flow-level fan-out — this supports **problem existence**. (2) It does not support **method performance**: no method ran on either corpus. (3) The v4 corpus failed the behavioral-source diversity criterion (G = 7 < 8). (4) The v5 corpus reached G_full = 8 but failed the concentration criterion (max share 0.6200 > 0.5), and under the frozen conservative rule its unfamiliar-cluster count was 2 — persistent aggregator/relayer addresses recur across windows and are marked OVERLAP_FAMILIAR — so the diversity criterion also fails conservatively. (5) Both corpora are archived as problem-validity/provenance assets; neither is an evaluation population.

**Why the gate is what it is, and why we did not change it.** The gate G ≥ 8 is not a universal statistical threshold and we do not claim it is one. G counts independent primary-address clusters, a proxy for **behavioral-source diversity**, not sample count: the corpora contain thousands of fan-out units, but because those units concentrate in few clusters they cannot be treated as thousands of independent external actors. The gate was fixed — together with the other three criteria — **before the external corpus structure and any method outcome were observed**; its scientific value comes from pre-specification plus conservative diversity control, not from any property of the number eight. Relaxing a criterion after observing the data would be a post-data change to a confirmatory adequacy rule; we therefore retain the preregistered failure decisions, and we note that no method prediction existed at the time adequacy was evaluated, so neither failure could have been influenced by any method outcome.

The provenance of these audits, including the earlier v3 external execution failure and the decision not to rerun, is summarized in §5.8 and detailed in Supplement S.3. The concentration of fan-out activity is discussed as a dataset-specific descriptive observation (§5.9). The raw per-block artifacts and frozen hash manifests of both corpora are retained in the reproducibility package (Supplement S.3).

# 5. Discussion

This section discusses the applicability of UOT-Q in cross-chain AML settings, the role of evidence coverage qualification, deployment boundaries, and directions for future extension.

## 5.1 What the Results Show

The experimental program supports a coherent picture across six layers.

**Flow-level transport (UOT).** Semi-synthetic stress tests show that unbalanced transport (UOT) with risk-weighted inputs can represent split and merge structure in the edge-inclusion sense (split edge-inclusion recall **0.946**, merge edge-inclusion recall **0.967**; §4.3)—edge-inclusion recall at decoded edge precision ≈0.02, while strict exact-topology recovery is **0.000** for every method on every bridge and the globally calibrated per-cell threshold rule dominates edge F1 (Table 2d). This indicates that soft, mass-bearing correspondence is a viable representational alternative to brittle one-to-one transaction matching when funds fan out or consolidate across bridge legs, under controlled semi-synthetic conditions only.

**Conditional-plan decoder repair.** A preregistered, hash-locked confirmatory holdout on untouched seeds across three bridges established that dual-cancelled conditional decoding repairs the raw transport plan's correspondence ranking: macro edge F1 rises by **+0.0754 [0.0706, 0.0803]** over raw-plan decoding, positive on every bridge, with all five preregistered mechanism-direction indicators in the repair direction and independent verification of every recomputed result (§4.3.5; Table 2e; Figure 5f). This is a decoding-level repair justified algebraically by the dual-cancellation Proposition (§3.4); the same holdout does not attribute it to the unbalanced relaxation (balanced vs unbalanced conditional decoding differ by −0.0003 with a CI including zero). The transport-plan-level increment beyond direct kernel decoding is positive but small in absolute terms (Δ_cost = +0.0048 [0.0013, 0.0089]; Δ_support = +0.0104 [0.0062, 0.0152]), so the transport solve should be read as a representation mechanism, not as a large ranking gain over its own kernel.

**Full-anchor ranked correspondence and admissible decoding (UOT / UOT-Q).** On all **7,296** anchor pairs, the raw tx-level projection is a **compatibility projection** only (pair F1 **0.589**, tx-level CVR **0.338**); it is not a forensic admissibility claim. Formal high-coverage decoding (**positive_delay_top3_rescue**) reaches pair F1 **0.590** with coverage **0.996** and tx-level CVR **0.005**; the forensic high-confidence subset (**joint_time_admissible_filter**) yields precision **0.889** and pair F1 **0.708** at coverage **0.845** (Table 5; §4.4). Fixed-delay leave-anchor-out audit confirms these gains are not explained by bridge-key or bridge-evidence leakage (Appendix Table A.1).

**Event-backed covered quotient consistency (UOT-Q).** On the **122** covered holdout pairs, the development-frozen bridge-transfer-key rule reproduces the transferId-consistent quotient grouping exactly (Supplement S.1). Because the covered-pair label and the rule are keyed on the same bridge-event transferId incidence, this is a **definitional consistency check** on the event-backed projection, not an independent accuracy result: it shows that the coverage machinery applies its own evidence-scoping rule consistently, not that every canonical edge in the full label universe is resolved (projection coverage ≈ 0.792).

**Precision-oriented baseline comparison.** On an independent held-out test set not used for model selection, UOT-Q **reports higher Precision and F1 than** the untuned **heuristic and style-adapted representation controls** (Table 4)—representation-capability controls, **not original-system Connector or ABCTracer runs**, and not a fair-budget comparison (the historical calibration budget was asymmetric and favored the proposed pipeline; §4.6). This difference is specific to the flow-stress holdout operating point: on the cached raw transaction-pair extension (Table 7) the adapted baselines remain stronger (Multichain 0.978/0.959 vs UOT-Q 0.010; PolyNetwork 0.998/0.998 vs 0.833), and under native closed-set semantics Connector reaches tx-pair F1 **0.9736** (Table 6; Appendix B). This operating point is relevant only when investigators prioritize fewer false-positive escalations under flow-stress evaluation, and even then the absolute precision remains 0.192 (§5.3).

**Executable workflow (Cross AML).** Prototype validation confirms that the same evidence-to-report pipeline used in evaluation can run as a standalone triage workflow, including dry-run replay without live RPC credentials (§4.7).

## 5.2 Coverage Qualification as an AML Safety Mechanism

AML automation should **fail safely** when evidence is incomplete [@fatf2023virtualassets; @rudin2019stop]. **Coverage qualification** encodes that principle: correspondence hypotheses are promoted to high-confidence output only when event-backed quotient structure and parser coverage support them; otherwise the system abstains or emits diagnostic-only leads.

Event-backed projection coverage is approximately **0.792** on our benchmark. Within the covered holdout subset the frozen rule reproduces its own quotient grouping exactly on all **122** pairs. **This is not full-universe recovery and not an independent accuracy measurement; it is a definitional consistency check within the evidence-covered scope (Supplement S.1).** Uncovered or weakly backed cases should surface evidence gaps and defer to analysts or additional collection rather than inherit covered-subset metrics. Coverage tiers and abstention are therefore a **design-time safety mechanism**, not a post-hoc excuse for poor global recall.

## 5.3 Precision Trade-offs vs. Recall-Oriented Retrieval

UOT-Q and the **style-adapted diagnostic baseline (ABCTracer-inspired)** occupy **different operating points**. UOT-Q is **precision-emphasizing and evidence-qualified**: it targets flow-stress Precision and F1 under coverage qualification and explicit unmatched mass. The **ABCTracer-inspired diagnostic adaptation** remains useful for **recall-oriented candidate retrieval** and high-recall behavior on adapted flow labels, as reflected in its higher recall and calibration profile on the held-out partition (Table 4)—**not as a claim about original ABCTracer system performance**. **Absolute-precision honesty.** At the flow-stress operating point the absolute precision is **0.192**: approximately four of five promoted flow edges are false positives. This operating point is therefore not presented as investigator-actionable precision; it is a representational and ranking-hygiene result, and the tool's reports require human review. The one healthy-precision figure (0.889, joint filter) lives in the one-to-one anchor regime where closed-set Connector reaches 0.9736 (Table 6), so no operating point with both investigator-grade precision and non-one-to-one coverage is claimed.

We do not treat either approach as universally dominant. The practical implication is **complementary deployment**: recall-oriented diagnostic tracing can widen the candidate pool, while UOT-Q can rank and qualify flow-level hypotheses for investigator triage. Recall-oriented and calibration trade-offs relative to **diagnostic baselines** should be weighed alongside Precision/F1 gains when choosing an operating point.

## 5.4 Observability, Cold Start, and Deployment

Cross-chain correspondence quality depends on what investigators can observe on-chain and supply as input.

**Observability layers.** Bridge protocols differ in event schemas; parsers must track ABI variants and upgrades. Standard JSON-RPC may lack historical logs, traces, or internal transactions needed for complete reconstruction. ERC-20 transfers and bridge-like logs may be partially parsed, leaving Tier C pairs diagnostic only. Off-chain flows—CEX deposits, mixers, and privacy pools—break on-chain continuity. Large-scale **candidate discovery** further depends on indexer, archive, or trace backends when destination hashes are not provided upfront.

**Cold-start behavior.** Investigators often encounter new bridges, tokens, or addresses without tuned parsers or historical labels. Cross AML supports a graded response: a **high-confidence covered match** when event-backed quotient evidence and scores exceed thresholds (Tiers A/B); a **low-confidence lead** when scores suggest follow-up but evidence is partial; **abstention** when coverage is Uncovered or Tier C evidence is insufficient for escalation; and an **evidence gap report** listing missing logs, unparsed events, or discovery limits for targeted re-collection.

Reports surface coverage tier, abstention reasons, and discovery limits explicitly, consistent with explainability and auditability expectations in regulated AML workflows [@arrieta2020explainable; @liu2022compliance]. Cross AML is a **research prototype** for triage—not production compliance software—and outputs require human review before external escalation.

## 5.5 Scope of Claims

**Final claims are intentionally scoped** to what the benchmark and held-out evaluation support.

These results do not establish:

- **Universal superiority** over original Connector, original ABCTracer, or their diagnostic adaptations across all reported settings;
- **Full-scope high precision or recall** across the entire flow-label universe when event-backed projection coverage is approximately **0.792**;
- **Key-metric dominance** on every structural, calibration, and retrieval dimension;
- **Balanced dominance** relative to all baselines on all key metrics;
- **External method performance** on a fully adequate independent temporal corpus (§4.9; the data-only audit did not meet the preregistered adequacy gate, and no method was executed on it); or
- **A UOT-specific advantage** over balanced OT (the confirmatory holdout's balanced vs unbalanced conditional decoding differ by −0.0003 with a CI including zero).

High covered-quotient metrics apply to the **event-backed covered subset** only; they must not be read as full-population recall. Industry threat intelligence [@chainalysis2024laundering; @elliptic2025crosschain] motivates the problem setting but does not constitute evidence of model dominance.

## 5.6 Limitations and Future Work

**External method performance remains unestablished.** An initial preregistered Level-I external execution entered the prediction path but failed before valid performance could be obtained because of an implementation defect and a zero-marginal solver-domain defect; the corpus was not rerun. The solver-interface repair was mathematically justified and validated only on non-test data before a new disjoint temporal corpus was collected. That corpus (§4.9) then failed the preregistered adequacy gate (G = 7 < 8), so no method predictions were generated: real structural method performance on a fully adequate independent temporal corpus is not established, and no external-performance claim is made. Both post-development corpora failed the preregistered adequacy criteria (v4: G = 7 < 8; v5: max cluster share 0.6200 > 0.5 with conservative G = 2), so no method ran on either; real structural method performance on a fully adequate independent temporal corpus is not established, and no external-performance claim is made. Fan-out activity was concentrated in few primary-address clusters in both windows; this is a dataset-specific descriptive observation, not a claim about the wider ecosystem, and it is what limited the preregistered external-performance inference. The external adequacy design is deliberately conservative: the familiarity rule itself causes the conservative collapse of G in both windows, because persistent aggregator/relayer addresses are exactly the addresses most likely to recur across temporal corpora and thus to be marked OVERLAP_FAMILIAR. The conservative count therefore under-represents recurrent infrastructure actors by construction. Therefore, failure to satisfy the diversity gate does **not** imply absence of real fan-out structure; it means the current study cannot support statistically diverse external method-performance inference under its frozen criterion. What constitutes an independent behavioral source for persistent cross-chain infrastructure is itself an open research question, left to future work as a new scientific design — it cannot retroactively change these failure records.

**Protocol settlement vs flow structure.** Protocol-level bridge settlement is primarily one-to-one (transaction-pairwise anchors); non-one-to-one structure arises at the forensic flow correspondence level, when settlements are aggregated into flows. We do not claim that the bridge protocol natively splits a single settlement into multiple destination settlements.

**UOT-specific advantage.** A UOT-specific advantage over balanced OT was not established (Δ_bot CI includes zero); the unbalanced relaxation is motivated by unmatched mass, partial/open-world correspondence, and abstention-compatible modeling, not by a proven ranking superiority.

**Real merge evidence.** Real N→1 merge prevalence and powered external evidence are limited (v4: 85 merge units; Table 1 merge share 0.17%); semi-synthetic structural stress tests remain necessary for controlled merge evaluation.

The canonical benchmark remains **Celer-centered** for the flow-level claims, but the ETH-BNB route has been extended to cached Multichain and PolyNetwork validation labels. This extension reports raw transaction-pair full-set metrics and shows that the adapted transaction-level baselines remain strong on near one-to-one anchors, while UOT-Q is a coverage-qualified flow-level model rather than a raw hash-pair oracle. Supervision still relies on official bridge anchors; generalization to weak or heuristic labels remains out of scope. Semi-synthetic split/merge results characterize transport behavior under controlled stress and do not replace full-pool real-data ranking; strict exact-topology recovery is zero for all methods on this benchmark family (Table 2d), and the decoder is fixed at k = 5 with observational (not causal) mechanism indicators. The 720 confirmatory template instances share one injected pattern composition, so the repair result is specific to this generator family, and fan-out units with ground-truth degree above 5 are structurally unrecoverable for exact recovery by any top-5 decoder (known representational limit). Unmatched and decoy analyses remain diagnostic rather than claims of strong rejection at scale.

Operational limits include dependence on bridge-event observability and parser coverage, limited statistical power in multi-seed ablations at current scale, and the need for richer backends when investigators expect large-scale destination discovery without seed hashes. Future work should broaden bridge and parser coverage, deploy on live networks with archive and trace tiers, run larger seed sweeps with tighter confidence reporting, and conduct investigator-in-the-loop studies on triage quality and time-to-decision. Any future external-validation study would require a new independent scientific question and fresh preregistration; no new temporal window may be opened merely to satisfy the G ≥ 8 gate.

## 5.7 Ethical Use

Automated cross-chain correspondence can harm individuals and organizations if misused. Reports provide **investigation leads**, not adjudicated guilt or legal proof. **Human review** is mandatory before escalation or external accusations. **Abstention** when evidence is insufficient is part of the safety design—suppressing abstention to inflate recall undermines both AML practice and the method’s intent. Investigators should combine on-chain hypotheses with off-chain intelligence, legal process, and institutional policy, and avoid unsupported public accusations based on model output alone.

## 5.8 Experimental Provenance and Reproducibility

The evaluation chain, summarized here and detailed in Supplement S.3 and the reproducibility package, satisfies four properties: the development/calibration splits were fixed before evaluation; the 301–305 confirmatory holdout was generated once, was never touched for selection, and was recomputed independently from raw per-cell artifacts (matching within 1e-9); both external data audits were prediction-blind (no method ran before or after adequacy failure); and the frozen candidate, decoder constants, and preregistration hashes are archived with the released package. Two distinct repairs appear in this paper and should not be conflated: the **decoder repair** (the dual-cancelled conditional decoder, Proposition 1, §3.4, confirmed on the 301–305 holdout) and the **solver-interface repair** (support-aware solver wrappers built after the v3 failure, validated only on non-test data, and never used to produce any new performance number). The v3 external execution failure, the quarantined blocks, the seed map, and the literal hash values are recorded in Supplement S.3.

## 5.9 Real-Data Fan-Out Concentration in the Independent Corpus

The two data-only audits found that real flow-level fan-out structure exists outside the development corpus in both windows (2,451 and 5,516 Tier-A source-level fan-out units), but that this activity was heavily concentrated in a few primary-address clusters (v4: seven clusters, maximum share 0.371277; v5: eight clusters, maximum share 0.6200). As dataset-specific descriptive observations, these numbers mean fan-out in those windows was dominated by a small number of persistent aggregator or router addresses. They do not license the generalization that the ecosystem is dominated by those actors, and they say nothing about method performance: the concentration is precisely what limited the preregistered external-performance inference in both windows, because the frozen adequacy criteria were not met and no method was executed on either corpus.

## 5.10 Data and Code Availability

All frozen artifacts referenced in this paper are part of the accompanying reproducibility package (to be released with the camera-ready version; the release URL and open-source license declaration will be inserted at that time). The package is self-contained: its manifest (`R5C_REPRODUCIBILITY_BUNDLE_MANIFEST.md`) enumerates the Cross AML prototype, the frozen evaluation scripts and hash-locked one-shot runner with the separately maintained verifier, the frozen confirmatory artifacts, the v3 failure record and quarantined blocks, and both post-development data-only corpora with their frozen design packages, hash manifests, data manifests, disjointness and adequacy reports, cluster summaries, and collection logs — each referenced by a package-relative path, so no reader depends on author-local machine paths. The v5 data manifest records `method_predictions = 0` and `method_runs = 0`. Raw chain data are re-fetchable from public Ethereum and BNB Smart Chain nodes using the documented block ranges; provider API keys are not released. The Celer anchor/Validation labels used for benchmark construction derive from published Celer data; their redistribution terms are stated in the package. No off-chain identity or exchange-internal data are included.

# 6. Conclusion

Cross-chain laundering forces AML analysis beyond single-chain transaction tracing toward **flow-level correspondence** across bridges, tokens, and ledgers. We introduced **CSFFC (Cross-Chain Suspicious Fund Flow Correspondence)** as the task formulation, with flows as AML observation units that support split, merge, delay, and partial observability.

**UOT** addresses CSFFC through an unbalanced optimal transport (UOT) formulation with risk-weighted inputs, producing soft correspondence plans with explicit unmatched mass. On semi-synthetic structural stress tests, UOT achieves mean split and merge edge-inclusion recall of **0.946** and **0.967**—inclusion recall at decoded edge precision ≈0.02—demonstrating that non-one-to-one flow matching is representable under controlled conditions; strict exact-topology recovery is zero for every tested method, a calibrated threshold rule dominates edge F1 (Table 2d), and a code-level capability audit shows that one-to-one decoders structurally cannot represent one-to-many correspondence.

**Conditional decoding.** We diagnosed that raw transport-plan decoding mixes kernel ranking with the solver's dual scalings, and a preregistered, hash-locked confirmatory holdout on untouched seeds across three bridges confirms the repair: dual-cancelled conditional decoding raises macro edge F1 by **+0.0754** (paired 95% CI **[0.0706, 0.0803]** (template-instance bootstrap; aggregation hierarchy template → seed → bridge → macro)), positive on every bridge, with all five preregistered mechanism-direction indicators in the repair direction and an independent verifier recomputing every result. The dual-cancellation Proposition (§3.4) makes the repair algebraic; this is a decoding-level contribution, presented as an adaptation of standard Sinkhorn structure rather than a new normalization, and it does not imply a UOT-over-BOT advantage (balanced vs unbalanced conditional decoding differ by −0.0003 with a CI including zero).

The **UOT-Q pipeline** extends the UOT formulation with event-backed quotient groups and **coverage qualification**, enabling abstention when evidence is insufficient. On the Celer benchmark—**7,296** anchor transaction pairs and **7,128** frozen flow labels—its coverage machinery reproduces its own event-backed quotient grouping exactly on the covered holdout pairs (a definitional consistency check, Supplement S.1). Development-sealed diagnostic comparisons against untuned heuristic and style-adapted representation controls are reported with their calibration asymmetry explicitly disclosed (§4.6); original Connector native behavior is reported separately as a closed-set upper-bound diagnostic (Table 6; Appendix B).

We release **Cross AML** (`tools/cross_aml/`), an open-source research prototype that implements RPC evidence collection, flow construction, coverage-qualified matching, and interpretable reporting for investigator triage.

Taken together, CSFFC, UOT, UOT-Q, and Cross AML provide an **evidence-aware, auditable, coverage-qualified** path from transaction hashes to scoped cross-chain fund-flow correspondence hypotheses—supporting AML workflows that prioritize explainability, abstention, and human review over brittle one-to-one hash matching. Method performance on a fully adequate independent real-world corpus remains unestablished: two prediction-blind post-development data audits (§4.9) established that real flow-level fan-out structure exists outside the development corpus, but both failed the preregistered adequacy criteria and no method was executed on either, so no external-performance claim is made.

---

# Appendix A — Fixed-Delay Anchor Audit

See `appendix_A_fixed_delay_anchor_audit.md` for **Table A.1** (baseline, leave-key-out, leave-anchor-out-strict, fake-oracle probe, and permuted-label control across raw projection, top-3 rescue, and joint filter decodings). Permuted-label control rows retain raw timing/CVR/coverage values for audit traceability; they are not interpreted as method performance (see table footnote).

# Appendix B — Connector Native Closed-Set Diagnostic and Route A Consistency Audit

See `appendix_B_connector_native_diagnostic.md`. Phase 1 canonical Connector native closed-set diagnostic: tx-pair F1 **0.9736** (n_predicted **6,954**, n_correct **6,937**, n_no_match **342**). Route A v2 reproduces Phase 1 full_native predictions exactly (diff = 0). Table 6 reports the symmetric masking degradation ladder including full_native Connector with closed-set caveats; Appendix B provides audit traceability. ABCTracer remains blocked (no official checkpoint).
