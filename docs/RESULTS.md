# Public-data CPU-pilot results

The end-to-end public-original-query pilot has run and its final metrics pass
an independent audit. The proposed quality gains were **not demonstrated**:
adaptation did not improve dense final nDCG, and the learned gate trails
dense-only and validation-tuned fixed fusion, although it beats equal fusion.
Working software is not evidence of an improved retrieval method.

## Locked experiment

Each language uses the same fixed 1,500 passages across compared systems.
Final evaluation contains 24 original MIRACL official-development query groups
per language, 72 total. Whole groups retain the original prepared splits.
All selected judged passages, including grade-zero judgments, are retained.
These are separate monolingual tracks, not a pooled cross-lingual benchmark.

The encoder pilot uses 72 original training queries across the three languages,
one SGD epoch and predeclared seeds 13 and 29. Gate fitting uses the separate
fusion-training split; validation alone selects its hyperparameters and fixed
fusion weights. Serving is top-10; ranking replay evaluates Recall@100.
Frozen and adapted encoders share 128-token right truncation and batch size 8.

A [729-file final lock](../artifacts/study/cpu-pilot/final-lock.json) was
verified before and after final execution. There was no final-set fitting,
retuning or winner selection. Seed 13/full gate remains the predeclared
deployment demonstration; seed 29 is replication. Both results are retained.

## Final language-macro nDCG@10

| Method | Frozen E5 | Adapted seed 13 | Adapted seed 29 |
|---|---:|---:|---:|
| Original-query BM25 | 0.585748 | 0.585748 | 0.585748 |
| Protected Roman-view character TF-IDF | 0.362282 | 0.362282 | 0.362282 |
| Original-query dense | 0.729563 | 0.729415 | 0.728356 |
| Equal fusion | 0.645935 | 0.641743 | 0.643710 |
| Validation-tuned fixed fusion | 0.695316 | 0.700173 | 0.699647 |
| Full learned gate | — | 0.670070 | 0.672833 |
| Retrained without uncertainty features | — | 0.665776 | 0.670133 |
| Retrained without agreement features | — | 0.679432 | 0.685106 |

The lexical baselines are unchanged across encoders. Gate ablations are genuine
retrained models, not post-hoc zeroing of fitted weights. The no-character
fixed-fusion result in the full report is only a weight intervention, not a
retrained character-branch ablation.

Full-gate MRR@10 is 0.650265/0.651802 and Recall@100 is 0.995370/1.000000
for seeds 13/29. Detailed per-language nDCG/MRR/Recall, group-bootstrap
intervals, missing-query counts, branch availability, measured shared-trace
costs and descriptive seed variation are in the
[complete report](../artifacts/study/cpu-pilot/final-report/report.md) and its
[machine-readable data](../artifacts/study/cpu-pilot/final-report/report.json).
Seeds share queries and are not independent query samples. Macro confidence
intervals are not invented by averaging language bounds.

## Descriptive error and ablation findings

Compared with dense-only over the same 72 groups, the full gate's query-level
nDCG wins/ties/losses are 18/22/32 for seed 13 and 17/19/36 for seed 29.
Against tuned fixed fusion they are 20/26/26 and 18/23/31. Against equal fusion
they are 31/34/7 and 29/34/9.

Uncertainty features have small positive final point differences versus their
retrained ablation (+0.004294/+0.002699 macro nDCG), while removing agreement
features improves the point estimate (+0.009362/+0.012273). These are
descriptive observations, not significance or general robustness claims.

For example, `hi:dev:230365#0` has full-gate nDCG 0.485229 versus
0.956591 for dense/fixed; `en:dev:4410` has full-gate nDCG zero versus
0.386853 for fixed fusion. These are verifiable ranking errors, not human
judgments about rewrite semantics. The
[independent final audit](../artifacts/checks/cpu-pilot-final-independent-audit-2026-10-20.json)
retains per-language contrasts and public-query loss examples. It recomputed
every query metric using separate standard-library formulas and ranking replay.
These results alone do not establish the cause of gate underperformance.

## Scope and unresolved larger experiments

- This is the 1,500-passage CPU pilot, **not** the prepared 50,000-passage
  experiment or full MIRACL benchmark. See [CAPACITY.md](CAPACITY.md).
- Only original public query/judgment gold is scored. The unavailable
  independently reviewed bilingual challenge and unreviewed variants are
  excluded. There is no noise-training, consistency-benefit or natural
  code-mixed/noisy robustness result.
- E5 has prior MIRACL-training exposure; the frozen result is not zero-shot
  MIRACL. Grouping handles exact normalized duplicates, not an independent
  semantic-paraphrase audit.
- Romanization is the recorded Dakshina-training/uroman baseline, not a
  validated IndicXlit run. Held-out word matching was low. Auxiliary LID has
  no Bengali gold; abstention and unavailable branches remain visible.
- A branch always masked on validation has no identifiable validated weight.
  Shared-trace retrieval costs are not standalone timings for every replayed
  baseline. Small pilot samples and retained judged passages limit generality.
- Automated HTTP/API/HTML testing is separate from browser visual review.
  The app returns ranked source passages and provenance, not generated answers.

Reproduction and real-model launch commands:
[PUBLIC_STUDY_RUN.md](PUBLIC_STUDY_RUN.md). Downloads, pins and licenses:
[REAL_DATA.md](REAL_DATA.md). Current engineering/integration status:
[PROGRESS.md](../PROGRESS.md).
