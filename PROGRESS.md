# Progress

## Module mapping

The requested module numbers refer to row order in `PROJECT_SPEC.md` section
15: 1=`data/prepare_corpus.py`, 2=`data/make_splits.py`,
12=`evaluation/evaluate.py`, 6=`retrieval/bm25.py`,
7=`retrieval/char_tfidf.py`, 8=`retrieval/dense.py`,
3=`data/generate_variants.py`, 4=`nlp/language_id.py`, and
5=`nlp/query_views.py`, 11=`retrieval/fusion.py`, and
9=`training/train_encoder.py`, 10=`training/train_gate.py`, and 13=`app.py`.

## Completed

- Module 1: streaming fixed-corpus preparation, qrel loading, deterministic
  distractor sampling, pooled ID rewriting, atomic artifacts, fingerprints, and
  fixture tests.
- Module 2: deterministic parent/group-level, language-stratified splits;
  official-development locking; review/judgment metadata; manifests and CLI.
- Module 12: strict ranked-run evaluation with hand-checked metrics, explicit
  exclusion policies, slices, robustness comparisons, group bootstrap, and CLI.
- Module 6: deterministic pure-Python BM25 over title/body, Unicode
  script-aware tokenization, reusable multi-view search, manifests, CLI, and an
  end-to-end preparation-to-evaluation fixture test.
- Module 7: sparse character 3–5-gram TF-IDF over explicit Romanized text,
  unavailable-branch accounting, manifests, CLI, and evaluator integration.
- Module 8: frozen E5 adapter, float32 vector persistence, encoder/index
  compatibility checks, deterministic inner-product and multiview search,
  explicit fixture CLI, and evaluator integration.
- Module 3: deterministic protected typos; supplied rewrites with provenance;
  split-safe candidate audit, content-bound reviews, and verified-only query
  artifacts with inherited judgments. Shared NFC token/protection spans include
  names, dates, numbers, negation, URLs, and email addresses.
- Module 4: deterministic supervised token language identification with
  character/context features, confidence/margin abstention, calibration on
  disjoint auxiliary data, persisted models, and train/predict CLIs.
- Module 5: exact original query preservation, protected NFC-span rewrites,
  at most two distinct native hypotheses, complete Roman-view availability,
  versioned supplied lexicons, an optional IndicXlit adapter, diagnostic
  features/call budgets, and single-query/batch CLIs.
- Module 11: shared-index four-branch retrieval orchestration, averaged
  within-branch RRF, fixed-weight final RRF, inactive-branch masking,
  deterministic candidate limits, contribution/cost traces, inference weight
  hook, reproducible run files, and CLI. Gate fitting remains Module 10.
- Module 9: training-only reviewed pairs and negative mining; differentiable
  multi-positive retrieval and consistency losses; real PyTorch optimizer
  loops; safe content-addressed checkpoints; and compatible trained dense
  loading/re-indexing. Optional pretrained E5 and explicit neural fixtures use
  the same preprocessing contract.
- Module 10: dependency-free linear softmax gate with observable features,
  masked nDCG utility targets, fusion-training-only scaling/supervision,
  content-bound review and split checks, deterministic optimization, sealed
  JSON artifacts, frozen-index/resource compatibility, validation comparisons
  to equal and language-macro-tuned fixed weights, hyperparameter selection,
  and integration with the fusion CLI through `--gate-model`.
- Module 13: local-only, dependency-free browser search and JSON API over
  the real retrieval pipeline; full passage metadata, query views, branch
  weights/contributions, explicit offline demo fixtures, persisted gate/dense
  loading, bounded inputs, escaped HTML, loopback/Host restrictions, and a
  wheel-installed `uncertainty-ir-demo` entry point.

## Verification (17 October 2026)

- Module 7 checkpoint: 54 tests passed with warnings treated as errors under
  two Python hash seeds.
- Module 8 checkpoint: 61 tests ran; 60 passed and the cached real-E5 test
  skipped explicitly because torch/transformers and the checkpoint are absent.
  Both Python hash seeds passed before the final metadata hardening; the final
  full-suite checkpoint also passed with warnings treated as errors.
- Module 3 checkpoint: 20 focused tests pass; the full suite runs 88 tests
  (87 pass, one optional E5 skip) under hash seeds 0 and 917 with warnings
  treated as errors. Preparation/splitting, reviewed variants, retrieval,
  serialization, and evaluation integration are covered.
- A subsequent Module 3 integrity regression increased its focused tests to 21;
  edited bound approvals are rejected on reload. The full suite then passed 89
  tests with one optional skip.
- Module 4 checkpoint: 13 focused tests pass; 102 accumulated tests run under
  both hash seeds (101 pass, one E5 skip), with warnings treated as errors and
  compilation passing. The supplied toy data trains, calibrates, persists, and
  predicts English/Romanized Hindi/Romanized Bengali tokens successfully.
- A finite-weight numerical regression raises Module 4's focused count to 14;
  training and calibration remain finite even with large valid sample weights.
- Module 5 final checkpoint: 23 focused tests and one complete pipeline test
  pass. The integration exercises a three-language pooled corpus, fixed splits,
  content-bound variant review, persisted LID/dense models, batch protected
  views, BM25, character TF-IDF, controlled dense retrieval, run serialization,
  and evaluation for originals and a reviewed typo.
- Module 5 accumulated suite: **127 tests run, 126 pass, one optional cached-E5
  test skips** under both `PYTHONHASHSEED=0` and `PYTHONHASHSEED=917`, with
  `PYTHONWARNINGS=error`. Compilation and dependency-free wheel creation pass.
- Module 11 was completed and checked before Module 9 implementation: 18
  focused tests and 145 whole-project tests ran under hash seeds 0 and 917
  (144 passed, one optional E5 skip). The pooled prepared-corpus pipeline now
  includes final fusion as well as all separate retrieval branches.
- Module 9: all 24 focused tests pass with actual CPU PyTorch and safetensors,
  including finite gradients, changed weights, falling training loss,
  deterministic training, the lambda-zero ablation, multi-positive masks,
  reviewed-pair/split isolation, and rejection of incompatible dense indexes.
- An additional real Transformers test passes using a tiny, randomly initialized
  one-layer XLM-R with 384-dimensional embeddings. It verifies attention-mask
  pooling, normalized embeddings, autograd/optimizer updates, safetensors saving,
  unmocked local model/tokenizer reload, and persisted dense-index search.
- A separate complete pipeline check passes: corpus preparation → grouped
  splits → reviewed variants → actual neural fixture training → sealed
  checkpoint → persisted trained dense index → LID/protected views → fusion →
  serialized evaluation. Its fixture scores are not benchmark results.
- Final extended suite with neural dependencies: **171 tests ran under each of
  hash seeds 0 and 917; 170 passed and the cached pretrained-E5 check skipped**
  in both runs, with `PYTHONWARNINGS=error`. All source compiles. A wheel builds
  without downloading dependencies; imports of fusion, dense retrieval,
  training, and checkpoints from that wheel pass outside the source directory,
  and the installed package exposes the `training` extra.
- All Python source and tests pass `compileall`.
- A wheel builds successfully without downloading dependencies.
- The committed three-language fixture completes all four CLIs and retrieves
  all three fixture judgments at rank 1. Its nDCG@10, MRR@10, and Recall@100
  are 1.0; these are smoke-test values, not MIRACL benchmark results.
- Module 10 was implemented and whole-project checked before starting Module
  13: 200 tests ran under hash seed 0 (199 passed, one cached-E5 skip), and an
  expanded 205-test checkpoint passed under hash seed 917 (204 passed, one
  cached-E5 skip), with optional neural dependencies and warnings as errors.
  The finalized 33 focused gate tests pass under both seeds. Four additional
  integration tests cover multilingual preparation/splits/review/indexes,
  actual neural fixture training and checkpoint reload followed by gate
  training, language-macro fixed-weight selection, and adaptive fusion CLI
  traces/output safety. These fixture results do not establish benchmark quality.

## Limitations and next work

- The initial corpus loader does not generate passage Romanizations. The
  separate real-data preparation stage has now populated compatible versioned
  passage views; incomplete fields remain explicitly unavailable.
- Public MIRACL, Dakshina Hindi/Bengali lexicon members, and COMI-LINGUA sources
  are downloaded with recorded pins and local content hashes. Dakshina used
  generation-pinned partial archive reads, not a verified full-archive MD5.
  No CMIR access is claimed. Real-data preparation, CPU-pilot baselines and
  locked final evaluation are complete below; full-profile scores are not claimed.
- Pinned pretrained multilingual-E5-small weights have been downloaded and
  actual offline pretrained inference, indexing and validation passed. Real
  original-query adaptation has completed both seeds 13 and 29. The isolated
  seed-29 retry succeeded after the first attempt's memory termination.
  Adapted retrieval has been evaluated in the locked CPU pilot without an
  improvement over frozen dense nDCG. Fixture tests independently cover the
  optimizer and Transformers persistence path without benchmark claims.
  Neural dependencies and model downloads remain explicit optional steps.
- IndicXlit direction/call/caching behavior is tested with a mocked upstream
  engine. Real IndicXlit inference has not executed locally; its optional adapter
  reports the upstream default weights as unpinned. Supply pinned/versioned
  resources for reproducible real-data experiments.
- Real public LID preparation, training, calibration and held-out validation
  completed. The label-balanced 2,000-token Hindi-English auxiliary test has
  80.3% top-label accuracy and 22.75% abstention; no Bengali gold is present.
  Fluent-speaker retrieval-query review is still unavailable; auxiliary
  public token labels do not establish query-level semantic equivalence.
  Name protection is conservative and heuristic;
  explicit entity spans are supported.
- All functional modules 1–13 and the approved public-only CPU pilot are
  complete, including actual training/indexing/gates, final evaluation,
  independent audits and real-model application checks. Reviewed noisy query
  variants/challenge are omitted by the approved scope, not verified. Full
  50k-profile experiments remain a separate capacity milestone; real IndicXlit,
  contextual Bengali LID validation and browser visual review are not claimed.
  Software completion is not evidence of better retrieval quality.

## Research execution (18–20 October 2026)

The requested stages are executed in order, with a verification checkpoint
before advancing:

1. Final engineering checks and progress-log reconciliation — complete.
2. Versioned real datasets, isolated splits, passage Romanization, and
   validation of real language resources — complete, with the quality and
   coverage limitations below.
3. Reviewed noisy/code-mixed variants and independently authored/reviewed
   challenge queries — scoped public-only policy/audit complete; the
   unavailable human challenge and unreviewed variant supervision/evaluation
   are explicitly excluded, not fabricated or marked experimentally verified.
4. Real-data BM25/character/pretrained-E5/fixed-fusion baselines — complete
   for the separately labeled CPU pilot, validation only.
5. Real E5 fine-tuning, compatible re-indexing, separate gate fitting, and
   validation-only selection — complete for the CPU pilot; original-only E5
   adaptation, not noise/consistency training.
6. Locked final evaluation, ablations, robustness/resource/error analysis,
   reporting, and real-artifact demo integration — complete for the approved
   original-query CPU scope under the 729-file lock, without post-final retuning.

No human review, real model training, or real benchmark score is inferred from
the fixture tests. Dataset/model downloads and their versions must be explicit.

### Public-data scope decision (19 October 2026)

The user approved a public-data-only study after the verified sources were
found insufficient to replace both bilingual reviewed challenge sets. Use
original MIRACL query/judgment gold. Automatic protected views remain inference
transformations, not independently reviewed query variants. Encoder adaptation
will use original-query retrieval supervision with consistency weight zero;
noise/consistency and natural-query robustness claims are out of scope until
reviewed pairs exist. See `configs/public-data-study.json` and
`docs/REAL_DATA.md`.

The CPU-only machine has four cores and approximately 3.7 GiB RAM. The initial
end-to-end compute run uses a separately sealed 1,500-passage-per-language
pilot derived within the existing full-profile splits, retaining every selected
query's judged passage. The completed 50,000-per-language data stays available;
pilot results must never be labeled as full-profile or full MIRACL results.

### Stage 2 verification (19 October 2026)

- All 83 pinned MIRACL assets independently rehashed: 5,218,934,859 bytes.
  Full source scans cover 32,893,221 English, 506,264 Hindi and 297,265
  Bengali passages. Three fixed tracks contain exactly 50,000 passages each.
- All 73,878 selected source judgments, including 55,712 grade-zero rows,
  remain intact; all judged passages exist. Official-development locks,
  normalized duplicate-group isolation, split assignments and artifact seals
  pass. One Hindi training/development duplicate is explicitly excluded.
- Public COMI-LINGUA human-majority labels supply 6,000 training tokens,
  2,000 calibration tokens and 2,000 official-test tokens. Dakshina adds 2,000
  explicitly weak Bengali training words. No test groups/spans overlap model
  fitting/calibration. A 180-epoch real LID model is saved and calibrated;
  held-out accuracy is 0.803 and nonabstained accuracy is 0.8693 at 0.7725
  coverage. These balanced auxiliary labels do not measure retrieval quality.
- Annotation-schema/offset rejections are counted: many excluded COMI training
  rows use unsupported `h/e/u` labels, not necessarily incorrect annotations.
  No aliases were guessed. Prepared-source coverage is therefore limited.
- New passage Romanizations contain 138,736 attested Dakshina-training pairs
  plus 321,330 explicitly algorithmic corpus-derived pairs. Complete body views
  exist for 49,289 English, 48,513 Hindi and 41,725 Bengali passages; incomplete
  fields stay unavailable. Original text and importer artifacts are preserved.
- Dakshina held-out word matching is low: native-to-Roman top-1 test matches
  0.1568 Hindi / 0.2172 Bengali; reverse top-2 matches 0.0714 / 0.0545. These
  are baseline coverage/variant-matching observations, not semantic judgments.
  No held-out lexicon labels fitted the resource. Real IndicXlit is not run.
- Actual language-resource integration passes structural preservation,
  protection, bounded-view and complete-Roman invariants for 24 encoder-train
  queries per language. Roman views are available for 24/24 English, 22/24
  Hindi and 0/24 Bengali sampled queries; uncertain Bengali words abstain.
  This is a known quality limitation, not permission to force rewrites.
- Full extended regression: **380/380 tests pass under each of hash seeds
  0 and 917**, warnings as errors, offline cached real E5 enabled. Runtime
  33.969s / 41.185s. The initial sandbox-only run's two socket errors were
  resolved by running real localhost HTTP checks with explicit permission.
- Source compilation passes. Reports: `artifacts/checks/miracl-v1-integrity-2026-10-19.json`,
  `artifacts/checks/lid-public-v1-test.json`, and
  `artifacts/checks/query-resources-2026-10-19.json`.
- Independent Romanization audit passes all 150,000 original records and
  independently reconstructed body/title views, exact Latin/numeric/punctuation
  preservation, full provenance and all input/output/resource hashes. Report:
  `artifacts/checks/romanization-integrity-2026-10-19.json`.

### Stage 3 public-only checkpoint (19 October 2026)

- The user-approved scope uses only public original-query gold; the unavailable
  reviewed noisy/code-mixed challenge is explicitly omitted.
- A real encoder-training-only generation audit produced 1,652 English typo
  candidates (1,641 unreviewed, the rest rejected). No candidate was granted
  inherited judgments; the canonical outputs retain originals only. Native
  Hindi/Bengali inputs produced no eligible Latin typo candidates.
- Unreviewed audit candidates are not training or scoring inputs. Checkpoint:
  `artifacts/checks/public-only-stage3-2026-10-19.json`.
- The sealed pilot retains all selected whole groups in their existing splits:
  24 encoder/16 fusion/16 validation/24 final groups per language. Hindi has 17
  validation queries because one retained group has multiple originals. Every
  language corpus contains 1,500 passages and all selected judgments.
- Actual pretrained/original-only adapted encoder runs will use the same
  explicit 128-token right truncation. Planned training uses SGD, batch size 1,
  one epoch and seeds 13/29, with no unreviewed variants and consistency zero.
  These are CPU pilot settings, not a tuned full-data training recipe.

### Stage 4 CPU-pilot checkpoint (19 October 2026)

- Three real pinned pretrained-E5 indexes built and reloaded over the same
  1,500-passage language tracks, using 128-token right truncation and batch 8.
- Original-query validation retrieval and sealed ranking replay pass all
  source hashes, query/branch budgets, serving-rank reconstruction and
  corpus-membership checks. Final-development metrics remain untouched.
- Validation language-macro nDCG@10: BM25 **0.6498**, protected-Roman-view
  character **0.3823**, pretrained E5 **0.7792**, equal fusion **0.7406**,
  validation-tuned fixed fusion **0.7944**. These small reduced-corpus pilot
  numbers are not full-profile/full MIRACL or natural noisy-query results.
- All validation dense branches have one original view, so the B3 label is
  valid. Character retrieval is unavailable on every sampled Bengali query;
  those queries score zero for that branch, not silently excluded.
- Full updated suite: **449/449 tests pass under each of hash seeds 0 and
  917**, no skips, warnings as errors, offline actual cached E5, and real
  localhost HTTP tests. Times 34.820s / 42.495s.
- Small metadata/checkpoint-memory fixes preserve predictions and training
  objectives: query traces no longer repeat megabytes of LID provenance, and
  encoder parameter-delta diagnostics use bounded chunks and release gradients.
- The scoped training pack independently contains only 72 original encoder
  queries and their 739 judgments (169 positive, 570 grade zero) across the
  shared 4,500-passage fixed pilot corpus. No held-out query/group/judgment
  overlaps supervision. Actual adaptation uses SGD learning rate 0.001,
  zero weight decay and seeds 13/29, declared before pilot retrieval metrics.
- Checkpoint: `artifacts/checks/cpu-pilot-stage4-2026-10-19.json`.
  Reproducible experiment commands: `docs/PUBLIC_STUDY_RUN.md`.

### Stage 5 completed CPU-pilot checkpoint (19–20 October 2026)

- The real E5 seed-13 checkpoint independently passes exact saved-file
  membership/content verification. It records 72 original encoder-training
  queries/groups, 72 finite-gradient optimizer updates, zero variant pairs,
  original-only retrieval training and no applicable consistency ablation.
- Training-objective loss is 0.302813 before and 0.293597 after adaptation;
  parameter-change L2 is 0.0120812. All 72 realized negatives are judged
  nonrelevant passages. These are training diagnostics, not held-out gains.
- Seed 13 checkpoint SHA-256:
  `90b513efc923f63aff427cdeb711fa21e8d7874c31320343e0955a74b8c7d674`.
  The first seed-29 attempt was killed before saving a checkpoint; the kernel
  memory-kill counter confirms out-of-memory termination. Parallel resource
  checks were stopped. The isolated retry succeeded with the same protocol:
  72 finite-gradient updates, loss 0.292541 to 0.283928, parameter-change L2
  0.0113844, 72 judged-nonrelevant negatives and zero variant pairs.
- Both checkpoints' complete training-data manifests independently reproduce
  from the packed source inputs and their recorded configurations. Checkpoint
  membership/hashes and finite diagnostics pass. Seed 29 SHA-256:
  `62d72d92e0dd8b79f36150421050cc3ed9bcea6d8c044d5db54714eda4eb6aaa`.
  Audit: `artifacts/checks/cpu-pilot-encoder-training-2026-10-19.json`.
  All six trained dense indexes have completed. Independent streaming checks
  pass all 9,000 trained vectors: exact per-language 1,500-ID membership,
  384-dimensional finite float32 unit vectors, seals/preprocessing/checkpoint
  identity, and no frozen-vector reuse. Every vector differs across seeds and
  from the frozen counterpart; maximum norm error is below 3.1e-8. Audit:
  `artifacts/checks/cpu-pilot-adapted-index-audit-2026-10-20.json`.
  All 18 full/retrained feature-ablated gates and 27 matched validation studies
  are complete. Independent rank replay and validation-scoped judgment checks
  reproduce every metric and fixed-grid selection. Excluded columns, gate
  compatibility, split IDs and four-candidate selection all pass. Audit:
  `artifacts/checks/cpu-pilot-gate-validation-audit-2026-10-20.json`.
- Before final evaluation, seed 13/full gate is predeclared as primary and
  deployment choice; seed 29 is replication. Both will be reported without
  selecting a winner on final scores. The rule is in the protocol config.
- The full learned gates underperform validation-tuned fixed fusion and
  dense-only on validation. Macro nDCG@10: full gate 0.7463/0.7635 versus
  tuned fixed 0.7970/0.7960 and dense 0.7821/0.7813 (seeds 13/29).
  These negative comparisons are retained; no model is selected on final scores.
- Pre-final regression again passes **484/484 under both hash seeds** with
  no skips, warnings as errors, actual cached E5 and permitted localhost HTTP
  checks (81.191s/69.268s); compilation passes. Stage 5 checkpoint:
  `artifacts/checks/cpu-pilot-stage5-2026-10-20.json`.
- A 729-file content-sealed lock freezes code/configuration, exact data,
  resources, both checkpoints, all indexes/gates and validation selections:
  `artifacts/study/cpu-pilot/final-lock.json`. Final evaluation subsequently
  completed without refitting or retuning; every frozen file still matches.
- The new batched execution and reporting helpers pass 35 focused tests under
  hash seed 917 with warnings as errors. The expanded whole project passes
  **484/484 tests under each of hash seeds 0 and 917**, no skips, offline real
  E5 and permitted actual localhost HTTP checks (83.380s / 80.212s).
  Compilation and a fresh offline wheel build/install pass. Both the installed
  module CLI and console entry point pass explicitly fixture-only checks from
  outside the checkout. Record: `artifacts/checks/engineering-expanded-2026-10-19.json`.

### Stage 6 completed CPU-pilot checkpoint (20 October 2026)

- All 27 frozen/adapted/language/gate final studies complete, using exactly
  72 original public final query groups (24 per language). Final mode verifies
  every locked input before and after execution and performs no fitting,
  hyperparameter selection or winner-based deployment change.
- Independent standard-library formulas and rank replay reproduce all
  per-query nDCG@10/MRR@10/Recall@100 values, language means, missing-query
  counts and selection/provenance chains. Its audit streams 1,231 non-weight
  hashes; the final runner separately verifies the complete locked closure.
  Audit: `artifacts/checks/cpu-pilot-final-independent-audit-2026-10-20.json`.
- Final macro nDCG: frozen dense **0.7296**; adapted dense **0.7294/0.7284**;
  tuned fixed **0.7002/0.6996**; full gate **0.6701/0.6728**; equal fusion
  **0.6417/0.6437**, seeds 13/29. Adaptation did not improve dense final
  performance; the full gate beats equal but trails dense/fixed. Negative
  results and both seeds are retained without retuning or significance claims.
- Retrained uncertainty/agreement ablations, per-language group intervals,
  coverage/masking, shared retrieval costs, descriptive seed variation and
  ranking-error contrasts are reported. No noise/consistency/human-challenge
  result is substituted for missing reviewed data.
- Real seed-13/full-gate app checks pass English/Hindi/Bengali: six native and
  protected-query cases, 21 actual localhost HTTP requests and 12 search calls.
  Source passages, finite scores/contributions, normalized masked weights,
  original/protected text, API/HTML/health and 400/404 behavior pass. All test
  servers close; app inference uses no relevance labels. Browser visual QA is
  not claimed. Record: `artifacts/checks/cpu-pilot-real-app-2026-10-20.json`.
- Final whole-project regression passes **484/484 tests under both hash
  seeds 0 and 917**, no skips, warnings as errors, real cached E5/offline and
  permitted localhost HTTP checks (77.065s/68.273s). Compilation passes.
  A fresh wheel builds/installs offline; installed source identity/imports and
  three clearly labeled fixture CLI smokes pass outside the checkout.
  Record: `artifacts/checks/engineering-final-2026-10-20.json`.
- The full 729-file frozen closure still verifies after all tests. Completion
  checkpoint: `artifacts/checks/cpu-pilot-stage6-complete-2026-10-20.json`.
  Readable results: `docs/RESULTS.md`; full report and machine-readable results:
  `artifacts/study/cpu-pilot/final-report/`. Launch/reproduction instructions:
  `docs/PUBLIC_STUDY_RUN.md`.
- Full prepared-50k/full MIRACL experiments remain outside the executed pilot
  and need separate capacity work. All downloaded/prepared datasets and real
  checkpoints remain available; nothing was replaced by synthetic results.

### Stage 1 checkpoint (18 October 2026)

- Full extended suite: **231 tests under each of hash seeds 0 and 917;
  230 passed, one optional cached-pretrained-E5 test skipped, no failures**.
  Warnings were errors and both Hugging Face/Transformers offline modes were
  enabled. Real localhost HTTP checks ran with explicit socket permission.
- `compileall` passed for all modules and tests.
- An offline wheel built and installed without dependency resolution. From
  outside the checkout, both `python -m app --demo --query ...` and the
  installed `uncertainty-ir-demo` entry point returned correctly marked
  six-passage fixture results for Romanized Hindi and native Bengali queries.
- Read-only review found no implementation blocker. Visual browser QA was not
  available; server-rendered HTML, API, real HTTP, escaping, and access/input
  boundaries have automated coverage.
- Detailed checkpoint: `artifacts/checks/engineering-2026-10-18.json`.

## Reproducing the extended local checks

The isolated dependencies installed for verification are in ignored `.venv`
subdirectories (CPU torch 2.14.1, Transformers 4.57.6, safetensors 0.8.0).
From the project root, run:

```bash
PYTHONPATH="$PWD/.venv/neural-deps:$PWD/.venv/transformers-deps" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONWARNINGS=error \
  python3 -m unittest discover -s tests -v
```

Repeat with `PYTHONHASHSEED=917`. Alternatively install `.[training]` into an
active virtual environment and run the suite normally. Without optional neural
dependencies, the numerical/model integration tests explicitly skip.
