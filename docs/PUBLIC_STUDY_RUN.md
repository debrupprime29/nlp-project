# Reproduce the public-data CPU pilot

This is the execution recipe for the user-approved scope in
[`configs/public-data-study.json`](../configs/public-data-study.json), not a
claim that every command below has already completed. Recorded results and
checkpoint status belong in [`PROGRESS.md`](../PROGRESS.md). Source versions,
licenses, acquisition and preparation are documented in
[`REAL_DATA.md`](REAL_DATA.md).

The pilot uses 1,500 passages per language, not the prepared 50,000-passage
tracks or the full MIRACL corpora. It retains whole groups in their existing
splits: 24 encoder-training, 16 fusion-training, 16 validation and 24 locked
final groups per language. Hindi has 17 validation queries because one group
contains multiple original queries. Do not resplit, regenerate a query-limited
profile, or substitute another corpus between compared methods.

All scored queries and supervised pairs are public MIRACL originals. The
encoder adaptation below is **original-only retrieval training**, not a
consistency/noise-training experiment. No unreviewed variant inherits qrels;
the unavailable independent human challenge is omitted. Official development
is `final_evaluation`: do not run its retrieval or scoring commands until every
Phase 5 model, index, gate, ablation and selection decision is frozen.

## 1. Offline prerequisites and checks

Run from the project root in Bash. These paths must already exist and be
verified; this recipe does not install or download anything:

- `artifacts/real/miracl-v1-cpu-pilot/{en,hi,bn}/`: sealed `corpus.jsonl`,
  `queries.jsonl`, `qrels.jsonl` and profile/split manifests.
- `artifacts/real/miracl-v1-romanized/transliteration.jsonl`: one shared
  Dakshina-training plus explicitly algorithmic uroman resource.
- `artifacts/models/lid-public-v1.json`: the frozen public auxiliary LID model.
- `artifacts/hf-cache`: multilingual E5-small revision
  `fd1525a9fd15316a2d503bf26ab031a61d056e98`.
- `.venv/{neural-deps,transformers-deps,language-deps}`: the three local optional
  dependency directories used in this checkout.
- `artifacts/research/cpu-pilot-training/{corpus,queries,qrels}.jsonl` and
  `manifest.json`: the packed encoder-training-only inputs, containing 4,500
  fixed passages, 72 original queries, 739 qrels and zero variant pairs.

```bash
set -euo pipefail
export PYTHONPATH="$PWD/.venv/neural-deps:$PWD/.venv/transformers-deps:$PWD/.venv/language-deps"
export HF_HUB_CACHE="$PWD/artifacts/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=13

pilot=artifacts/real/miracl-v1-cpu-pilot
study=artifacts/study/cpu-pilot
lexicon=artifacts/real/miracl-v1-romanized/transliteration.jsonl
lid=artifacts/models/lid-public-v1.json
e5_revision=fd1525a9fd15316a2d503bf26ab031a61d056e98
```

Keep the following shell function available for each phase checkpoint. It
runs the whole regression suite twice, with numerical test thread counts
bounded independently of the four-thread study runs. The HTTP tests require
permission to bind a local loopback socket. A socket-permission failure is not
a passing test; obtain that permission and rerun. Optional neural/E5 skips
must be disclosed, not counted as real-model verification.

```bash
regression_check() {
  for test_hash_seed in 0 917; do
    PYTHONHASHSEED="$test_hash_seed" PYTHONWARNINGS=error \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 -m unittest discover -s tests -v
  done
  python3 -m compileall -q app.py data evaluation nlp retrieval training tests
}

regression_check
```

Use new output directories for a new experiment. The commands deliberately
omit `--force`, `--allow-download` and all fixture flags. Do not overwrite an
existing checkpoint/run or edit its manifest to bypass a compatibility error.
When reproducing an already completed run, verify its recorded artifacts or
choose a distinct study root and rerun the dependent validation selections.

## 2. Shared retrieval and analysis commands

These are shell-session convenience functions, not additional project CLIs.
They use the existing CLI flags explicitly. `collect_original_dense` creates
the separately manifested original-query dense baseline: the multi-view dense
branch in a fusion trace is not automatically that baseline.

```bash
collect_original_dense() {
  local track_dir="$1" language="$2" backend="$3" checkpoint="$4" split="$5"
  local encoder_args=(--backend "$backend" --device cpu --batch-size 8 --max-length 128)
  if [[ -n "$checkpoint" ]]; then
    encoder_args+=(--checkpoint "$checkpoint")
  fi
  python3 -m retrieval.dense search \
    --index-dir "$track_dir/dense" \
    --queries "$pilot/$language/queries.jsonl" --split "$split" \
    --output "$track_dir/$split.dense-original.run.jsonl" \
    --manifest "$track_dir/$split.dense-original.manifest.json" \
    --top-k 100 "${encoder_args[@]}"
}

collect_and_analyse() {
  local track_dir="$1" result_dir="$2" language="$3" backend="$4"
  local checkpoint="$5" split="$6" gate="${7:-}"
  local encoder_args=(--backend "$backend" --device cpu --batch-size 8 --max-length 128)
  local gate_args=() selection_args=()
  local analysis_dir="$result_dir/validation-analysis"
  if [[ -n "$checkpoint" ]]; then
    encoder_args+=(--checkpoint "$checkpoint")
  fi
  if [[ -n "$gate" ]]; then
    gate_args+=(--gate-model "$gate")
  fi
  if [[ "$split" == final_evaluation ]]; then
    selection_args+=(--selection "$result_dir/validation-analysis/validation-selection.json")
    analysis_dir="$result_dir/final-analysis"
  fi
  python3 -m retrieval.fusion \
    --corpus "$pilot/$language/corpus.jsonl" \
    --queries "$pilot/$language/queries.jsonl" --split "$split" \
    --lid-model "$lid" --lexicon "$lexicon" --dense-index "$track_dir/dense" \
    --output "$result_dir/$split.run.jsonl" \
    --manifest "$result_dir/$split.run.manifest.json" \
    --trace "$result_dir/$split.trace.jsonl" \
    --top-k 10 --per-branch-top-k 100 --per-view-top-k 100 --k-view 60 --k 60 \
    "${encoder_args[@]}" "${gate_args[@]}"
  python3 -m evaluation.study_runs \
    --trace "$result_dir/$split.trace.jsonl" \
    --run-manifest "$result_dir/$split.run.manifest.json" \
    --corpus "$pilot/$language/corpus.jsonl" \
    --queries "$pilot/$language/queries.jsonl" \
    --qrels "$pilot/$language/qrels.jsonl" --split "$split" \
    --dense-original-run "$track_dir/$split.dense-original.run.jsonl" \
    --dense-original-manifest "$track_dir/$split.dense-original.manifest.json" \
    --output-dir "$analysis_dir" --grid-resolution 4 --bootstrap-samples 1000 --seed 13 \
    "${selection_args[@]}"
}
```

Serving remains top-10, including during gate fitting and trace collection.
The study postprocessor reconstructs top-100 evaluation rankings from the
stored top-100 branch candidates without rerunning retrieval. Increasing the
serving top-k to 100 changes the frozen pipeline and breaks gate/selection
compatibility; it is not needed to evaluate Recall@100.

`validation-analysis/validation-selection.json` freezes the validation-only
fixed-weight grid winner. The grid metric is language-macro actual fused
nDCG@10, with deterministic tie-breaking. The directory must be fresh. Final
replay requires that sealed selection and the same pipeline, corpus, query
source, validation groups and fusion settings. Gate changes produce different
pipeline fingerprints, so full and ablated gates each need their own
validation analysis/selection.

The analysis publishes BM25 original, protected Roman-view character, separate
dense original, dense multi-view when distinct, equal fusion, selected fixed
fusion and the actual trace weights. For a gated trace, `trace_weights` is the
adaptive learned gate. `no_character_fixed` is only a renormalized fixed-weight
intervention, **not** a retrained-gate/full-system character ablation. B4
single-best normalization, noise training, consistency training and the human
challenge remain explicitly unavailable; do not relabel another output as one
of them. Latency/candidate costs belong to the actual shared retrieval trace,
not separately measured runtimes for every replayed baseline.

## 3. Phase 4: frozen encoder, validation only

Build each language index using four CPU threads, maximum length 128 and batch
size 8. Right truncation is intentional and shared by frozen/adapted systems;
these are not the full-spec 512-token settings.

```bash
for language in en hi bn; do
  python3 -m retrieval.dense build-index \
    --corpus "$pilot/$language/corpus.jsonl" \
    --index-dir "$study/frozen/$language/dense" \
    --backend e5 --model intfloat/multilingual-e5-small --revision "$e5_revision" \
    --device cpu --batch-size 8 --max-length 128
done

for language in en hi bn; do
  track_dir="$study/frozen/$language"
  collect_original_dense "$track_dir" "$language" e5 "" validation
  collect_and_analyse "$track_dir" "$track_dir" "$language" e5 "" validation
done

regression_check
```

Before Phase 5, verify all three index manifests report 1,500 documents,
384-dimensional real E5 vectors, `fixture_only: false`, the pinned revision,
and the intended 128-token preprocessing. Inspect each validation study's
query/group counts, missing-run counts, masked branches, coverage, candidate
costs and metrics. Preserve the selection files and input/output hashes. Do
not inspect final metrics to decide whether to proceed or change settings.

## 4. Phase 5a: original-only encoder adaptation and reindexing

The predeclared CPU protocol uses plain SGD, learning rate 0.001, no weight
decay, gradient clipping 1, temperature 0.05, one epoch and batch size 1. It is
a bounded adaptation pilot, not an Adam-equivalent tuned recipe. The requested
negative budget is one (`--random-negatives 1`, BM25 negatives zero); mining
prefers a judged nonrelevant passage when available. Inspect the realized
negative-source counts rather than calling every sampled negative random.
Batch size 1 supplies no cross-query in-batch negatives.

Run the two predeclared seeds sequentially. Do not pick a winning seed using
final evaluation. Any validation-based deployment choice or two-seed reporting
rule must be recorded before the final lock.

```bash
for encoder_seed in 13 29; do
  model_dir="$study/adapted-seed-$encoder_seed"
  python3 -m training.train_encoder \
    --corpus artifacts/research/cpu-pilot-training/corpus.jsonl \
    --queries artifacts/research/cpu-pilot-training/queries.jsonl \
    --qrels artifacts/research/cpu-pilot-training/qrels.jsonl \
    --output-dir "$model_dir/checkpoint" \
    --backend e5 --model intfloat/multilingual-e5-small --revision "$e5_revision" \
    --device cpu --epochs 1 --batch-size 1 --max-length 128 \
    --optimizer sgd --learning-rate 0.001 --weight-decay 0 --gradient-clip 1 \
    --temperature 0.05 --consistency-weight 0 \
    --random-negatives 1 --bm25-negatives 0 --seed "$encoder_seed"

  python3 - "$model_dir/checkpoint" <<'PY'
import math
import sys
from pathlib import Path
from training.checkpoints import verify_checkpoint

checkpoint = verify_checkpoint(Path(sys.argv[1]))
metadata = checkpoint["metadata"]
training = metadata["training"]
assert metadata["fixture_only"] is False
assert training["training_regime"] == "original_only_retrieval"
assert training["variant_pair_count"] == 0
assert training["consistency_applicable"] is False
assert training["lambda_zero_ablation"] is False
assert training["gradients_finite"] is True
assert training["optimizer_updates"] > 0
assert math.isfinite(training["initial_loss"]) and math.isfinite(training["final_loss"])
assert math.isfinite(training["parameter_delta_l2"]) and training["parameter_delta_l2"] > 0
print(checkpoint["checkpoint_sha256"], training["initial_loss"], training["final_loss"])
PY

  for language in en hi bn; do
    track_dir="$model_dir/$language"
    python3 -m retrieval.dense build-index \
      --corpus "$pilot/$language/corpus.jsonl" --index-dir "$track_dir/dense" \
      --backend e5-trained --checkpoint "$model_dir/checkpoint" \
      --device cpu --batch-size 8 --max-length 128
    collect_original_dense "$track_dir" "$language" e5-trained "$model_dir/checkpoint" validation
    collect_and_analyse "$track_dir" "$track_dir" "$language" \
      e5-trained "$model_dir/checkpoint" validation
  done
  regression_check
done
```

Checkpoint verification hashes the exact saved files. Index construction and
validation retrieval reload the saved checkpoint, exercising the real
checkpoint-to-index-to-query path. Every adapted index must bind that
checkpoint's content hash; never reuse frozen vectors with an adapted query
encoder. Retain finite-gradient/update diagnostics and training history. A
loss decrease is a training-objective observation, not evidence of held-out
retrieval improvement; unchanged/worse validation performance must be reported.

The verified Transformers 4.57.6 runtime emits a Mistral-regex warning when
loading these locally saved non-Mistral (`model_type: bert`, XLM-Roberta fast
tokenizer) checkpoints. Its default warning path does not patch tokenization.
Do not apply the suggested Mistral-specific rewrite to this tokenizer. An
actual saved-versus-pinned-original comparison passed for all 4,500 pilot
passages, 169 non-final queries and six additional special/protected-text
checks: token IDs, masks and other returned fields are identical at the same
128-token/padding settings. Both seeds' saved fast-tokenizer assets match.
Evidence: `artifacts/checks/e5-saved-tokenizer-equivalence-2026-10-19.json`.

## 5. Phase 5b: full gate and genuine feature ablations

Fit a separate gate for each language and encoder seed. Only `fusion_train`
examples fit the gate; validation chooses among temperature 0.25/0.5 and L2
0.001/0.01 using actual fused nDCG@10. Final labels are not fit or selected on.
The default gate learning rate is made explicit below; each fit has 300 epochs.

Both ablations are **retrained** with excluded columns disabled during fitting
and inference. `no_uncertainty` removes mean token entropy and uncertain-token
fraction; `no_agreement` removes all six pairwise branch-overlap features.
Neither means disabling whole branches or modifying a fitted gate afterward.

```bash
for encoder_seed in 13 29; do
  model_dir="$study/adapted-seed-$encoder_seed"
  for language in en hi bn; do
    track_dir="$model_dir/$language"
    for gate_name in full no_uncertainty no_agreement; do
      excluded_features=()
      if [[ "$gate_name" == no_uncertainty ]]; then
        excluded_features=(--exclude-features mean_token_entropy uncertain_token_fraction)
      elif [[ "$gate_name" == no_agreement ]]; then
        excluded_features=(--exclude-features \
          overlap_bm25_original_bm25_native \
          overlap_bm25_original_char_tfidf \
          overlap_bm25_original_dense \
          overlap_bm25_native_char_tfidf \
          overlap_bm25_native_dense \
          overlap_char_tfidf_dense)
      fi
      python3 -m training.train_gate \
        --corpus "$pilot/$language/corpus.jsonl" \
        --queries "$pilot/$language/queries.jsonl" --qrels "$pilot/$language/qrels.jsonl" \
        --lid-model "$lid" --lexicon "$lexicon" --dense-index "$track_dir/dense" \
        --backend e5-trained --checkpoint "$model_dir/checkpoint" \
        --device cpu --batch-size 8 --max-length 128 \
        --epochs 300 --learning-rate 0.05 --validation \
        --temperature-candidates 0.25 0.5 --l2-candidates 0.001 0.01 \
        --output "$track_dir/gates/$gate_name.json" \
        --report "$track_dir/gates/$gate_name.report.json" \
        "${excluded_features[@]}"
      collect_and_analyse "$track_dir" "$track_dir/$gate_name-gate" "$language" \
        e5-trained "$model_dir/checkpoint" validation "$track_dir/gates/$gate_name.json"
    done
  done
  regression_check
done
```

Check reports for correct fusion/validation groups, no overlapping held-out
groups, actual selected hyperparameters, the expected excluded-feature lists,
finite gate parameters, zero-utility exclusions and unavailable-branch masks.
The validation runs also test loading each gate against its exact frozen
pipeline. A gate from another language, checkpoint, index or resource version
must fail compatibility checks, not be silently substituted.

Before any final run, record a lock/checkpoint containing the approved scope,
pilot lineage, exact input/resource/checkpoint/index/gate hashes, both encoder
seeds, all fixed selections, feature-ablation definitions, hyperparameters,
128-token truncation, top-10 serving/top-100 evaluation settings, and the
predeclared reporting/deployment selection rule. Confirm the whole regression
suite passes and each validation analysis is complete. There is no implicit
"all frozen" CLI switch: this is an explicit review checkpoint. Do not execute
the next section while any of these decisions remain open.

## 6. Phase 6: locked final evaluation, then whole-project checks

Execute only after the preceding lock is complete. These commands reuse the
matching validation selections; they perform no final-set selection or
training. Each analysis directory is fresh and final outputs are retained even
if a result is negative. Do not rerun a changed configuration as the same
locked experiment after inspecting results.

```bash
for language in en hi bn; do
  track_dir="$study/frozen/$language"
  collect_original_dense "$track_dir" "$language" e5 "" final_evaluation
  collect_and_analyse "$track_dir" "$track_dir" "$language" e5 "" final_evaluation
done

for encoder_seed in 13 29; do
  model_dir="$study/adapted-seed-$encoder_seed"
  for language in en hi bn; do
    track_dir="$model_dir/$language"
    collect_original_dense "$track_dir" "$language" e5-trained "$model_dir/checkpoint" final_evaluation
    collect_and_analyse "$track_dir" "$track_dir" "$language" \
      e5-trained "$model_dir/checkpoint" final_evaluation
    for gate_name in full no_uncertainty no_agreement; do
      collect_and_analyse "$track_dir" "$track_dir/$gate_name-gate" "$language" \
        e5-trained "$model_dir/checkpoint" final_evaluation "$track_dir/gates/$gate_name.json"
    done
  done
done

regression_check
```

Read `final-analysis/study_manifest.json`, the per-method `*.evaluation.json`
and `*.run.jsonl` beside it. Report nDCG@10, MRR@10, Recall@100, missing-query
counts and information-need-group bootstrap intervals, including per-language
results and both seeds. Macro-average language metrics explicitly when
summarizing tracks; do not treat pooled monolingual labels as cross-lingual
judgments or count the two seeds as independent query samples. Retain group
membership and per-query values for any later paired analysis. Small pilot
samples imply wide uncertainty. Resource abstention/masking and actual measured
retrieval costs belong alongside quality results.

## 7. Serve the verified, validation-selected pipeline

Choose the deployment seed by the rule fixed before final evaluation, not by
final-set winners. This example uses seed 13 and Hindi; change the language to
`en` or `bn` and use that language's matching full gate/index. The app loads no
qrels. A one-query smoke check returns ranked passages with provenance, views,
weights and branch contributions; it does not generate an answer.

```bash
deployment_seed=13
language=hi
model_dir="$study/adapted-seed-$deployment_seed"
python3 app.py \
  --corpus "$pilot/$language/corpus.jsonl" \
  --lid-model "$lid" --lexicon "$lexicon" \
  --dense-index "$model_dir/$language/dense" \
  --backend e5-trained --checkpoint "$model_dir/checkpoint" \
  --gate-model "$model_dir/$language/gates/full.json" \
  --device cpu --max-length 128 --batch-size 8 \
  --query 'भारत की राजधानी क्या है?'
```

Repeat a native-language and protected-token/number query for each track, and
inspect returned document IDs, displayed source passages, finite scores,
masked weights and preserved protected spans. For the local UI, omit `--query`
and optionally specify `--host 127.0.0.1 --port 8000`. Separate per-language
servers need separate ports. On the current 3.7-GiB host, run one language
server at a time; different ports do not reduce model/index memory use.
Do not combine `--gate-model` with `--weights` or
change fusion/model settings after gate fitting. Browser visual review remains
separate from automated API/HTTP/escaping checks.

## Batched execution with enforced final locks

`evaluation.pilot_workflow` is an alternative to the repeated collection and
gate-fitting commands above. It reuses the same shared language resources and
one encoder at a time. It does not train encoders or construct dense indexes.
Use it only after both checkpoints and all six adapted indexes are complete.
For this route, omit the `collect_original_dense`/`collect_and_analyse` calls
from Phase 5a and the manual Phase 5b loop: the helper requires fresh adapted
validation/gate outputs and creates all of them itself. Do not mix the two
routes in the same output directories.

```bash
python3 -m evaluation.pilot_workflow \
  --mode gates --profile "$pilot" --study-root "$study" \
  --lid-model "$lid" --lexicon "$lexicon" \
  --seeds 13 29 --device cpu --max-length 128 --batch-size 8

regression_check
```

The helper fits each full/uncertainty-ablated/agreement-ablated gate on the
same fusion-training examples, selects only on validation, and publishes each
variant's matching validation trace and fixed-weight selection. It preserves
the standalone CLIs' candidate order, hyperparameters and ranking protocol.
Final relevance rows are not interpreted during gate fitting.

After reviewing those results, create `final-lock.json` containing a
`files_sha256` mapping from canonical absolute file paths to full SHA-256
digests. The `lock_sha256` is SHA-256 of canonical UTF-8 JSON for the other
fields (sorted keys, compact separators, no NaN). The required closure is
available without loading a model or relevance labels from
`evaluation.pilot_workflow.required_input_paths(...)`; it covers the profile,
resources, checkpoints, indexes, all gates, validation artifacts, relevant
source/configuration files and the pinned frozen model cache. Retain upstream
training/lineage manifests and runtime-version records in the lock as well.
The lock must also record the rule in `configs/public-data-study.json`:
seed 13/full gate is primary, seed 29 is replication, both are reported, and
final scores never select the deployment seed.

Only once the lock and whole-project checks pass:

```bash
python3 -m evaluation.pilot_workflow \
  --mode final --profile "$pilot" --study-root "$study" \
  --lid-model "$lid" --lexicon "$lexicon" \
  --seeds 13 29 --device cpu --max-length 128 --batch-size 8 \
  --final-lock "$study/final-lock.json"

python3 -m evaluation.report_pilot \
  --study-root "$study" --final-lock "$study/final-lock.json" \
  --scope-config configs/public-data-study.json \
  --output-dir "$study/final-report"

regression_check
```

Final mode verifies every advertised frozen file before and after execution,
requires the complete input closure and compatible validation selections, and
refuses to overwrite outputs. It creates `final.*`, `original-final.*` and
`final-analysis/` files rather than the standalone recipe's `final_evaluation.*`
names. The report checks all 27 model/language/gate tracks and creates sealed
`report.json` plus readable `report.md`; incomplete or inconsistent tracks
fail rather than disappearing from the summary. Neither CLI accepts fixture
models as benchmark evidence.

## Required qualifications in the final report

This is a clean-original-query, reduced-corpus CPU pilot. It does not establish
natural noisy/code-mixed robustness, consistency benefits, independently
human-reviewed challenge performance, full 50k/full MIRACL results, or
cross-lingual retrieval. E5-small has prior MIRACL-training exposure, so the
frozen baseline is not zero-shot MIRACL.

The shared Romanization baseline is Dakshina-training plus algorithmic uroman,
not a validated real IndicXlit run. Held-out attested-variant matching was low
(native-to-Roman top-1: 15.68% Hindi / 21.72% Bengali; reverse top-2: 7.14% /
5.45%). These are word-variant coverage results, not retrieval accuracy or
semantic verification. In the earlier structural check on 24 encoder-training
queries per language, Roman views were available for 24 English, 22 Hindi and
zero Bengali queries. Bengali uncertainty abstention must remain enabled; do
not force unavailable branches or infer robust Bengali normalization from
working end-to-end plumbing. The independent auxiliary LID test has no Bengali
gold, so it cannot validate contextual Bengali LID accuracy. Preserve these
limitations and any negative retrieval results in the final handoff.
