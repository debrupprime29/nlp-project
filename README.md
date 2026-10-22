# Uncertainty-Aware Multi-View Retrieval

## Try the app from a fresh clone

For the **actual trained English/Hindi/Bengali app**, follow
[the portable runtime guide](sharing/README.md). The code belongs in Git;
the required model/data bundle is a separate GitHub Release asset. A clone
alone does not contain trained weights or datasets, and the guide's release
download commands require the publisher to upload that release first.

For a tiny dependency-free **synthetic demo**, use Python 3.10+ and run
`python3 app.py --demo`, then open `http://127.0.0.1:8000/`. This is not the
trained model or a retrieval-quality benchmark.

This repository is an incremental implementation of the research system in
[`PROJECT_SPEC.md`](PROJECT_SPEC.md). The completed build slice follows the
requested source-layout order: modules 1, 2, 12, 6, 7, 8, 3, 4, 5, 11, 9, 10,
then 13.

The base implementation uses only the Python standard library so fixture tests
run offline. Large datasets and neural-model dependencies will remain explicit,
optional integration steps; the project never downloads data implicitly.

Public real-data preparation and its reproducible commands are documented in
[docs/REAL_DATA.md](docs/REAL_DATA.md). That track is separate from completed
module/fixture tests; see [PROGRESS.md](PROGRESS.md) for execution status and
remaining research requirements.
The separately labeled public-data CPU-pilot recipe is in
[docs/PUBLIC_STUDY_RUN.md](docs/PUBLIC_STUDY_RUN.md).
See [docs/CAPACITY.md](docs/CAPACITY.md) before attempting complete 50k-profile
experiments on a small-memory machine.

The public-original-query CPU pilot has completed locked final evaluation:
two real E5 adaptation seeds, six adapted indexes, 18 gates, and 27 final
model/language/gate tracks. See [the results summary](docs/RESULTS.md) and
[the complete report](artifacts/study/cpu-pilot/final-report/report.md).
The proposed gains were not demonstrated: adaptation did not improve dense
final nDCG, and the learned gate trails dense-only/tuned fixed fusion while
beating equal fusion. These are 1,500-passage pilot results, not full-50k or
noisy/human-challenge evidence. Engineering and app-check status remains in
[PROGRESS.md](PROGRESS.md).

## Module 1: prepare a fixed corpus

Passage input is MIRACL-like JSONL (`docid`, `title`, `text`). Qrels may be
JSONL or three/four-column TREC text.

```bash
python3 -m data.prepare_corpus \
  --passages en=/path/to/en-passages.jsonl \
  --qrels en=/path/to/en-qrels.tsv \
  --output-dir artifacts/en-corpus \
  --target-size 50000 \
  --seed 13 \
  --dataset-version miracl-pinned-version
```

Repeat `--passages` and `--qrels` for `hi` and `bn` to build a pooled corpus.
Pooled passage IDs are language-prefixed and qrels are rewritten through the
same saved map. All judged passages are retained, even when this requires
exceeding `--target-size`.

## Module 2: create leakage-safe query splits

```bash
python3 -m data.make_splits \
  --train-queries en=/path/to/en-train-queries.jsonl \
  --train-queries hi=/path/to/hi-train-queries.jsonl \
  --train-queries bn=/path/to/bn-train-queries.jsonl \
  --dev-queries en=/path/to/en-dev-queries.jsonl \
  --dev-queries hi=/path/to/hi-dev-queries.jsonl \
  --dev-queries bn=/path/to/bn-dev-queries.jsonl \
  --output-dir artifacts/query-splits \
  --seed 13 \
  --dataset-version miracl-pinned-version
```

The splitter assigns whole `group_id` units, stratifies official training data
by benchmark language, and reserves official development data as
`final_evaluation`. Variants use `parent_query_id`; closely related parents must
be given the same explicit `group_id`. Unreviewed variants receive no inherited
`judgment_query_id`.

## Module 12: evaluate a ranked run

```bash
python3 -m evaluation.evaluate \
  --queries artifacts/query-splits/queries.jsonl \
  --qrels artifacts/en-corpus/qrels.jsonl \
  --run artifacts/runs/bm25-validation.jsonl \
  --split validation \
  --output artifacts/evaluations/bm25-validation.json \
  --bootstrap-samples 1000 \
  --seed 13
```

The evaluator reports nDCG@10, MRR@10, Recall@100, language and variant
slices, clean-to-noisy drop, and parent-group bootstrap confidence intervals.
Use `--baseline-run` to compute a same-query normalization harm rate. Run files
are strict JSONL records with `query_id`, `doc_id`, `branch`, one-based `rank`,
and finite `score`.

## Module 6: run the BM25 baseline

```bash
python3 -m retrieval.bm25 \
  --corpus artifacts/en-corpus/corpus.jsonl \
  --queries artifacts/query-splits/queries.jsonl \
  --split validation \
  --output artifacts/runs/bm25-validation.jsonl \
  --top-k 100 \
  --k1 1.2 \
  --b 0.75 \
  --title-weight 2.0
```

The baseline uses Okapi BM25 with NFC/casefolded, script-aware tokens and a
weighted title field. It preserves negation and numbers and applies no stemming,
stopword removal, spelling correction, or forced rewriting. The same
`BM25Index` supports native-script views through its bounded `search_views` API.
This pure-Python backend is intended for fixtures and reduced corpora, not the
full 32.9M-passage English collection.

## Module 7: run character TF-IDF

```bash
python3 -m retrieval.char_tfidf \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/query-splits/queries.jsonl \
  --split validation \
  --output artifacts/runs/char-tfidf-validation.jsonl \
  --top-k 100 --min-n 3 --max-n 5
```

This branch indexes only explicit `romanized_text`. It uses NFC, case folding,
collapsed whitespace, raw character 3–5-grams, smoothed TF-IDF, and cosine
similarity. Empty `pending` Hindi/Bengali fields are reported as unavailable;
native passage text is never used as a hidden fallback.

## Module 8: frozen multilingual E5

Install the optional neural dependencies with `python3 -m pip install '.[dense]'`.
Model downloads are explicit; commands use the local cache unless
`--allow-download` is supplied.

```bash
python3 -m retrieval.dense build-index \
  --corpus artifacts/corpus/corpus.jsonl \
  --index-dir artifacts/indexes/e5 \
  --device cpu --allow-download

python3 -m retrieval.dense search \
  --index-dir artifacts/indexes/e5 \
  --queries artifacts/query-splits/queries.jsonl \
  --split validation \
  --output artifacts/runs/e5-validation.jsonl
```

The default checkpoint revision is pinned. Inputs use E5's `query: ` and
`passage: ` prefixes, attention-mask mean pooling, L2 normalization, and
inner-product ranking. A passage title precedes its body on a separate line;
each original passage has one vector and is truncated from the right at 512
tokens. No chunking is used. Saved indexes check model/tokenizer revision,
dimension, preprocessing, and vector checksums before search. Float32 vector
values are stored in JSONL; exhaustive search is intended for reduced corpora.

This is the frozen B3 baseline; encoder fine-tuning belongs to Module 9. E5's
training included MIRACL training data, so results must be described as an
off-the-shelf pretrained baseline. See the
[official model card](https://huggingface.co/intfloat/multilingual-e5-small).

Offline CLI smoke tests explicitly use `--backend fixture-hash
--allow-fixture-encoder` on both commands. That backend is lexical, marks its
artifacts `fixture_only`, and uses the run label `dense_fixture_hash`. It does
not measure E5 semantic quality.

## Module 3: generate and review variants

Generate variants from the saved split assignments:

```bash
python3 -m data.generate_variants \
  --queries artifacts/query-splits/queries.jsonl \
  --split encoder_train \
  --output-dir artifacts/variants --seed 13
```

Automatic generation makes deterministic Latin-script typos while preserving
protected names, numbers, dates, URLs, and negation. Supplied full-query
Romanized, code-mixed, mixed-script, or human-written proposals can be loaded
with `--proposals proposals.jsonl`. Every candidate retains its parent's
group, source partition, and split. Synthetic proposals are unreviewed by
default; the generator does not infer semantic equivalence.

The output contains `queries.jsonl` (existing queries plus accepted verified
variants), `candidates.jsonl` (the audit), `review_queue.jsonl`, and a manifest.
A review file binds each decision to the candidate ID, parent ID, and candidate
SHA-256; pass it with `--reviews reviews.jsonl` on a subsequent run. Only an
explicit meaning-preserving verification can grant inherited judgments.
Human-written items require a reviewer distinct from the author. Protection
checks conservatively require exact preservation of protected text, including
negation; cross-script changes to those spans need separately designed and
validated preservation support.

A proposal record has this shape (the parent must exist in the split file):

```json
{"parent_query_id":"q-hi","text":"bhukamp kyu aate hain","variant_type":"romanized","languages":["hi"],"provenance":{"source":"human-supplied-romanization","revision":"v1"}}
```

After checking the candidate against its original, write an explicit decision
using the candidate's saved values:

```json
{"candidate_id":"COPY_FROM_QUEUE","candidate_sha256":"COPY_FROM_QUEUE","parent_query_id":"q-hi","reviewer_id":"reviewer-2","decision":"verified","meaning_preserved":true}
```

Keep the same input, proposal files, seed, and generation settings when applying
reviews. A changed candidate invalidates its earlier approval. Candidate and
review artifacts are local data; the tool does not contact reviewers.

## Module 4: train token language identification

```bash
python3 -m nlp.language_id train \
  --examples examples/tiny/lid_train.jsonl \
  --validation-examples examples/tiny/lid_validation.jsonl \
  --output artifacts/lid/tiny.json --seed 13

python3 -m nlp.language_id predict \
  --model artifacts/lid/tiny.json --text 'earthquake kyu aate hain'
```

The classifier uses standard-library multinomial logistic regression with
character 2–5-grams, neighboring tokens, script, and shape features. Output
includes probabilities, entropy, and an uncertainty flag for each token; low
confidence or a small label margin causes abstention. All span offsets refer to
the NFC processing text. Omitting `--model` selects a conservative script
evidence baseline. Roman script alone does not identify English.

Label examples use `text`, token `start`/`end`, `label` (`en`, `hi`, `bn`,
`other`), `partition`, `group_id`, and `source`. Fit only on
`auxiliary_train`; optional temperature calibration uses disjoint
`auxiliary_validation` groups. The tool rejects final-evaluation labels and
duplicate training/validation text spans. Saved models include fingerprints,
feature vocabulary, configuration, and calibration provenance. The committed
tiny examples exercise training and inference mechanics; they do not establish
language-identification quality on natural code-mixed queries.

## Module 5: build protected query views

```bash
python3 -m nlp.query_views \
  --text 'earthquake kyu aate hain 2026' \
  --lid-model artifacts/lid/tiny.json \
  --backend lexicon --lexicon examples/tiny/xlit_lexicon.jsonl \
  --output artifacts/views/example.jsonl
```

Each record preserves the exact original input and a separate NFC processing
text. It includes token decisions, protections, edits, at most two distinct
native hypotheses, an optional complete Roman view, and diagnostic counts.
Confident Hindi/Bengali Roman tokens can be transliterated; English tokens,
uncertain tokens, names, numbers, dates, negation, and contacts retain their
original text. Native views identical to the original are omitted. A Roman view
is unavailable if any remaining native text cannot be converted safely.

The supplied lexicon contains a few manual fixture pairs. Real applications
must supply a versioned lexicon with appropriate coverage, or explicitly enable
the optional IndicXlit backend with `--backend indicxlit
--allow-model-download` after installing its dependencies. The upstream engine
may download model assets when initialized; this flag authorizes that behavior.
Its beam scores remain uncalibrated. See the
[official IndicXlit interface](https://github.com/AI4Bharat/IndicXlit/blob/master/app/README.md).

For batch output, use `--queries queries.jsonl --split validation` in place of
`--text`. The original/native views can be passed to the existing BM25/dense
`search_views` APIs, and the Roman view to character TF-IDF. Query Romanization
does not populate the passage corpus: Hindi/Bengali character retrieval still
requires compatible, explicitly supplied passage `romanized_text`.

## Module 11: fuse the retrieval branches

```bash
python3 -m retrieval.fusion \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/query-splits/queries.jsonl --split validation \
  --lid-model artifacts/lid/model.json --lexicon /path/to/transliteration.jsonl \
  --dense-index artifacts/indexes/e5 \
  --output artifacts/runs/fusion-validation.jsonl \
  --trace artifacts/runs/fusion-validation.trace.jsonl
```

The pipeline searches original BM25, distinct native-view BM25 using the same
index, available Romanized character TF-IDF, and dense original/native views.
It averages reciprocal ranks within a multi-view branch, then applies weighted
reciprocal rank fusion across branches. Raw retrieval scores are not added.
Each branch retains at most 100 candidates (union at most 400); the default
final result depth is 10. Empty branches are masked and their weights are
renormalized away. Without a dense index, that branch is unavailable.

Equal weights are the default. Explicit weights must name all four branches;
zero explicitly disables a branch's ranking contribution, while the searches
still run for diagnostics. For example:

```bash
--weights '{"bm25_original":1,"bm25_native":1,"char_tfidf":1,"dense":2}'
```

Weights may also come from a JSON file. A programmatic inference-only weight
provider can consume observable query/view and ranking features. Module 10
fits and persists that learned provider. Per-query traces include
views, branch ranks, active weights, contributions, actual search counts,
intermediate candidates, and latency including view generation. Fixture dense
indexes require an explicit fixture run label and produce fixture manifests.

## Module 9: train the shared encoder

Install the optional neural dependencies with
`python3 -m pip install '.[training]'`. CPU-only PyTorch installation options
are listed in the [official installation guide](https://pytorch.org/get-started/locally/).

```bash
python3 -m training.train_encoder \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/variants/queries.jsonl \
  --qrels artifacts/corpus/qrels.jsonl \
  --output-dir artifacts/checkpoints/e5-trained \
  --epochs 1 --batch-size 4 --max-length 512 \
  --learning-rate 0.00002 --temperature 0.05 --consistency-weight 0.1 \
  --random-negatives 1 --bm25-negatives 1 --seed 13 \
  --device cpu --allow-download
```

Only `encoder_train` queries and their training qrels supply supervision.
Verified variants must retain valid content-bound reviews. All known positives
enter the multi-positive numerator, including other known positives appearing
in a batch. Judged nonrelevant negatives are preferred; remaining random,
in-batch, and BM25 candidates are explicitly recorded as potentially false
negatives. Mining uses training queries only. Query/document prefixes, masked
mean pooling, L2 normalization, and truncation match dense inference.

Training uses original-query retrieval plus reviewed-variant retrieval and
weighted cosine consistency. `--consistency-weight 0` removes only consistency,
retaining variant retrieval for the ablation. Without reviewed variants, the
run records zero variant pairs and trains original retrieval only. Checkpoint
metadata records losses, finite gradients, optimizer steps, parameter changes,
data/config fingerprints, and the initialization revision.

Saved checkpoints use safetensors and a manifest sealing all model/tokenizer
files. Trained dense indexes are tied to that checkpoint's content hash; a
different encoder checkpoint requires passage re-encoding. Build a trained
index with the same maximum length used for training:

```bash
python3 -m retrieval.dense build-index \
  --corpus artifacts/corpus/corpus.jsonl --index-dir artifacts/indexes/e5-trained \
  --backend e5-trained --checkpoint artifacts/checkpoints/e5-trained \
  --max-length 512 --device cpu

python3 -m retrieval.fusion \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/query-splits/queries.jsonl --split validation \
  --dense-index artifacts/indexes/e5-trained \
  --backend e5-trained --checkpoint artifacts/checkpoints/e5-trained \
  --max-length 512 --output artifacts/runs/fusion-trained-validation.jsonl
```

The explicit `--backend fixture-torch --allow-fixture-encoder` training mode
optimizes a small neural embedding model with real autograd. It is used for
offline convergence/checkpoint/integration checks and marks artifacts as
`fixture_only`. Loading that checkpoint through `e5-trained` also requires
`--allow-fixture-encoder`. Tests additionally exercise actual Transformers
pooling, gradients, safetensors, and local reload with a small randomly
initialized XLM-R architecture. These checks establish mechanics, not pretrained
E5 retrieval quality or a real-data research result.

## Module 10: train adaptive fusion

Freeze the encoder and its compatible indexes before fitting the gate:

```bash
python3 -m training.train_gate \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/variants/queries.jsonl \
  --qrels artifacts/corpus/qrels.jsonl \
  --lid-model artifacts/lid/model.json --lexicon /path/to/transliteration.jsonl \
  --dense-index artifacts/indexes/e5-trained \
  --backend e5-trained --checkpoint artifacts/checkpoints/e5-trained \
  --output artifacts/gate/model.json --validation

python3 -m retrieval.fusion \
  --corpus artifacts/corpus/corpus.jsonl \
  --queries artifacts/variants/queries.jsonl --split validation \
  --lid-model artifacts/lid/model.json --lexicon /path/to/transliteration.jsonl \
  --dense-index artifacts/indexes/e5-trained \
  --backend e5-trained --checkpoint artifacts/checkpoints/e5-trained \
  --gate-model artifacts/gate/model.json \
  --output artifacts/runs/fusion-adaptive-validation.jsonl \
  --trace artifacts/runs/fusion-adaptive-validation.trace.jsonl
```

Use the same retrieval/view settings and encoder maximum length at training
and inference. Only `fusion_train` queries supply gate supervision; verified
variants require the same content-bound reviews as encoder training. Observable
features describe script composition, token uncertainty, query structure,
rewrites, protected rewrite attempts, branch overlap, and branch-score shape.
Branch-score features use the within-branch averaged reciprocal-rank scores,
not an addition of incomparable raw retriever scores.

The standard-library linear softmax model learns cross-entropy against
temperature-softmax branch nDCG@10 targets. Scaling is fitted only to retained
fusion-training observations. Empty branches are masked, and queries without
positive judgments or with all-zero branch utility are excluded and counted.
With `--validation`, the report compares actual fused nDCG@10 against equal
weights and validation-tuned fixed weights, reporting per-language,
language-macro, and query-mean scores. Fixed-weight selection and optional
`--temperature-candidates 0.1 0.25 0.5 --l2-candidates 0 0.001` selection use
language-macro fused nDCG@10 on validation only. The default report path is
`model.report.json` alongside `model.json`. Final-evaluation labels are never
used for fitting or tuning. Saved JSON artifacts seal the model, feature schema,
and frozen pipeline identity; changed indexes or NLP resources require a new
gate. Search uses no qrels. `--gate-model` and `--weights` are mutually exclusive.

## Module 13: local search demonstration

Start the fully offline, explicitly synthetic demonstration:

```bash
python3 app.py --demo
```

Open `http://127.0.0.1:8000` in your browser. The page accepts English, Hindi,
Bengali, Romanized, and mixed-script queries and displays the actual pipeline's
ranked passages, source/language, original/native/Roman views, active weights,
and per-branch contributions. Demo data, language labels, lexicon, and hash
embeddings are small fixtures, not pretrained E5 or benchmark results.

For a single JSON search without starting a server:

```bash
python3 app.py --demo --query 'bhukamp kyu aate hain'
```

Use your prepared corpus and frozen model artifacts for the real pipeline:

```bash
python3 app.py \
  --corpus artifacts/corpus/corpus.jsonl \
  --lid-model artifacts/lid/model.json --lexicon /path/to/transliteration.jsonl \
  --dense-index artifacts/indexes/e5-trained \
  --backend e5-trained --checkpoint artifacts/checkpoints/e5-trained \
  --gate-model artifacts/gate/model.json
```

The gate must match the exact frozen pipeline settings/resources used to train
it. Omit `--gate-model` for equal fusion, or supply `--weights` for explicit
fixed fusion. Omit `--dense-index` for a lexical-only demonstration; unavailable
branches are masked rather than simulated. No qrels are loaded for live search.
Pretrained models remain local-cache-only unless `--allow-download` is given.

The application is a loopback-only research demo, not a public production
service. It provides `/api/search?q=...` JSON search and `/healthz`, uses bounded
query input and escaped server-rendered HTML, and does not expose local files.
After installation, `uncertainty-ir-demo --demo` is equivalent to the source
command. No hosted deployment is created by these commands.

## Tiny offline example

The committed `examples/tiny` files are fixture data, not experimental results.
They allow the corpus, split, retrieval, and evaluation CLIs to be exercised
without a download:

```bash
python3 -m data.prepare_corpus \
  --passages en=examples/tiny/en_passages.jsonl \
  --passages hi=examples/tiny/hi_passages.jsonl \
  --passages bn=examples/tiny/bn_passages.jsonl \
  --qrels en=examples/tiny/en_qrels.jsonl \
  --qrels hi=examples/tiny/hi_qrels.jsonl \
  --qrels bn=examples/tiny/bn_qrels.jsonl \
  --output-dir /tmp/uir-corpus --target-size 2 --dataset-version tiny-fixture

python3 -m data.make_splits \
  --dev-queries en=examples/tiny/en_dev_queries.jsonl \
  --dev-queries hi=examples/tiny/hi_dev_queries.jsonl \
  --dev-queries bn=examples/tiny/bn_dev_queries.jsonl \
  --output-dir /tmp/uir-splits --dataset-version tiny-fixture

python3 -m retrieval.bm25 \
  --corpus /tmp/uir-corpus/corpus.jsonl \
  --queries /tmp/uir-splits/queries.jsonl \
  --split final_evaluation \
  --output /tmp/uir-bm25.jsonl

python3 -m evaluation.evaluate \
  --queries /tmp/uir-splits/queries.jsonl \
  --qrels /tmp/uir-corpus/qrels.jsonl \
  --run /tmp/uir-bm25.jsonl \
  --split final_evaluation \
  --output /tmp/uir-evaluation.json --bootstrap-samples 100
```

## Offline checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q app.py data evaluation retrieval nlp training tests
```

Install the `training` extra to execute numerical training and checkpoint tests;
without those dependencies, the relevant tests skip explicitly. Tests never
download pretrained weights. A cached pinned-E5 check skips if its checkpoint
is unavailable.

The checkpoint below is historical. The later public-data CPU pilot has now
completed; see [docs/RESULTS.md](docs/RESULTS.md) for its actual results and
[sharing/README.md](sharing/README.md) for the portable trained-app setup.

The 18 October 2026 engineering checkpoint ran 231 tests under each of hash
seeds 0 and 917: 230 passed and the optional uncached-pretrained-E5 check
skipped in each run. Both runs included actual neural fixture training and
real localhost HTTP tests, with warnings treated as errors and model downloads
disabled. The HTTP checks require loopback-socket permission in restricted
execution environments. Compilation, offline wheel build/install, and the
installed demo outside the source checkout also passed. This is software
verification, not a real-data retrieval result.
