# Public-data execution

This is the real-data preparation track, distinct from the six-passage demo
and synthetic neural tests. Final retrieval evaluation remains locked. The
project does not download any resource implicitly.

## Sources and provenance

| Resource | Pinned version | Purpose and qualification |
|---|---|---|
| [MIRACL queries/qrels](https://huggingface.co/datasets/miracl/miracl) | `5be20db9509754dadad47689368639fcec739c00` | Original human queries and passage judgments; clean monolingual tracks, not reviewed noisy variants |
| [MIRACL corpus](https://huggingface.co/datasets/miracl/miracl-corpus) | `d921ec7e349ce0d28daf30b2da9da5ee698bef0d` | Original Wikipedia passages; retain the source IDs and underlying Wikipedia attribution |
| [Dakshina](https://github.com/google-research-datasets/dakshina) | v1.0, GCS generation `1590618303711280` | CC-BY-SA-4.0; attested word Romanizations, not query relevance or semantic review |
| [COMI-LINGUA](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) | `0214cb358e59e36da60e01d3d59ddd24df897749` | CC-BY-4.0; public human Hindi-English token annotations, not Bengali retrieval-query gold |
| [Multilingual E5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `fd1525a9fd15316a2d503bf26ab031a61d056e98` | MIT model; off-the-shelf baseline with prior MIRACL-training exposure, not zero-shot MIRACL |
| [uroman](https://github.com/isi-nlp/uroman) | `1.3.1.1` | Algorithmic out-of-vocabulary Romanization; bundled USC permissive license and publication acknowledgment apply |

The MIRACL cards declare Apache-2.0. Resource licenses remain separate from
the project code. Original cards, checksums, immutable revisions, and the
installed uroman license are retained locally. For publications acknowledge:
“This project uses the universal romanizer software 'uroman' written by Ulf
Hermjakob, USC Information Sciences Institute (2015-2020)”. See its bundled
license for the exact notice and reference.

The source locks are `configs/{miracl-v1,comi-lingua,e5-small,uroman}.sources.json`.
Acquisition verifies full sizes and publisher-provided SHA-256 or Git blob
hashes before publishing files. The MIRACL lock contains 83 files totaling
5,218,934,859 bytes, including all corpus shards for the three languages and
only the official train/development query and judgment files.

## Acquire explicitly, then work offline

Run the planner only when intentionally refreshing the pinned source lock:

```bash
python3 -m data.plan_miracl --allow-network --output configs/miracl-v1.sources.json
```

The current checked source lock should normally be reused:

```bash
python3 -m data.acquire_resources --lock configs/miracl-v1.sources.json \
  --output-dir artifacts/raw/miracl-v1 --max-bytes 5300000000 --allow-download
python3 -m data.acquire_resources --lock configs/comi-lingua.sources.json \
  --output-dir artifacts/raw/comi-lingua --max-bytes 150000000 --allow-download
python3 -m data.acquire_resources --lock configs/e5-small.sources.json \
  --output-dir artifacts/hf-cache --max-bytes 500000000 --allow-download
python3 -m data.fetch_dakshina \
  --output-dir artifacts/raw/dakshina-v1 --allow-download
python3 -m data.acquire_resources --lock configs/uroman.sources.json \
  --output-dir artifacts/raw/uroman --max-bytes 1000000 --allow-download
python3 -m pip install --no-index --no-deps --target .venv/language-deps \
  artifacts/raw/uroman/uroman-1.3.1.1-py3-none-any.whl
```

uroman also needs `regex>=2024.5.15`; the checked local Transformers dependency
directory already supplies it. Alternatively install the explicit
`.[language-resources]` extra into an environment. Never execute dataset code
or load an untrusted pickled model.

The Dakshina helper requests only six Hindi/Bengali lexicons with bounded HTTP
ranges from the immutable generation: 5,434,441 response-body bytes instead of
the full 2 GB tar. Each member is sealed with its locally computed SHA-256.
It does **not** verify the full archive's MD5 and records this explicitly.

To reverify cached MIRACL files without networking:

```bash
python3 -m data.acquire_resources --lock configs/miracl-v1.sources.json \
  --output-dir artifacts/raw/miracl-v1 --max-bytes 0 --verify-only
```

## Fixed benchmark and isolation

```bash
python3 -m data.import_miracl --source-dir artifacts/raw/miracl-v1 \
  --output-dir artifacts/real/miracl-v1-reduced50k \
  --lock configs/miracl-v1.sources.json --languages en hi bn \
  --target-size 50000 --seed 13
```

This scans every original corpus shard, audits unique IDs on disk, preserves
all judgments (including grade zero), includes all judged passages, and fills
the remaining 50,000-per-language budget by stable hash sampling. These are
**reduced-corpus experiments**, not full MIRACL benchmark scores. Evaluation
tracks are separate monolingual collections; pooling them does not create
cross-lingual relevance judgments.

Exact NFKC/casefold/whitespace duplicates are grouped within each language.
Development takes precedence over overlapping training queries. This is not
a claim to have detected all semantic paraphrases. Internal official-training
groups split approximately 70/15/15 into encoder/fusion/validation. Official
development is `final_evaluation`, reserved for the locked final assessment.
Its qrels are used earlier only to construct the fixed benchmark corpus.

Do not create a smaller query-limit profile and then mix its checkpoints with
the complete profile: selecting fewer queries causes a new split. A compute
pilot should filter within the existing complete split and retain that lineage.

## Public auxiliary language labels

```bash
python3 -m data.prepare_lid_resources \
  --comi-root artifacts/raw/comi-lingua \
  --source-lock configs/comi-lingua.sources.json \
  --dakshina-root artifacts/raw/dakshina-v1 \
  --output-dir artifacts/auxiliary-lid-v1
python3 -m nlp.language_id train \
  --examples artifacts/auxiliary-lid-v1/auxiliary_train.jsonl \
  --validation-examples artifacts/auxiliary-lid-v1/auxiliary_validation.jsonl \
  --output artifacts/models/lid-public-v1.json --epochs 180 --seed 13
python3 -m evaluation.evaluate_lid \
  --model artifacts/models/lid-public-v1.json \
  --examples artifacts/auxiliary-lid-v1/auxiliary_test.jsonl \
  --output artifacts/checks/lid-public-v1-test.json
```

Only two-of-three human annotation majorities are used; predicted tags are
ignored. Invalid indices, ambiguous alignments, and missing majorities are
rejected and counted. Full normalized sentence groups from official test are
quarantined before training-source grouping/sampling. The bounded training
set contains 6,000 public human token examples plus 2,000 explicitly weak
Bengali Dakshina-training word examples (weight 0.25); calibration and test
contain 2,000 human tokens each. These are label-balanced samples, not a
natural-distribution accuracy estimate.

Calibration uses a group-disjoint portion of the training source; official
test is never fitted or used for model/hyperparameter selection. Public test
contains no Bengali gold. Bengali contextual calibration and manually labeled
retrieval-query validation remain unverified, regardless of Hindi-English
test accuracy.

## Passage Romanization

Set the local optional dependency paths for this checkout:

```bash
export PYTHONPATH="$PWD/.venv/language-deps:$PWD/.venv/neural-deps:$PWD/.venv/transformers-deps"
export HF_HUB_CACHE="$PWD/artifacts/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m data.romanize_corpus \
  --corpus en=artifacts/real/miracl-v1-reduced50k/en/corpus.jsonl \
  --corpus hi=artifacts/real/miracl-v1-reduced50k/hi/corpus.jsonl \
  --corpus bn=artifacts/real/miracl-v1-reduced50k/bn/corpus.jsonl \
  --dakshina-dir artifacts/raw/dakshina-v1 \
  --output-dir artifacts/real/miracl-v1-romanized \
  --allow-uroman --evaluate-heldout
```

The importer seals remain unchanged. New corpus files preserve original text,
titles, IDs and other metadata, adding explicit body/title Romanization and
coverage provenance. A complete unavailable field stays empty; native text is
never silently indexed as a Roman character view. Latin/numeric spans are
preserved exactly. The new shared `transliteration.jsonl` contains Dakshina
**train only** pairs and clearly tagged algorithmic mappings from unlabeled
fixed-corpus vocabulary. Pass it to existing `--lexicon` APIs. Query protection
and uncertainty rules still apply; corpus and query decisions can differ for
protected negation.

This is an explicitly documented resource baseline, **not execution of
IndicXlit**. The real optional IndicXlit adapter has not been validated.
Dakshina dev/test word coverage and attested-variant matches are measured only
after freezing the resource. They are not semantic/retrieval-quality scores
and never authorize inheriting relevance labels for generated queries.

The completed run has complete body Romanizations for 49,289/50,000 English,
48,513/50,000 Hindi, and 41,725/50,000 Bengali passages. This is availability,
not correctness. Held-out native-to-Roman top-1 attested-variant test matching
is only 15.68% Hindi and 21.72% Bengali; reverse top-2 matching is 7.14% and
5.45%. Keep these negative/limited results in the report. They are not tuned
away after inspecting the test. A structural check on 24 encoder-training
queries per language produced Roman views for 24 English, 22 Hindi and zero
Bengali queries: Bengali uncertainty correctly disables unsafe rewriting.
This model/resource baseline does not establish robust Bengali normalization.

## Public review coverage and the next decision

Public data supplies genuine human annotations for specific purposes, but
does not automatically verify this project's generated query variants.

- MIRACL: human clean queries and passage judgments, not paired noisy or
  bilingual rewrites.
- Dakshina: human-attested word spellings, not query-level meaning review.
- COMI-LINGUA: human Hindi-English annotations, but no Bengali retrieval gold.
- [CMQA 2018](https://github.com/khyathiraghavi/code_switched_QA): human Hindi-English
  QA; many questions have no retrieval passage. An article-grounded subset
  could support a separately labeled external QA adaptation, not MIRACL qrels
  or Bengali challenge claims.
- [CMIR](https://cmir-iitbhu.github.io/cmir/dataset.html): access form remains
  required; no access or labels are fabricated and no form was submitted.

None of these verified public sources establishes the specified independently
written and second-speaker-reviewed Hindi-English **and** Bengali-English
challenge set or meaning checks for newly generated MIRACL rewrites. Unreviewed
candidates must remain excluded from inherited-qrel training/evaluation.
On 19 October 2026 the user explicitly approved the reduced-scope
public-data-only plan in `configs/public-data-study.json`. The missing human
challenge and unreviewed variant supervision/scoring are excluded, not
considered validated. Real encoder adaptation may use original MIRACL
query–passage supervision with consistency weight zero. Automatic views can
still be evaluated as inference transformations of original queries, but that
does not establish natural noisy/code-mixed robustness.

An initial CPU pilot retains existing split assignments and all selected
queries' judged passages, targeting 1,500 passages and 24 encoder/16 fusion/16
validation/24 locked-final groups per language. It is a distinct profile from
the prepared 50,000-passage tracks. Do not call pilot numbers full-profile
results. Larger runs remain a separate compute milestone.
