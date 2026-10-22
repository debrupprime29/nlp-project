# Run the trained app on another computer

The real app needs its trained checkpoint, selected passage collections, indexes,
gates and language resources. A source-code clone alone does **not** contain them.
The publisher attaches `cpu-pilot-runtime-v1.0.0.tar.gz` and its `.sha256` file to
the GitHub Release `runtime-v1.0.0`. These commands become usable after that
release has actually been uploaded. No release URL or successful upload is
implied by this guide.

## Requirements

- Git, CPython **3.12**, and a browser. The pinned real-runtime dependency set
  targets **Linux x86_64 or Windows using WSL2**. Native Windows/macOS installs
  are not claimed tested. The small synthetic demo only needs Python 3.10+.
- GitHub CLI (`gh`) for the download commands, or download the two release assets
  in your browser. A private repository requires authorized GitHub access.
- CPU is sufficient; no GPU or retraining is needed. Allow several GB of disk
  space for dependencies, the compressed bundle and its extracted payload.
  Run one language server at a time on small-memory machines.
- Internet is needed explicitly for cloning, dependency installation and the
  release download. Once installed, model loading and search run offline.

## 1. Clone the matching release and install dependencies

Target repository: `debrupprime29/nlp-project`. These are Bash commands for
Linux/WSL2, not Python code. The release must be published before cloning its tag.

```bash
git clone --branch runtime-v1.0.0 https://github.com/debrupprime29/nlp-project.git nlp-project
cd nlp-project

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r sharing/requirements-runtime.txt
python -m pip install -r sharing/requirements-torch-cpu.txt
python -m pip check
```

The runtime is pinned to the versions used by the audited pilot. Do not use a
CUDA Torch build just to try this CPU app. Existing laptop-specific
`PYTHONPATH` settings are unnecessary in this clean virtual environment.

## 2. Download and verify the required model/data bundle

```bash
mkdir -p artifacts/releases
gh release download runtime-v1.0.0 --repo debrupprime29/nlp-project \
  --pattern 'cpu-pilot-runtime-v1.0.0.tar.gz*' --dir artifacts/releases

python -m sharing.runtime_bundle unpack \
  --archive artifacts/releases/cpu-pilot-runtime-v1.0.0.tar.gz \
  --sha256-file artifacts/releases/cpu-pilot-runtime-v1.0.0.tar.gz.sha256 \
  --destination artifacts/shared-runtime
```

For a browser download, place those same two files in `artifacts/releases/`,
then run only the unpack command. Download them from a trusted publisher;
checksums detect changed bytes but do not authenticate an untrusted publisher.
The installer rejects path traversal, links, unexpected files, corrupt payloads
and oversized archives, and refuses to overwrite an existing runtime directory.

## 3. Start the real trained app

Choose **one** command:

```bash
python -m sharing.run_app --language en
```

Hindi:

```bash
python -m sharing.run_app --language hi
```

Bengali (the code is `bn`, not `bengali`):

```bash
python -m sharing.run_app --language bn
```

Wait for `Local passage search: http://127.0.0.1:8000/`, then open that address
on the **same computer**. Initial checksum verification/model loading may take
several minutes on a small-memory machine. Use Ctrl+C to stop before switching languages. The collections are separate;
selecting Hindi does not combine or automatically route all three collections.
This is a local research server, not a public internet deployment.

Input is a nonblank question or phrase (at most 4,096 characters/256 tokens).
Output is up to ten ranked source passages, titles, language, scores, query
views and branch contributions—not a generated answer. Each collection contains
1,500 passages, so unfamiliar topics may return irrelevant nearest passages.
Scores are not confidence probabilities. Romanized/code-mixed quality is limited.

Fresh manual queries whose topics are present in the bundle:

| Language | Query |
|---|---|
| English | Which company introduced the 4004 microprocessor, and what was its original purpose? |
| Hindi | अंतरराष्ट्रीय न्यायालय में न्यायाधीश कितने वर्ष के लिए चुने जाते हैं? |
| Bengali | একই মৌলের সমস্থানিকগুলোর নিউট্রন সংখ্যায় কী পার্থক্য থাকে? |

These are illustrative manual checks, not newly reviewed retrieval gold.

## 4. Check health, API and command-line search

With the English server running, in another terminal:

```bash
curl http://127.0.0.1:8000/healthz
curl --get http://127.0.0.1:8000/api/search \
  --data-urlencode 'q=Where does visible light sit within the electromagnetic spectrum?'
```

Health must report `status: "ok"`, `document_count: 1500` and
`fixture_only: false`. Search returns JSON with `hits`, `views`,
`active_weights` and diagnostics. Active weights sum to one; masked branches
can be absent. Empty API queries return HTTP 400. Browser HTML and JSON represent
the same retriever.

To get one JSON result without starting a server (stop an existing server first):

```bash
python -m sharing.run_app --language hi \
  --query 'अंतरराष्ट्रीय न्यायालय में न्यायाधीश कितने वर्ष के लिए चुने जाते हैं?'
```

The launcher verifies all runtime files and matching source-code hashes before
loading. It uses the saved seed-13 checkpoint/full gates unchanged, CPU,
128-token truncation and ten returned hits, without downloads or training.
It hides only the proven Mistral-regex **false-positive** warning for this
hash-verified XLM-R tokenizer; it does not change tokenizer behavior. Do not set
`fix_mistral_regex=True` on this checkpoint and do not edit its sealed files.

## Tests and troubleshooting

```bash
python -m unittest sharing.test_runtime_bundle -v
python -m unittest discover -s tests -v
python -m compileall -q app.py sharing data evaluation nlp retrieval training tests
```

The shared app bundle deliberately omits the *separate pretrained* E5 cache and
research-only inputs, so the optional cached-pretrained-E5 unit test can skip on
a new clone. That is not a claim that every original research check ran. The
direct real-app smoke below checks the included adapted checkpoint instead:

```bash
python -m sharing.smoke_runtime --runtime-dir artifacts/shared-runtime
```

The smoke checks all three languages sequentially, real HTTP/HTML/API behavior,
ranked passages and diagnostics; it does not score novel queries for relevance.
If localhost binding is denied in a sandbox, run it in a normal local terminal
or grant loopback-socket permission; a denied socket is not a passing test.

If a port is occupied, stop the previous app or add `--port 8001`. For a checksum
or compatibility failure, use the matching code release and unmodified bundle;
do not bypass manifests or retrain against different settings. To reinstall,
choose a new `--destination` and matching `--runtime-dir`, rather than overwriting.

## What is and is not distributed

Included: one adapted E5 checkpoint, all three 1,500-passage collections,
their three dense indexes/full gates, auxiliary LID and transliteration lookup,
per-file hashes, source cards, license notices and passage attribution sidecar.
Training downloads, qrels, query splits, seed-29 replication, HF cache,
installed dependencies and experiment history are not needed for live search.

This is the **original-query CPU pilot**, not full MIRACL, the prepared 50k
experiment or a verified noisy/code-mixed challenge. The learned gate did not
outperform dense-only or tuned fixed fusion in the recorded study. See
[`../docs/RESULTS.md`](../docs/RESULTS.md) and the report PDF for actual results.
Third-party resources retain their separate terms in
[`notices/THIRD_PARTY_NOTICES.md`](notices/THIRD_PARTY_NOTICES.md).

## Publisher: prepare and upload (not run automatically)

From the original workspace with its verified artifacts:

```bash
python3 -m unittest sharing.test_runtime_bundle -v
python3 -m sharing.runtime_bundle build \
  --output artifacts/releases/cpu-pilot-runtime-v1.0.0.tar.gz
```

If the two asset files have already been built and verified, reuse them; the
exporter deliberately refuses to overwrite an existing archive/checksum pair.

Commit the source code, `sharing/`, docs and PDF to Git. Keep `artifacts/`,
`.venv/`, credentials and caches ignored. After pushing the matching commit to
`main`, authenticate GitHub CLI in your own terminal, then publish the two assets:

In your local Git checkout whose `origin` is `debrupprime29/nlp-project`, stage
the complete source explicitly (no model binaries):

```bash
git add .gitignore README.md PROJECT_SPEC.md PROGRESS.md pyproject.toml app.py \
  data/ nlp/ retrieval/ training/ evaluation/ tests/ examples/ configs/ docs/ \
  sharing/ reports/build_report.py reports/uncertainty_aware_retrieval_report.pdf
git diff --cached --stat
git commit -m 'Add portable trained app runtime and run guide'
git push origin main
```

Check `git remote -v` first. If this is not yet a Git checkout, initialize or
clone the target repository before these commands. If the remote contains work
you do not have locally, reconcile it rather than force-pushing over it.

```bash
gh auth login
gh release create runtime-v1.0.0 \
  artifacts/releases/cpu-pilot-runtime-v1.0.0.tar.gz \
  artifacts/releases/cpu-pilot-runtime-v1.0.0.tar.gz.sha256 \
  --repo debrupprime29/nlp-project --target main \
  --title 'Trained multilingual CPU-pilot app v1.0.0' \
  --notes-file sharing/RELEASE_NOTES.md
```

Models exceed GitHub's 100 MiB ordinary-Git file limit; use release assets instead
of `git add -f artifacts/`. This release does not choose a blanket open-source
license for the owner's code. Review separate upstream terms before wider or
commercial redistribution. If you change source or runtime semantics later,
create a new matching bundle/release instead of editing the old study artifacts.

The local portability/test evidence is summarized in [VALIDATION.md](VALIDATION.md).
