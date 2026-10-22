# Local release validation — 22 October 2026

Status: **prepared and tested locally; GitHub publication is a separate step**.
This record does not claim that the release/tag has already been uploaded.

## What was checked

- Built a whitelist-only runtime with 38 files, including 20 inference files
  and 18 source/license/provenance/attribution files.
- Archive: `cpu-pilot-runtime-v1.0.0.tar.gz`, 336,976,386 bytes (about 321 MiB).
  Expanded payload: 632,344,920 bytes.
- SHA256:
  `a2df8cb195713f6bcbc43397d2be57bd9422c1f8fe26243c5c616640dbb4507d`.
- All **31** new packaging/security/launcher unit tests pass. They cover
  deterministic output, checksums, membership, corrupt input, traversal/link
  attacks, expansion limits and refusal to overwrite existing files.
- Created a clean source copy in a different directory with spaces in its path.
  No original `artifacts/`, installed dependencies or pretrained HF cache were
  copied into it. Installed the pinned CPU dependencies into a new Python3.12
  virtual environment from the public package indexes; `pip check` passes.
- Safely extracted the release bundle into that clean copy. The real adapted
  checkpoint and matching gates work unchanged for English, Hindi and Bengali.
- Genuine model integration passes **six query cases, 18 search invocations
  and 24 actual localhost HTTP requests**. Checked API/HTML consistency, health,
  invalid-input400, missing-route404, original/protected text, finite ranked
  scores, source identity, masked normalized weights and server cleanup.
- Whole original regression suite in the clean CPU environment, hash seed0,
  warnings treated as errors: **484 run, 483 passed, one expected skip, no
  failures/errors**. The skipped check requires the separate *pretrained* E5
  cache, deliberately absent from the minimal inference bundle. The included
  *adapted* E5 checkpoint was exercised independently in the real-app checks.
- Compilation passes. All 729 original frozen research inputs and seven final
  research-result artifacts were verified unchanged after packaging changes.

Cold model/resource loading on the small-memory validation host took minutes;
the app prints verification/loading status before the localhost URL. Subsequent
queries reuse the loaded model. Run one language server at a time.

These are portability and engineering checks, **not relevance accuracy,
robustness, statistical superiority or a new scientific benchmark**. See the
existing results report for the pilot's negative quality findings. Native
Windows/macOS and browser visual review are not claimed tested.

Machine-readable local evidence (not included in the runtime payload):
`artifacts/sharing-validation/fresh-runtime-smoke.json` and
`artifacts/sharing-validation/fresh-regression-seed0.json`.
