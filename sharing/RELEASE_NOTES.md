# Trained multilingual passage-search app: CPU pilot v1.0.0

This release supplies the actual adapted seed-13 E5 checkpoint, three selected
English/Hindi/Bengali collections (1,500 passages each), matching dense indexes,
full learned gates and auxiliary query resources. It is not the six-passage
synthetic demo. No retraining or upstream model download is required after the
explicit runtime/dependency installation.

Download both `cpu-pilot-runtime-v1.0.0.tar.gz` and its `.sha256` companion.
Use the source at tag `runtime-v1.0.0` and follow `sharing/README.md`.
The pinned real runtime targets CPython3.12/Linuxx86_64 or WSL2, CPU, with one
language server at a time. NativeWindows/macOS portability is not claimed.

The app returns ranked passages and provenance, not generated answers.
Recorded retrieval gains were not demonstrated: dense-only and validation-tuned
fixed fusion beat the full learned gate. This is an original-query reduced-corpus
CPU pilot, not full MIRACL or verified noisy/code-mixed robustness evidence.

Resource licenses are separate. Preserve bundled source cards, per-passage
Wikipedia attribution and notices; no blanket code license is newly granted.
SHA256 checksums provide byte integrity, not publisher authenticity.
