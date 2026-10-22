# Capacity and larger experiments

The completed data preparation preserves 50,000 passages per language. The
initial executable research study is a separate 1,500-passage-per-language
CPU pilot; it must not be reported as the full prepared-profile experiment.

The available machine has four CPU cores, approximately 3.7 GiB RAM and
1 GiB swap, with no detected GPU. Current in-memory Python index layouts are
the limiting factor for larger runs, not only encoder batch size.

## Evidence from the pilot

The frozen validation pipeline manifests contain these character indexes:

| Language | Vocabulary | Postings | Extrapolated 50k posting payload alone |
|---|---:|---:|---:|
| English | 226,236 | 2,187,145 | 5.98 GiB |
| Hindi | 190,118 | 1,692,777 | 4.62 GiB |
| Bengali | 178,431 | 1,569,801 | 4.29 GiB |

These are **linear extrapolations, not measured full-profile footprints**.
They assume the pilot's average postings per source passage and the measured
CPython shallow layout: a 56-byte pair tuple, a 24-byte weight float, and an
8-byte container reference. Vocabulary/IDF tables, dictionaries, document IDs,
and constructor counters are not included. The current constructor also
retains all document counters while constructing posting lists/final tuples,
so its peak is higher than the final posting payload.

Other simultaneous consumers include approximately 449 MiB of E5 float32
parameters, Python dense-vector objects (about 588 MiB projected at 50k,
versus 73 MiB of raw float32), BM25 postings, and the 460,066-record shared
lexicon with both lookup directions. These estimates must not be treated as
proof that a particular full-scale run has been attempted or failed.

Actual forward-only E5 measurements on a small public training sample were
about 13.9 passages/s at 128 tokens/batch 8/four threads, and 2.5 passages/s at
512 tokens/batch 4. Corpus-length variation and serialization affect full runs.
The pilot uses the same explicit 128-token truncation for all compared
encoders; shortening context is a methodological limitation, not a quality
improvement.

## Next capacity milestone

Before executing complete 50k-profile comparisons, either provide a
larger-memory execution environment, or implement and independently verify
bounded/two-pass character construction and packed sparse/dense storage.
Preserve float precision, scoring/rank order, corpus fingerprints, query
protection, source provenance, and gate/index compatibility in that work.
Profile one language before starting all larger runs. Never resolve a memory
problem by silently dropping judged passages or relabeling pilot scores.
