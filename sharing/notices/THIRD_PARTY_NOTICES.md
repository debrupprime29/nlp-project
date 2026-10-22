# Third-party notices for the CPU-pilot runtime bundle

This bundle is the original-query, public-data CPU pilot, not the full MIRACL
collection or a general-purpose question-answering model. It contains a locally
adapted checkpoint, 1,500 selected passages for each of English, Hindi and Bengali,
their dense indexes and learned full gates, the auxiliary language-ID model, and
the recorded transliteration resource. It excludes the training downloads,
training history, caches and installed dependencies. Dependencies installed
separately keep their own licenses.

These notices record upstream license declarations and provenance. They are not
a legal opinion or a representation that all possible redistribution, privacy,
trademark or other rights have been cleared. In particular, upstream dataset
metadata does not supersede the licenses of underlying Wikipedia content.

## Project code and separate resource licenses

No new license for the project's own code is granted by this document. The
existing project metadata states "Research use; dataset licenses apply
separately"; the project owner must choose any broader code license explicitly.
Do not apply an MIT, Apache or Creative Commons label to the entire repository
merely because an individual dependency or dataset uses that license.

The source cards and provenance files included in the bundle should be retained
with redistributions. If a publisher supplies additional copyright or NOTICE
material, retain that material too. Changed/derived components and their upstream
sources are identified below; no upstream endorsement is implied.

## 1. Multilingual E5-small and the adapted seed-13 checkpoint

- Upstream model: <https://huggingface.co/intfloat/multilingual-e5-small>
- Immutable revision: `fd1525a9fd15316a2d503bf26ab031a61d056e98`.
- Upstream model-card declaration: MIT. Preserve the original model card.
- Related upstream implementation and copyright notice:
  <https://github.com/microsoft/unilm/tree/master/e5> and
  <https://github.com/microsoft/unilm/blob/master/LICENSE>.
  The latter says `Copyright (c) Microsoft Corporation`; its MIT text is retained
  as `MIT-MICROSOFT.txt`. The pinned Hugging Face tree supplies the model-card MIT
  declaration, not a separate model-specific LICENSE file.
- Local change: original-query retrieval adaptation, seed 13, one epoch; the
  bundle's checkpoint is not the unchanged upstream model. The checkpoint's
  `encoder_manifest.json` records its training and source identity.
- E5's original training included MIRACL training data. This is not a zero-shot
  MIRACL experiment. The checkpoint supplies passage embeddings, not generated
  natural-language answers.

## 2. MIRACL passages, original supervision, and Wikipedia attribution

- Corpus: <https://huggingface.co/datasets/miracl/miracl-corpus>, revision
  `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`.
- Queries/judgments used for training and evaluation:
  <https://huggingface.co/datasets/miracl/miracl>, revision
  `5be20db9509754dadad47689368639fcec739c00`.
- Project and paper: <https://github.com/project-miracl/miracl> and
  <https://arxiv.org/abs/2210.09984>.
- Both retained MIRACL cards declare Apache-2.0; a copy is provided in
  `Apache-2.0.txt`. The runtime bundle does not need the original query/qrel
  files, but the training provenance is retained.
- The passages originate from Wikipedia. Keep the underlying Wikipedia text
  attribution and applicable share-alike notices separately from MIRACL's
  dataset-level declaration. Wikimedia's reuse terms are at
  <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content>.
  Wikipedia text is generally available under CC BY-SA, with version and any
  additional terms determined by the source; older dumps may carry CC BY-SA 3.0
  notices. Relevant licenses:
  <https://creativecommons.org/licenses/by-sa/3.0/> and
  <https://creativecommons.org/licenses/by-sa/4.0/>.
- Local changes: deterministic passage selection, additional language/provenance
  fields, and protected Romanized title/body fields. Original title/body text and
  MIRACL source passage identifiers are retained.
- Keep the bundle's per-passage attribution JSONL with the corpora and indexes.
  Each record should identify the local document ID, MIRACL source passage ID,
  language, original article title, and the article URL in that language's
  Wikipedia, e.g. `https://en.wikipedia.org/wiki/Intel_4004`. A URL based on the
  retained title points to the article and its author history; it is not a claim
  that a contemporary page is the exact historic dump revision. Retain the
  MIRACL card's original Wikipedia dump links for historic source context.
- If displaying or redistributing passages, preserve or expose the corresponding
  article attribution and license notices. This metadata is not a page-by-page
  audit of imported third-party text or additional attribution requirements.

## 3. Dakshina-derived transliteration material

- Dataset creators: the Google Research Dakshina project; credit its contributors
  and the authors identified by the retained source link.
- Source: <https://github.com/google-research-datasets/dakshina>.
- Version: Dakshina v1.0, archive generation `1590618303711280`.
- Pinned archive:
  <https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar?generation=1590618303711280>.
- License: CC BY-SA 4.0,
  <https://creativecommons.org/licenses/by-sa/4.0/>.
- Local changes: selected Hindi and Bengali TRAIN lexicon rows are normalized
  and converted into scored lookup records; algorithmic uroman records are added
  with distinct source labels. Held-out lexicons were not used to fit these
  lookup records. The derived lookup data and its Dakshina-derived contributions
  are distributed under CC BY-SA 4.0; this statement does not relicense the
  unrelated project code, pretrained E5 weights or uroman software.
- The corpora's Romanized fields use this resource. The auxiliary language-ID
  training also uses weak Bengali word examples from Dakshina TRAIN; retain this
  provenance together with the model. No blanket determination is made here
  about the copyright status or applicable license of learned model parameters.
- Acquisition used generation-pinned HTTPS byte ranges for selected archive
  members. The full archive MD5 was not verified. Member hashes and this
  qualification are recorded in `dakshina_manifest.json`.

## 4. COMI-LINGUA and the auxiliary language-ID model

- Dataset creators: Lingo Research Group, IIT Gandhinagar, and the contributors
  credited by the source card.
- Source: <https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA>.
- Immutable revision: `0214cb358e59e36da60e01d3d59ddd24df897749`.
- Upstream declaration: CC BY 4.0,
  <https://creativecommons.org/licenses/by/4.0/>.
- Local use: human Hindi-English token labels, normalized and split for the
  auxiliary language-ID model. Majority labels are used; weak Bengali examples
  come separately from Dakshina, not Bengali human COMI annotations. The bundle
  distributes the fitted model, not the full COMI training collection.
- Preserve both COMI and Dakshina provenance for the fitted model. The model's
  features and learned parameters are not newly claimed to have a single
  dataset's license solely because that dataset contributed training examples.

## 5. uroman software and algorithmic Romanization

- Source: <https://github.com/isi-nlp/uroman>.
- Recorded version: `1.3.1.1`.
- License: the upstream USC permissive license, reproduced in
  `UROMAN-LICENSE.txt`; do not replace it with a PyPI license classifier.
- Local use: deterministic algorithmic out-of-vocabulary Romanization, recorded
  as `uroman_algorithmic`, not a human-attested spelling or reviewed query.
- Required publication acknowledgment:

  "This project uses the universal romanizer software 'uroman' written by Ulf
  Hermjakob, USC Information Sciences Institute (2015-2020)".

- Upstream reference: Ulf Hermjakob, Jonathan May, and Kevin Knight. 2018.
  Out-of-the-box universal romanization tool uroman. In Proceedings of the 56th
  Annual Meeting of Association for Computational Linguistics, Demo Track.

## 6. Indexes, gates and bundle provenance

The three dense indexes derive from the included adapted checkpoint and the
corresponding selected corpora. The full gates and language-ID model are locally
fitted artifacts. Preserve source-resource notices with them; their inclusion
does not remove upstream obligations or grant a new blanket code/data license.

Acquisition locks, corpus/romanization/checkpoint manifests, source cards,
per-file checksums and the passage attribution sidecar provide provenance. Old
absolute training paths in preserved manifests are historical references, not
instructions to read those locations on another user's machine. Do not include
tokens, passwords, local credential files, cached HTTP authorization headers or
unrelated user files in a published release.

This notice is a practical record of the supplied upstream declarations and
local transformations, not a substitute for reviewing applicable upstream terms
before wider redistribution or commercial use.
