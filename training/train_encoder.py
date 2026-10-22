"""Train a shared E5 bi-encoder on partition-bound, reviewed query pairs.

Neural packages are optional and imported only when training is requested.
The explicit fixture backend trains a small embedding model with real autograd;
it verifies mechanics offline and does not claim multilingual E5 quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from data.generate_variants import VariantError, generate_variants
from data.make_splits import ENCODER_TRAIN, OFFICIAL_TRAIN
from data.prepare_corpus import CorpusPreparationError, Qrel, _canonical_json_bytes, _clean_id, iter_jsonl, load_qrels
from retrieval.bm25 import BM25Document, BM25Error, BM25Index
from retrieval.dense import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DenseConfig, DenseDocument, DenseError, MODEL_MAX_LENGTH, PASSAGE_PREFIX, QUERY_PREFIX, format_passage, format_query, load_corpus


class EncoderTrainingError(ValueError):
    """Invalid training data, configuration, dependency, or neural output."""


PARAMETER_DELTA_CHUNK_ELEMENTS = 1_000_000


@dataclass(frozen=True)
class EncoderTrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 2e-5
    temperature: float = 0.05
    consistency_weight: float = 0.1
    random_negatives: int = 1
    bm25_negatives: int = 0
    in_batch_negatives: bool = True
    seed: int = 13
    max_length: int = MODEL_MAX_LENGTH
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EncoderTrainingError(f"{name} must be a positive integer")
        if self.max_length > MODEL_MAX_LENGTH:
            raise EncoderTrainingError("max_length cannot exceed the 512-token E5 limit")
        for name in ("random_negatives", "bm25_negatives"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EncoderTrainingError(f"{name} must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EncoderTrainingError("seed must be an integer")
        if not isinstance(self.in_batch_negatives, bool):
            raise EncoderTrainingError("in_batch_negatives must be a boolean")
        for name in ("learning_rate", "temperature", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise EncoderTrainingError(f"{name} must be finite and positive")
        for name in ("consistency_weight", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise EncoderTrainingError(f"{name} must be finite and nonnegative")
        if self.optimizer not in {"adamw", "sgd"}:
            raise EncoderTrainingError("optimizer must be adamw or sgd")


@dataclass(frozen=True)
class EncoderTrainingExample:
    original_query_id: str
    judgment_query_id: str
    group_id: str
    original_text: str
    variant_query_id: str | None
    variant_text: str | None
    positive_doc_ids: tuple[str, ...]
    negative_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncoderTrainingData:
    examples: tuple[EncoderTrainingExample, ...]
    documents: tuple[DenseDocument, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class EncoderTrainingResult:
    history: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


class TrainableEncoder(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...
    def parameters(self) -> Iterable[Any]: ...
    def train(self) -> Any: ...
    def eval(self) -> Any: ...
    def encode_training(self, texts: Sequence[str], *, max_length: int) -> Any: ...
    def save_pretrained(self, directory: Path) -> None: ...


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise EncoderTrainingError("Training records must be objects")
    return dict(value)


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EncoderTrainingError("Training metadata must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _priority(seed: int, query_id: str, doc_id: str) -> str:
    return _digest([seed, query_id, doc_id])


def prepare_encoder_training(
    records: Sequence[Mapping[str, Any]],
    qrels: Sequence[Qrel | Mapping[str, Any]],
    documents: Sequence[DenseDocument | Mapping[str, Any]],
    config: EncoderTrainingConfig = EncoderTrainingConfig(),
) -> EncoderTrainingData:
    """Prepare examples without consulting held-out relevance labels.

    Complete split artifacts and qrel files are accepted. Only encoder-training
    judgment IDs are read for supervision or mining; other qrels are counted and
    ignored before validating their document IDs or relevance values.
    """

    config.validate()
    inputs = [_mapping(value) for value in records]
    selected = [record for record in inputs if record.get("split") == ENCODER_TRAIN]
    if not selected:
        raise EncoderTrainingError("No encoder_train queries were supplied")
    for record in selected:
        if record.get("source_partition") != OFFICIAL_TRAIN:
            raise EncoderTrainingError("Encoder training requires official_train queries")
        if record.get("variant_type", "original") != "original" and record.get("verified") is True:
            if not all(field in record for field in ("candidate_id", "candidate_sha256", "review")):
                raise EncoderTrainingError("Training variants require a content-bound review")
    try:
        validated = generate_variants(
            inputs, typo_variants=0,
            max_variants_per_parent=max(1, len(selected)), split=ENCODER_TRAIN,
        )
    except (VariantError, CorpusPreparationError, ValueError) as exc:
        raise EncoderTrainingError(f"Invalid assigned/reviewed training queries: {exc}") from exc
    selected_queries = list(validated.queries)
    parents = {record["query_id"]: record for record in selected_queries if record["variant_type"] == "original"}
    if not parents:
        raise EncoderTrainingError("No original encoder-training queries were supplied")
    judgment_ids = {str(record["judgment_query_id"]) for record in parents.values()}
    # A judgment ID must never be owned by both training and a held-out query.
    for record in inputs:
        if record.get("split") != ENCODER_TRAIN and record.get("judgment_query_id") in judgment_ids:
            raise EncoderTrainingError("A training judgment_query_id is shared with a held-out split")
    normalized_documents: list[DenseDocument] = []
    for value in documents:
        record = _mapping(value)
        try:
            document = DenseDocument(
                _clean_id(record.get("doc_id"), "doc_id"),
                str(record.get("title", "")), str(record.get("text", "")),
                str(record.get("language", "")), str(record.get("source", "unspecified")),
            )
            # Reuse exactly the inference-side preprocessing contract.
            format_passage(document.title, document.text)
        except (CorpusPreparationError, DenseError) as exc:
            raise EncoderTrainingError(str(exc)) from exc
        normalized_documents.append(document)
    normalized_documents.sort(key=lambda document: document.doc_id)
    doc_by_id = {document.doc_id: document for document in normalized_documents}
    if len(doc_by_id) != len(normalized_documents):
        raise EncoderTrainingError("Duplicate training corpus doc_id")
    if not doc_by_id:
        raise EncoderTrainingError("Training corpus is empty")
    scoped: dict[tuple[str, str], int] = {}
    ignored_qrels = 0
    for value in qrels:
        record = _mapping(value)
        query_id = str(record.get("query_id", ""))
        if query_id not in judgment_ids:
            ignored_qrels += 1
            continue
        try:
            doc_id = _clean_id(record.get("doc_id"), "doc_id")
        except CorpusPreparationError as exc:
            raise EncoderTrainingError(str(exc)) from exc
        relevance = record.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 0:
            raise EncoderTrainingError("Training relevance must be a nonnegative integer")
        if doc_id not in doc_by_id:
            raise EncoderTrainingError(f"Training judgment refers to missing passage {doc_id!r}")
        key = (query_id, doc_id)
        if key in scoped and scoped[key] != relevance:
            raise EncoderTrainingError("Conflicting training relevance judgments")
        scoped[key] = relevance
    positives = {query_id: tuple(sorted(doc_id for (qid, doc_id), relevance in scoped.items() if qid == query_id and relevance > 0)) for query_id in judgment_ids}
    if any(not values for values in positives.values()):
        missing = sorted(query_id for query_id, values in positives.items() if not values)
        raise EncoderTrainingError("No positive training judgments for: " + ", ".join(missing))
    bm25 = None
    if config.bm25_negatives:
        bm25 = BM25Index([BM25Document(document.doc_id, document.title, document.text, document.language, document.source) for document in normalized_documents])
    negatives: dict[str, tuple[str, ...]] = {}
    source_counts: Counter[str] = Counter()
    target = config.random_negatives + config.bm25_negatives
    for parent_id, parent in sorted(parents.items()):
        judgment_id = parent["judgment_query_id"]
        known_positive = set(positives[judgment_id])
        chosen: list[str] = []
        judged_nonrelevant = sorted(
            (doc_id for (qid, doc_id), relevance in scoped.items() if qid == judgment_id and relevance == 0),
            key=lambda doc_id: (_priority(config.seed, parent_id, doc_id), doc_id),
        )
        for doc_id in judged_nonrelevant[:target]:
            chosen.append(doc_id)
            source_counts["judged_nonrelevant"] += 1
        if bm25 is not None and len(chosen) < target:
            try:
                mined = bm25.search(parent["text"], top_k=min(len(doc_by_id), target + len(known_positive) + len(chosen)))
            except BM25Error as exc:
                raise EncoderTrainingError(f"Training-only BM25 mining failed: {exc}") from exc
            added = 0
            for hit in mined:
                if hit.doc_id not in known_positive and hit.doc_id not in chosen:
                    chosen.append(hit.doc_id)
                    source_counts["bm25_unjudged"] += 1
                    added += 1
                    if added >= config.bm25_negatives or len(chosen) >= target:
                        break
        remaining = sorted(
            (doc_id for doc_id in doc_by_id if doc_id not in known_positive and doc_id not in chosen),
            key=lambda doc_id: (_priority(config.seed, parent_id, doc_id), doc_id),
        )
        for doc_id in remaining[:max(0, target - len(chosen))]:
            chosen.append(doc_id)
            source_counts["random_unjudged"] += 1
        if known_positive.intersection(chosen):
            raise EncoderTrainingError("A known positive was selected as a negative")
        negatives[parent_id] = tuple(chosen)
    variants: dict[str, list[Mapping[str, Any]]] = {parent_id: [] for parent_id in parents}
    for record in selected_queries:
        if record["variant_type"] != "original":
            variants[record["parent_query_id"]].append(record)
    examples: list[EncoderTrainingExample] = []
    for parent_id, parent in sorted(parents.items()):
        paired = sorted(variants[parent_id], key=lambda record: record["query_id"])
        for variant in paired or [None]:
            examples.append(EncoderTrainingExample(
                original_query_id=parent_id, judgment_query_id=parent["judgment_query_id"], group_id=parent["group_id"],
                original_text=parent["text"], variant_query_id=variant["query_id"] if variant else None,
                variant_text=variant["text"] if variant else None,
                positive_doc_ids=positives[parent["judgment_query_id"]], negative_doc_ids=negatives[parent_id],
            ))
    used_qrels = [{"query_id": query_id, "doc_id": doc_id, "relevance": relevance} for (query_id, doc_id), relevance in sorted(scoped.items())]
    corpus_sha = _digest([asdict(document) for document in normalized_documents])
    fingerprint = _digest({"queries": selected_queries, "qrels": used_qrels, "corpus_sha256": corpus_sha, "examples": [asdict(example) for example in examples]})
    manifest = {
        "schema_version": 1, "selected_split": ENCODER_TRAIN,
        "training_fingerprint": fingerprint, "training_data_sha256": fingerprint,
        "corpus_sha256": corpus_sha, "used_qrels_sha256": _digest(used_qrels),
        "training_query_ids": sorted(record["query_id"] for record in selected_queries),
        "judgment_query_ids": sorted(judgment_ids), "group_ids": sorted({record["group_id"] for record in selected_queries}),
        "original_query_count": len(parents), "verified_variant_count": sum(len(values) for values in variants.values()),
        "example_count": len(examples), "variant_pair_count": sum(example.variant_text is not None for example in examples),
        "ignored_qrel_count": ignored_qrels,
        "excluded_unreviewed_variant_count": sum(record.get("variant_type", "original") != "original" and record.get("verified") is not True for record in selected),
        "negative_policy": {"known_positives_excluded": True, "unjudged_negatives": "potential_false_negatives_audit_required", "mining_queries_split": ENCODER_TRAIN, "judged_nonrelevant_preferred": True, "in_batch_negatives": config.in_batch_negatives, "heldout_judgments_used": False},
        "negative_source_counts": dict(sorted(source_counts.items())),
        "negative_mining_config": {
            "random_negatives": config.random_negatives,
            "bm25_negatives": config.bm25_negatives,
            "seed": config.seed,
        },
    }
    return EncoderTrainingData(tuple(examples), tuple(normalized_documents), manifest)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise EncoderTrainingError("Encoder training requires optional torch; install the project's dense extra") from exc
    return torch


def build_training_batch_masks(
    examples: Sequence[EncoderTrainingExample],
    doc_ids: Sequence[str],
    *,
    in_batch_negatives: bool = True,
) -> tuple[Any, Any]:
    """Mark every own known positive, including other rows' positive passages."""
    torch = _torch()
    if len(doc_ids) != len(set(doc_ids)):
        raise EncoderTrainingError("Batch candidate doc_ids must be unique")
    positives, candidates = [], []
    for example in examples:
        own_positive = set(example.positive_doc_ids)
        if not own_positive or not own_positive.issubset(doc_ids):
            raise EncoderTrainingError("Batch must contain every known positive")
        allowed = own_positive | set(example.negative_doc_ids)
        positives.append([doc_id in own_positive for doc_id in doc_ids])
        candidates.append([in_batch_negatives or doc_id in allowed for doc_id in doc_ids])
    return torch.tensor(positives, dtype=torch.bool), torch.tensor(candidates, dtype=torch.bool)


def multi_positive_retrieval_loss(
    query_embeddings: Any,
    document_embeddings: Any,
    positive_mask: Any,
    candidate_mask: Any = None,
    *,
    temperature: float = 0.05,
) -> Any:
    """Negative log probability of the union of all known positive passages."""
    torch = _torch()
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise EncoderTrainingError("temperature must be finite and positive")
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2 or query_embeddings.shape[1] != document_embeddings.shape[1] or not query_embeddings.shape[0] or not document_embeddings.shape[0]:
        raise EncoderTrainingError("Query/document embeddings must be nonempty compatible matrices")
    expected = (query_embeddings.shape[0], document_embeddings.shape[0])
    positive_mask = torch.as_tensor(positive_mask, device=query_embeddings.device)
    if positive_mask.dtype != torch.bool or tuple(positive_mask.shape) != expected:
        raise EncoderTrainingError("positive_mask must be a boolean query-by-document matrix")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(positive_mask)
    else:
        candidate_mask = torch.as_tensor(candidate_mask, device=query_embeddings.device)
    if candidate_mask.dtype != torch.bool or tuple(candidate_mask.shape) != expected:
        raise EncoderTrainingError("candidate_mask must be a boolean query-by-document matrix")
    if not bool(positive_mask.any(dim=1).all()) or bool((positive_mask & ~candidate_mask).any()):
        raise EncoderTrainingError("Each query needs a positive included in its candidate mask")
    if not bool(torch.isfinite(query_embeddings).all()) or not bool(torch.isfinite(document_embeddings).all()):
        raise EncoderTrainingError("Non-finite training embeddings")
    queries = torch.nn.functional.normalize(query_embeddings.float(), p=2, dim=1)
    documents = torch.nn.functional.normalize(document_embeddings.float(), p=2, dim=1)
    scores = (queries @ documents.transpose(0, 1)) / temperature
    numerator = torch.logsumexp(scores.masked_fill(~positive_mask, -torch.inf), dim=1)
    denominator = torch.logsumexp(scores.masked_fill(~candidate_mask, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def consistency_loss(original_embeddings: Any, variant_embeddings: Any, pair_mask: Any = None) -> Any:
    torch = _torch()
    if original_embeddings.ndim != 2 or original_embeddings.shape != variant_embeddings.shape:
        raise EncoderTrainingError("Consistency embeddings must be aligned matrices")
    if not bool(torch.isfinite(original_embeddings).all()) or not bool(torch.isfinite(variant_embeddings).all()):
        raise EncoderTrainingError("Non-finite consistency embeddings")
    if pair_mask is not None:
        pair_mask = torch.as_tensor(pair_mask, device=original_embeddings.device)
        if pair_mask.dtype != torch.bool or tuple(pair_mask.shape) != (original_embeddings.shape[0],):
            raise EncoderTrainingError("pair_mask must be a boolean row selector")
        original_embeddings, variant_embeddings = original_embeddings[pair_mask], variant_embeddings[pair_mask]
    if not original_embeddings.shape[0]:
        return (original_embeddings.sum() + variant_embeddings.sum()) * 0.0
    return (1.0 - torch.nn.functional.cosine_similarity(original_embeddings.float(), variant_embeddings.float(), dim=1)).mean()


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchFixtureTrainingEncoder:
    """Explicit small neural fixture: trainable embeddings, mean pooling, L2."""

    def __init__(self, dimension: int = 16, vocabulary_size: int = 256, *, seed: int = 13, device: str = "cpu", allow_fixture: bool = False) -> None:
        if not allow_fixture:
            raise EncoderTrainingError("The fixture-torch backend requires explicit allow_fixture=True")
        for name, value in (("dimension", dimension), ("vocabulary_size", vocabulary_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EncoderTrainingError(f"{name} must be a positive integer")
        torch = _torch()
        if device not in {"cpu", "cuda", "auto"}:
            raise EncoderTrainingError("device must be cpu, cuda, or auto")
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self._device == "cuda" and not torch.cuda.is_available():
            raise EncoderTrainingError("CUDA is unavailable")
        self._dimension, self._vocabulary_size, self._seed = dimension, vocabulary_size, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._embedding = torch.nn.Embedding(vocabulary_size, dimension).to(self._device)
        self._torch = torch

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "fixture_torch", "model_id": "fixture/trainable-token-embedding", "init_revision": "fixture-torch-v1", "dimension": self.dimension, "fixture_only": True, "fine_tuned": True, "init_seed": self._seed, "vocabulary_size": self._vocabulary_size}

    def parameters(self) -> Iterable[Any]:
        return self._embedding.parameters()

    def train(self) -> Any:
        return self._embedding.train()

    def eval(self) -> Any:
        return self._embedding.eval()

    def token_ids(self, text: str, max_length: int) -> list[int]:
        for prefix in (QUERY_PREFIX, PASSAGE_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        terms = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE) or ["#empty"]
        return [int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "big") % self._vocabulary_size for term in terms[:max_length]]

    def encode_training(self, texts: Sequence[str], *, max_length: int) -> Any:
        if not 1 <= max_length <= MODEL_MAX_LENGTH:
            raise EncoderTrainingError("max_length must be between 1 and 512")
        torch = self._torch
        if not texts:
            return torch.empty((0, self.dimension), device=self._device)
        pooled = [self._embedding(torch.tensor(self.token_ids(text, max_length), dtype=torch.long, device=self._device)).mean(dim=0) for text in texts]
        return torch.nn.functional.normalize(torch.stack(pooled), p=2, dim=1)

    def encode(self, texts: Sequence[str], *, batch_size: int = 8, max_length: int = MODEL_MAX_LENGTH) -> Sequence[Sequence[float]]:
        if batch_size <= 0:
            raise EncoderTrainingError("batch_size must be positive")
        self.eval()
        vectors = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                vectors.extend(self.encode_training(texts[start:start + batch_size], max_length=max_length).cpu().tolist())
        return vectors

    def save_pretrained(self, directory: Path) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise EncoderTrainingError("Checkpoint persistence requires optional safetensors") from exc
        directory.mkdir(parents=True, exist_ok=True)
        config = {"schema_version": 1, "dimension": self.dimension, "vocabulary_size": self._vocabulary_size, "tokenizer": "nfc_casefold_unicode_words_sha256_v1", "init_seed": self._seed}
        (directory / "fixture_encoder.json").write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        save_file({"embedding.weight": self._embedding.weight.detach().float().cpu().contiguous()}, str(directory / "fixture_model.safetensors"))


class TransformersTrainingEncoder:
    """Shared, trainable E5 model initialized from an immutable upstream SHA."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_MODEL_REVISION, *, device: str = "auto", allow_download: bool = False) -> None:
        if model_id != DEFAULT_MODEL_ID or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise EncoderTrainingError("E5 training requires multilingual-e5-small and an immutable 40-character revision")
        torch = _torch()
        try:
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EncoderTrainingError("E5 training requires optional transformers and safetensors; install the dense extra") from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise EncoderTrainingError("device must be auto, cpu, or cuda")
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self._device == "cuda" and not torch.cuda.is_available():
            raise EncoderTrainingError("CUDA is unavailable")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=not allow_download, trust_remote_code=False)
            self._model = AutoModel.from_pretrained(model_id, revision=revision, local_files_only=not allow_download, trust_remote_code=False, use_safetensors=True)
        except (OSError, ValueError) as exc:
            raise EncoderTrainingError("Could not load pinned E5; cache it or explicitly allow a download") from exc
        self._model.to(self._device)
        self._tokenizer.truncation_side = "right"
        self._torch = torch
        if self._model.config.hidden_size != 384:
            raise EncoderTrainingError("multilingual-e5-small must have 384-dimensional embeddings")
        resolved_revision = str(getattr(self._model.config, "_commit_hash", None) or revision)
        if resolved_revision != revision:
            raise EncoderTrainingError("Loaded E5 revision differs from the requested initialization")
        self._metadata = {"backend": "transformers_e5", "model_id": model_id, "init_revision": revision, "dimension": 384, "fine_tuned": True, "fixture_only": False, "torch_version": str(torch.__version__), "transformers_version": str(transformers.__version__), "device": self._device}

    @property
    def dimension(self) -> int:
        return 384

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def parameters(self) -> Iterable[Any]:
        return self._model.parameters()

    def train(self) -> Any:
        return self._model.train()

    def eval(self) -> Any:
        return self._model.eval()

    def encode_training(self, texts: Sequence[str], *, max_length: int) -> Any:
        if not 1 <= max_length <= MODEL_MAX_LENGTH:
            raise EncoderTrainingError("max_length must be between 1 and 512")
        batch = self._tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {key: value.to(self._device) for key, value in batch.items()}
        output = self._model(**batch)
        mask = batch["attention_mask"]
        tokens = output.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
        pooled = tokens.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1)
        return self._torch.nn.functional.normalize(pooled, p=2, dim=1)

    def save_pretrained(self, directory: Path) -> None:
        self._model.save_pretrained(str(directory), safe_serialization=True)
        self._tokenizer.save_pretrained(str(directory))


def _batch_objective(examples: Sequence[EncoderTrainingExample], documents: Mapping[str, DenseDocument], encoder: TrainableEncoder, config: EncoderTrainingConfig) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    doc_ids = sorted({doc_id for example in examples for doc_id in (*example.positive_doc_ids, *example.negative_doc_ids)})
    positive_mask, candidate_mask = build_training_batch_masks(examples, doc_ids, in_batch_negatives=config.in_batch_negatives)
    preprocessing = DenseConfig(max_length=config.max_length)
    original = encoder.encode_training([format_query(example.original_text, preprocessing) for example in examples], max_length=config.max_length)
    passages = encoder.encode_training([format_passage(documents[doc_id].title, documents[doc_id].text, preprocessing) for doc_id in doc_ids], max_length=config.max_length)
    original_losses = []
    for row in range(len(examples)):
        original_losses.append(multi_positive_retrieval_loss(original[row:row + 1], passages, positive_mask[row:row + 1], candidate_mask[row:row + 1], temperature=config.temperature))
    paired_indices = [row for row, example in enumerate(examples) if example.variant_text is not None]
    variant_by_row = {}
    variant_losses = []
    cons_losses = []
    if paired_indices:
        variants = encoder.encode_training([format_query(examples[row].variant_text, preprocessing) for row in paired_indices], max_length=config.max_length)
        for local_row, original_row in enumerate(paired_indices):
            variant_loss = multi_positive_retrieval_loss(variants[local_row:local_row + 1], passages, positive_mask[original_row:original_row + 1], candidate_mask[original_row:original_row + 1], temperature=config.temperature)
            cons = consistency_loss(original[original_row:original_row + 1], variants[local_row:local_row + 1])
            variant_by_row[original_row] = variant_loss + config.consistency_weight * cons
            variant_losses.append(variant_loss)
            cons_losses.append(cons)
    total = torch.stack([loss + variant_by_row.get(row, 0.0) for row, loss in enumerate(original_losses)]).mean()
    zero = original.sum() * 0.0
    return total, torch.stack(original_losses).mean(), torch.stack(variant_losses).mean() if variant_losses else zero, torch.stack(cons_losses).mean() if cons_losses else zero


def _parameter_delta_l2(parameters: Sequence[Any], initial_parameters: Sequence[Any], *, chunk_elements: int = PARAMETER_DELTA_CHUNK_ELEMENTS) -> float:
    """Match float-subtraction/double-squared L2 without full double tensors.

    E5's vocabulary embedding dominates its parameter count. Diagnostic delta
    computation must not materialize several embedding-sized double arrays.
    Only a bounded chunk is copied to CPU/converted to float64 at a time.
    """
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int) or not 1 <= chunk_elements <= PARAMETER_DELTA_CHUNK_ELEMENTS:
        raise EncoderTrainingError("Parameter-delta chunk size must be an integer in [1, 1000000]")
    if len(parameters) != len(initial_parameters):
        raise EncoderTrainingError("Parameter-delta snapshots must match the model")
    torch = _torch()
    squared_chunks = []
    with torch.no_grad():
        for parameter, initial in zip(parameters, initial_parameters, strict=True):
            if parameter.shape != initial.shape:
                raise EncoderTrainingError("Parameter-delta snapshot shapes differ")
            current, before = parameter.detach().reshape(-1), initial.detach().reshape(-1)
            for start in range(0, current.numel(), chunk_elements):
                difference = (current[start:start + chunk_elements].to(device="cpu") - before[start:start + chunk_elements].to(device="cpu")).to(dtype=torch.float64)
                difference.square_()
                squared_chunks.append(float(difference.sum()))
    squared = math.fsum(squared_chunks)
    if not math.isfinite(squared):
        raise EncoderTrainingError("Parameter delta became non-finite")
    return math.sqrt(squared)


def train_encoder(data: EncoderTrainingData, encoder: TrainableEncoder, config: EncoderTrainingConfig = EncoderTrainingConfig()) -> EncoderTrainingResult:
    """Optimize actual neural parameters; report deterministic evaluation loss."""
    config.validate()
    if not data.examples or data.manifest.get("selected_split") != ENCODER_TRAIN:
        raise EncoderTrainingError("Encoder training requires prepared nonempty encoder_train data")
    expected_mining = {
        "random_negatives": config.random_negatives,
        "bm25_negatives": config.bm25_negatives,
        "seed": config.seed,
    }
    if data.manifest.get("negative_mining_config") != expected_mining:
        raise EncoderTrainingError("Training negative-mining settings differ from prepared data; prepare it again")
    torch = _torch()
    _seed_everything(config.seed)
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if not parameters:
        raise EncoderTrainingError("Encoder has no trainable parameters")
    before = [parameter.detach().cpu().clone() for parameter in parameters]
    optimizer_class = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.SGD
    optimizer = optimizer_class(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    doc_by_id = {document.doc_id: document for document in data.documents}
    history: list[Mapping[str, Any]] = []

    def evaluate_objective() -> float:
        encoder.eval()
        accumulated = 0.0
        with torch.no_grad():
            for start in range(0, len(data.examples), config.batch_size):
                batch = data.examples[start:start + config.batch_size]
                loss = _batch_objective(batch, doc_by_id, encoder, config)[0]
                if not bool(torch.isfinite(loss)):
                    raise EncoderTrainingError("Non-finite evaluation objective")
                accumulated += float(loss.cpu()) * len(batch)
        return accumulated / len(data.examples)

    initial_loss = evaluate_objective()
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = list(range(len(data.examples)))
        rng.shuffle(order)
        encoder.train()
        for start in range(0, len(order), config.batch_size):
            batch = [data.examples[index] for index in order[start:start + config.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            total, original, variant, consistency = _batch_objective(batch, doc_by_id, encoder, config)
            if not bool(torch.isfinite(total)):
                raise EncoderTrainingError("Non-finite encoder training loss")
            total.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
            except RuntimeError as exc:
                raise EncoderTrainingError("Non-finite encoder gradients") from exc
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise EncoderTrainingError("Non-finite encoder parameters after optimizer update")
            history.append({"step": len(history) + 1, "epoch": epoch + 1, "example_count": len(batch), "loss": float(total.detach().cpu()), "original_retrieval_loss": float(original.detach().cpu()), "variant_retrieval_loss": float(variant.detach().cpu()), "consistency_loss": float(consistency.detach().cpu()), "gradient_norm": float(grad_norm.detach().cpu())})
    # Optimizer gradients no longer participate in training. Release them
    # before final forward evaluation and the diagnostic snapshot comparison.
    optimizer.zero_grad(set_to_none=True)
    final_loss = evaluate_objective()
    parameter_delta = _parameter_delta_l2(parameters, before)
    paired = data.manifest["variant_pair_count"] > 0
    manifest = {"schema_version": 1, "trainer": "shared_e5_multi_positive_contrastive_consistency_v1", "config": asdict(config), "preprocessing": DenseConfig(max_length=config.max_length).compatibility_mapping(), "training_data": dict(data.manifest), "training_fingerprint": data.manifest["training_fingerprint"], "training_data_sha256": data.manifest["training_fingerprint"], "seed": config.seed, "optimizer_updates": len(history), "initial_loss": initial_loss, "final_loss": final_loss, "parameter_delta_l2": parameter_delta, "gradients_finite": True, "variant_pair_count": data.manifest["variant_pair_count"], "lambda_zero_ablation": config.consistency_weight == 0 and paired, "consistency_applicable": paired, "training_regime": "reviewed_variant_retrieval_consistency" if paired else "original_only_retrieval", "encoder": dict(encoder.metadata), "torch_version": str(torch.__version__), "evaluation_policy": "training_objective_only_no_heldout_labels", "in_batch_negative_policy": "all_batch_candidates_unjudged_for_other_queries_except_all_own_known_positives", "training_history": history}
    return EncoderTrainingResult(tuple(history), manifest)


def save_trained_checkpoint(encoder: TrainableEncoder, result: EncoderTrainingResult, output_dir: Path, *, overwrite: bool = False) -> Mapping[str, Any]:
    """Stage a complete content-bound checkpoint before replacing an old one."""
    from training.checkpoints import verify_checkpoint, write_checkpoint_manifest
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise EncoderTrainingError("Checkpoint output cannot be a symlink")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if not overwrite:
            raise EncoderTrainingError("Refusing to overwrite existing checkpoint; pass --force")
        # --force can replace only a verified checkpoint, never an arbitrary dir.
        verify_checkpoint(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-stage-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        encoder.save_pretrained(stage)
        metadata = dict(encoder.metadata)
        metadata.update(fine_tuned=True, preprocessing=result.manifest["preprocessing"], training=dict(result.manifest))
        checkpoint = write_checkpoint_manifest(stage, metadata)
        if output_dir.exists():
            if any(output_dir.iterdir()):
                backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-previous-", dir=output_dir.parent))
                backup.rmdir()
                output_dir.rename(backup)
            else:
                output_dir.rmdir()
        stage.rename(output_dir)
        if backup is not None:
            shutil.rmtree(backup)
        return checkpoint
    except BaseException:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("e5", "fixture-torch"), default="e5")
    parser.add_argument("--allow-fixture-encoder", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=MODEL_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--random-negatives", type=int, default=1)
    parser.add_argument("--bm25-negatives", type=int, default=0)
    parser.add_argument("--no-in-batch-negatives", action="store_true")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fixture-dimension", type=int, default=16)
    parser.add_argument("--fixture-vocabulary-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        for input_path in (args.corpus, args.queries, args.qrels):
            if input_path.resolve() == args.output_dir.resolve() or args.output_dir.resolve() in input_path.resolve().parents:
                raise EncoderTrainingError("Checkpoint output cannot contain or overwrite an input artifact")
        config = EncoderTrainingConfig(epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, learning_rate=args.learning_rate, temperature=args.temperature, consistency_weight=args.consistency_weight, random_negatives=args.random_negatives, bm25_negatives=args.bm25_negatives, in_batch_negatives=not args.no_in_batch_negatives, seed=args.seed, optimizer=args.optimizer, weight_decay=args.weight_decay, gradient_clip=args.gradient_clip)
        config.validate()
        data = prepare_encoder_training(list(iter_jsonl(args.queries)), load_qrels(args.qrels), load_corpus(args.corpus), config)
        _seed_everything(config.seed)
        if args.backend == "fixture-torch":
            if args.allow_download:
                raise EncoderTrainingError("fixture-torch does not download models")
            encoder: TrainableEncoder = TorchFixtureTrainingEncoder(args.fixture_dimension, args.fixture_vocabulary_size, seed=config.seed, device=args.device, allow_fixture=args.allow_fixture_encoder)
        else:
            if args.allow_fixture_encoder:
                raise EncoderTrainingError("--allow-fixture-encoder requires --backend fixture-torch")
            encoder = TransformersTrainingEncoder(args.model, args.revision, device=args.device, allow_download=args.allow_download)
        result = train_encoder(data, encoder, config)
        checkpoint = save_trained_checkpoint(encoder, result, args.output_dir, overwrite=args.force)
    except (EncoderTrainingError, CorpusPreparationError, DenseError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"checkpoint": str(args.output_dir), "checkpoint_sha256": checkpoint["checkpoint_sha256"], "fixture_only": bool(encoder.metadata.get("fixture_only")), "optimizer_updates": result.manifest["optimizer_updates"], "initial_loss": result.manifest["initial_loss"], "final_loss": result.manifest["final_loss"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
