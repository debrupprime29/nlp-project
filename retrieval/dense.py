"""Frozen and locally fine-tuned E5 retrieval with persistent compatibility checks.

The research baseline uses ``intfloat/multilingual-e5-small`` without
fine-tuning.  Neural dependencies are optional and imported lazily, so the
rest of the project remains usable offline.  A deliberately named, explicitly
enabled fixture encoder exists only for deterministic pipeline smoke tests; it
is never selected as a fallback for E5.

Passages are encoded as ``passage: {title}\n{text}`` (omitting the title line
when it is empty), queries as ``query: {text}``, and all vectors are L2
normalized before inner-product search.  The E5 adapter uses attention-mask
mean pooling and right-side truncation to at most 512 tokens.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from data.make_splits import ALL_SPLITS
from data.prepare_corpus import (
    CorpusPreparationError,
    _canonical_json_bytes,
    _clean_id,
    _first,
    canonical_language,
    iter_jsonl,
)
from retrieval.bm25 import (
    BM25Error,
    DEFAULT_TOP_K,
    TOKENIZER_VERSION,
    RetrievalResult,
    _atomic_write_json,
    _atomic_write_jsonl,
    _file_sha256,
    tokenize,
    write_run_artifacts,
)


SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
DEFAULT_DIMENSION = 384
MODEL_MAX_LENGTH = 512
MAX_DENSE_QUERY_CHARACTERS = 131_072
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
INDEX_MANIFEST_NAME = "dense_manifest.json"
VECTOR_FILE_NAME = "vectors.jsonl"


class DenseError(ValueError):
    """Raised when dense-retrieval input or configuration is invalid."""


class DenseDependencyError(DenseError):
    """Raised when explicitly requested neural dependencies are unavailable."""


class DenseCompatibilityError(DenseError):
    """Raised when an encoder and saved index cannot safely be combined."""


@dataclass(frozen=True)
class DenseConfig:
    max_length: int = MODEL_MAX_LENGTH
    batch_size: int = 8
    query_prefix: str = QUERY_PREFIX
    passage_prefix: str = PASSAGE_PREFIX
    passage_format: str = "title_newline_text_skip_empty_title_v1"
    pooling: str = "attention_mask_mean"
    normalization: str = "l2"
    metric: str = "inner_product"
    truncation_side: str = "right"
    truncation_policy: str = "single_original_passage_head"
    dtype: str = "float32"
    text_normalization: str = "unicode_nfc"

    def validate(self) -> None:
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise DenseError("max_length must be an integer")
        if not 1 <= self.max_length <= MODEL_MAX_LENGTH:
            raise DenseError(
                f"max_length must be between 1 and {MODEL_MAX_LENGTH}"
            )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise DenseError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise DenseError("batch_size must be positive")
        supported = {
            "query_prefix": (self.query_prefix, QUERY_PREFIX),
            "passage_prefix": (self.passage_prefix, PASSAGE_PREFIX),
            "passage_format": (
                self.passage_format,
                "title_newline_text_skip_empty_title_v1",
            ),
            "pooling": (self.pooling, "attention_mask_mean"),
            "normalization": (self.normalization, "l2"),
            "metric": (self.metric, "inner_product"),
            "truncation_side": (self.truncation_side, "right"),
            "truncation_policy": (
                self.truncation_policy,
                "single_original_passage_head",
            ),
            "dtype": (self.dtype, "float32"),
            "text_normalization": (self.text_normalization, "unicode_nfc"),
        }
        for field, (actual, expected) in supported.items():
            if actual != expected:
                raise DenseError(
                    f"Unsupported {field} {actual!r}; expected {expected!r}"
                )

    def compatibility_mapping(self) -> dict[str, Any]:
        """Return only settings that change stored/query vector semantics."""

        return {
            "max_length": self.max_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "passage_format": self.passage_format,
            "pooling": self.pooling,
            "normalization": self.normalization,
            "metric": self.metric,
            "truncation_side": self.truncation_side,
            "truncation_policy": self.truncation_policy,
            "dtype": self.dtype,
            "text_normalization": self.text_normalization,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DenseConfig":
        if not isinstance(value, Mapping):
            raise DenseCompatibilityError("Saved dense configuration is not an object")
        required = set(cls().compatibility_mapping())
        missing = sorted(required - set(value))
        if missing:
            raise DenseCompatibilityError(
                "Saved dense configuration is missing: " + ", ".join(missing)
            )
        try:
            config = cls(
                max_length=value["max_length"],
                batch_size=value.get("batch_size", 8),
                query_prefix=value["query_prefix"],
                passage_prefix=value["passage_prefix"],
                passage_format=value["passage_format"],
                pooling=value["pooling"],
                normalization=value["normalization"],
                metric=value["metric"],
                truncation_side=value["truncation_side"],
                truncation_policy=value["truncation_policy"],
                dtype=value["dtype"],
                text_normalization=value["text_normalization"],
            )
            config.validate()
        except DenseError as exc:
            raise DenseCompatibilityError(
                f"Invalid saved dense configuration: {exc}"
            ) from exc
        return config


@dataclass(frozen=True)
class DenseDocument:
    doc_id: str
    title: str
    text: str
    language: str
    source: str


@dataclass(frozen=True)
class DenseQuery:
    query_id: str
    text: str
    split: str | None = None


@dataclass(frozen=True)
class DenseHit:
    doc_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class DenseMultiViewSearch:
    hits: tuple[DenseHit, ...]
    unique_view_count: int
    search_count: int
    intermediate_candidate_count: int


@runtime_checkable
class DenseEncoder(Protocol):
    """Minimal injectable encoder contract used by the index and tests."""

    @property
    def dimension(self) -> int: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Sequence[Sequence[float]]: ...


def _nfc_text(value: Any, *, field: str, query: bool = False) -> str:
    if not isinstance(value, str):
        raise DenseError(f"{field} must be a string")
    if "\x00" in value:
        raise DenseError(f"{field} contains a NUL byte")
    if query and len(value) > MAX_DENSE_QUERY_CHARACTERS:
        raise DenseError(
            f"Query exceeds the {MAX_DENSE_QUERY_CHARACTERS}-character safety limit"
        )
    return unicodedata.normalize("NFC", value)


def format_query(text: str, config: DenseConfig = DenseConfig()) -> str:
    config.validate()
    return config.query_prefix + _nfc_text(text, field="Query text", query=True)


def format_passage(
    title: str, text: str, config: DenseConfig = DenseConfig()
) -> str:
    config.validate()
    normalized_title = _nfc_text(title, field="Passage title")
    normalized_text = _nfc_text(text, field="Passage text")
    if not normalized_title.strip() and not normalized_text.strip():
        raise DenseError("Passage has no encodable title or text")
    body = (
        f"{normalized_title}\n{normalized_text}"
        if normalized_title.strip()
        else normalized_text
    )
    return config.passage_prefix + body


def _encoder_signature(encoder: DenseEncoder) -> dict[str, Any]:
    dimension = encoder.dimension
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise DenseCompatibilityError("Encoder dimension must be a positive integer")
    metadata = encoder.metadata
    if not isinstance(metadata, Mapping):
        raise DenseCompatibilityError("Encoder metadata must be an object")
    required = ("backend", "model_id", "resolved_revision")
    missing = [field for field in required if not str(metadata.get(field, "")).strip()]
    if missing:
        raise DenseCompatibilityError(
            "Encoder metadata is missing: " + ", ".join(missing)
        )
    declared_dimension = metadata.get("dimension", dimension)
    if declared_dimension != dimension:
        raise DenseCompatibilityError(
            "Encoder metadata dimension does not match encoder.dimension"
        )
    signature = {
        "backend": str(metadata["backend"]),
        "model_id": str(metadata["model_id"]),
        "resolved_revision": str(metadata["resolved_revision"]),
        "tokenizer_revision": str(
            metadata.get("tokenizer_revision", metadata["resolved_revision"])
        ),
        "dimension": dimension,
        "fixture_only": bool(metadata.get("fixture_only", False)),
    }
    if metadata.get("fine_tuned", False):
        checkpoint_sha256 = metadata.get("checkpoint_sha256")
        if metadata.get("fine_tuned") is not True or not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
            raise DenseCompatibilityError("Trained encoders need a verified checkpoint SHA256")
        if signature["resolved_revision"] != checkpoint_sha256 or signature["tokenizer_revision"] != checkpoint_sha256:
            raise DenseCompatibilityError("Trained encoder and tokenizer revisions must equal the checkpoint fingerprint")
        signature.update({"fine_tuned": True, "checkpoint_sha256": checkpoint_sha256})
    # Fail early if a custom encoder supplies values that cannot be recorded.
    try:
        _canonical_json_bytes(signature)
    except (TypeError, ValueError) as exc:
        raise DenseCompatibilityError("Encoder metadata is not JSON serializable") from exc
    return signature


def _compatibility_payload(
    encoder_signature: Mapping[str, Any], config: DenseConfig
) -> dict[str, Any]:
    return {
        "encoder": dict(encoder_signature),
        "preprocessing": config.compatibility_mapping(),
    }


def _validate_checkpoint_preprocessing(encoder: DenseEncoder, config: DenseConfig) -> None:
    metadata = encoder.metadata
    if not isinstance(metadata, Mapping):
        raise DenseCompatibilityError("Encoder metadata must be an object")
    if metadata.get("fine_tuned", False) and metadata.get("preprocessing") != config.compatibility_mapping():
        raise DenseCompatibilityError("Dense preprocessing does not match the trained checkpoint")


def _index_kind(signature: Mapping[str, Any]) -> tuple[str, str, bool]:
    trained = bool(signature.get("fine_tuned", False))
    if signature["fixture_only"]:
        return ("dense_trained_test_fixture", "trained_neural_test_fixture", True) if trained else ("dense_test_fixture", "deterministic_test_fixture", False)
    return ("fine_tuned_multilingual_e5", "fine_tuned_multilingual_e5", True) if trained else ("frozen_multilingual_e5", "off_the_shelf_pretrained_frozen", False)


def _default_branch(signature: Mapping[str, Any]) -> str:
    if signature["fixture_only"]:
        return "dense_fixture_trained" if signature.get("fine_tuned") else "dense_fixture_hash"
    return "dense_e5_trained" if signature.get("fine_tuned") else "dense_e5_frozen"


def _compatibility_hash(
    encoder_signature: Mapping[str, Any], config: DenseConfig
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_compatibility_payload(encoder_signature, config))
    ).hexdigest()


def _vector_digest(
    doc_ids: Sequence[str], vectors: Sequence[Sequence[float]]
) -> str:
    digest = hashlib.sha256()
    for doc_id, vector in zip(doc_ids, vectors, strict=True):
        digest.update(
            _canonical_json_bytes({"doc_id": doc_id, "vector": list(vector)})
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _normalized_vector(
    value: Sequence[float],
    *,
    expected_dimension: int,
    context: str,
    require_unit: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise DenseError(f"{context} is not a numeric vector")
    try:
        raw_values = list(value)
    except TypeError as exc:
        raise DenseError(f"{context} is not a numeric vector") from exc
    if len(raw_values) != expected_dimension:
        raise DenseCompatibilityError(
            f"{context} has dimension {len(raw_values)}; expected {expected_dimension}"
        )
    vector: list[float] = []
    for item in raw_values:
        if isinstance(item, bool):
            raise DenseError(f"{context} contains a nonnumeric value")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise DenseError(f"{context} contains a nonnumeric value") from exc
        if not math.isfinite(number):
            raise DenseError(f"{context} contains a non-finite value")
        vector.append(number)
    norm = math.sqrt(sum(number * number for number in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise DenseError(f"{context} has a zero or invalid norm")
    if require_unit and not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise DenseCompatibilityError(
            f"{context} is not L2 normalized (norm={norm:.8g})"
        )
    if require_unit:
        # Preserve the serialized float values exactly so the logical artifact
        # checksum is stable. Validation must not silently repair saved data.
        return tuple(vector)
    return tuple(
        struct.unpack("!f", struct.pack("!f", number / norm))[0]
        for number in vector
    )


class TransformersE5Encoder:
    """Lazy optional adapter for the frozen multilingual-E5 checkpoint."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        device: str = "auto",
        allow_download: bool = False,
    ) -> None:
        if model_id != DEFAULT_MODEL_ID:
            raise DenseError(f"The frozen E5 baseline requires model {DEFAULT_MODEL_ID!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise DenseCompatibilityError("E5 revision must be an immutable 40-character commit SHA")
        if device not in {"auto", "cpu", "cuda"}:
            raise DenseError("device must be auto, cpu, or cuda")
        try:
            import torch  # type: ignore[import-not-found]
            import transformers  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DenseDependencyError(
                "Frozen E5 requires the optional 'torch' and 'transformers' "
                "packages; install the project's dense extra"
            ) from exc
        if not model_id.strip() or not revision.strip():
            raise DenseError("model_id and revision must not be empty")
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise DenseDependencyError("CUDA was requested but is unavailable")
        else:
            resolved_device = device
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=not allow_download,
                trust_remote_code=False,
            )
            self._model = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=not allow_download,
                trust_remote_code=False,
                use_safetensors=True,
            )
        except (OSError, ValueError) as exc:
            hint = (
                "Check network/cache access and the pinned revision."
                if allow_download
                else "Cache the pinned model first or pass --allow-download."
            )
            raise DenseDependencyError(f"Could not load frozen E5. {hint}") from exc
        self._tokenizer.truncation_side = "right"
        self._model.eval()
        self._model.to(resolved_device)
        dimension = getattr(self._model.config, "hidden_size", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise DenseCompatibilityError("Loaded model has no valid hidden_size")
        if dimension != DEFAULT_DIMENSION:
            raise DenseCompatibilityError("E5-small must produce 384-dimensional embeddings")
        resolved_revision = str(
            getattr(self._model.config, "_commit_hash", None) or revision
        )
        tokenizer_revision = str(
            getattr(self._tokenizer, "init_kwargs", {}).get("_commit_hash")
            or resolved_revision
        )
        self._torch = torch
        self._device = resolved_device
        self._dimension = dimension
        self._metadata = {
            "backend": "transformers_auto_model",
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "tokenizer_revision": tokenizer_revision,
            "dimension": dimension,
            "fixture_only": False,
            "device": resolved_device,
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
        }

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Sequence[Sequence[float]]:
        if batch_size <= 0:
            raise DenseError("batch_size must be positive")
        if not 1 <= max_length <= MODEL_MAX_LENGTH:
            raise DenseError(
                f"max_length must be between 1 and {MODEL_MAX_LENGTH}"
            )
        all_vectors: list[list[float]] = []
        torch = self._torch
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._model(**encoded)
                attention_mask = encoded["attention_mask"]
                token_embeddings = output.last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )
                pooled = token_embeddings.sum(dim=1) / attention_mask.sum(
                    dim=1
                )[..., None].clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vectors.extend(pooled.to(dtype=torch.float32).cpu().tolist())
        return all_vectors


class FixtureHashEncoder:
    """Deterministic lexical encoder for tests, never a substitute for E5."""

    def __init__(self, dimension: int = 64, *, allow_fixture: bool = False) -> None:
        if not allow_fixture:
            raise DenseError(
                "FixtureHashEncoder is test-only; explicitly set allow_fixture=True"
            )
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 8:
            raise DenseError("Fixture encoder dimension must be an integer of at least 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "fixture_hash_v1",
            "model_id": "fixture/hash-encoder",
            "requested_revision": "v1",
            "resolved_revision": "v1",
            "tokenizer_revision": TOKENIZER_VERSION,
            "dimension": self.dimension,
            "fixture_only": True,
        }

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Sequence[Sequence[float]]:
        del batch_size, max_length
        vectors: list[list[float]] = []
        for text in texts:
            payload = text
            for prefix in (QUERY_PREFIX, PASSAGE_PREFIX):
                if payload.startswith(prefix):
                    payload = payload[len(prefix) :]
                    break
            terms = tokenize(payload)
            features = list(terms)
            normalized = unicodedata.normalize("NFC", payload).casefold()
            features.extend(
                f"#3:{normalized[index:index + 3]}"
                for index in range(max(0, len(normalized) - 2))
            )
            if not features:
                features = ["#empty"]
            vector = [0.0] * self.dimension
            for feature, count in Counter(features).items():
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[index] += sign * count
            vectors.append(vector)
        return vectors


class TrainedCheckpointEncoder(TransformersE5Encoder):
    """Verified local fine-tuned E5, or an explicitly enabled neural fixture.

    Checkpoints are content-addressed and use safetensors exclusively.  The
    fixture is a trainable mean-of-token-embeddings model for integration tests;
    its metadata and run labels distinguish it from multilingual E5.
    """

    def __init__(self, checkpoint: str | Path, *, device: str = "auto", allow_fixture: bool = False) -> None:
        from training.checkpoints import CheckpointError, verify_checkpoint
        try:
            manifest = verify_checkpoint(Path(checkpoint))
        except CheckpointError as exc:
            raise DenseCompatibilityError(str(exc)) from exc
        source = manifest["metadata"]
        fixture = source["fixture_only"]
        if fixture and not allow_fixture:
            raise DenseError("A trained fixture encoder requires explicit allow_fixture=True")
        if device not in {"auto", "cpu", "cuda"}:
            raise DenseError("device must be auto, cpu, or cuda")
        try:
            import torch
        except ImportError as exc:
            raise DenseDependencyError("Trained encoders require the optional torch dependency") from exc
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise DenseDependencyError("CUDA was requested but is unavailable")
        self._fixture = fixture
        self._checkpoint_preprocessing = source["preprocessing"]
        self._torch, self._device, self._dimension = torch, resolved_device, source["dimension"]
        directory = Path(checkpoint)
        if fixture:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise DenseDependencyError("Trained fixtures require safetensors") from exc
            try:
                config = json.loads((directory / "fixture_encoder.json").read_text(encoding="utf-8"))
                expected_fields = {"schema_version", "dimension", "vocabulary_size", "tokenizer", "init_seed"}
                if not isinstance(config, dict) or set(config) != expected_fields or config["schema_version"] != 1 or config["dimension"] != self._dimension or config["tokenizer"] != "nfc_casefold_unicode_words_sha256_v1":
                    raise DenseCompatibilityError("Invalid trained fixture configuration")
                vocabulary_size = config["vocabulary_size"]
                if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size < 1:
                    raise DenseCompatibilityError("Invalid trained fixture vocabulary size")
                state = load_file(str(directory / "fixture_model.safetensors"), device="cpu")
                if set(state) != {"embedding.weight"}:
                    raise DenseCompatibilityError("Invalid trained fixture tensor keys")
                weight = state["embedding.weight"]
                if tuple(weight.shape) != (vocabulary_size, self._dimension) or weight.dtype != torch.float32 or not bool(torch.isfinite(weight).all()):
                    raise DenseCompatibilityError("Invalid trained fixture embedding shape, dtype, or values")
                self._vocabulary_size = vocabulary_size
                self._model = torch.nn.Embedding.from_pretrained(weight, freeze=True).to(resolved_device).eval()
            except Exception as exc:
                if isinstance(exc, DenseCompatibilityError):
                    raise
                raise DenseCompatibilityError(f"Could not load trained fixture: {exc}") from exc
        else:
            try:
                import transformers
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise DenseDependencyError("Fine-tuned E5 requires torch and transformers") from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
                self._model = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_safetensors=True)
            except (OSError, ValueError, RuntimeError) as exc:
                raise DenseCompatibilityError(f"Could not load verified local E5 checkpoint: {exc}") from exc
            self._tokenizer.truncation_side = "right"
            if getattr(self._model.config, "hidden_size", None) != self._dimension:
                raise DenseCompatibilityError("Fine-tuned model dimension does not match its manifest")
            self._model.float().to(resolved_device).eval()
        fingerprint = manifest["checkpoint_sha256"]
        self._metadata = {
            "backend": "fixture_torch_trained" if fixture else "transformers_e5_finetuned",
            "model_id": source["model_id"], "init_revision": source["init_revision"],
            "requested_revision": fingerprint, "resolved_revision": fingerprint,
            "tokenizer_revision": fingerprint, "dimension": self._dimension,
            "fixture_only": fixture, "fine_tuned": True,
            "checkpoint_sha256": fingerprint, "checkpoint_path": str(directory),
            "preprocessing": copy.deepcopy(source["preprocessing"]),
            "training": copy.deepcopy(source["training"]),
            "device": resolved_device, "torch_version": str(torch.__version__),
        }
        if not fixture:
            self._metadata["transformers_version"] = str(transformers.__version__)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._metadata)

    def encode(self, texts: Sequence[str], *, batch_size: int, max_length: int) -> Sequence[Sequence[float]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise DenseError("batch_size must be a positive integer")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length != self._checkpoint_preprocessing["max_length"]:
            raise DenseCompatibilityError("Query truncation length does not match trained checkpoint")
        if not self._fixture:
            return super().encode(texts, batch_size=batch_size, max_length=max_length)
        vectors = []
        for text in texts:
            payload = _nfc_text(text, field="Encoder input")
            for prefix in (QUERY_PREFIX, PASSAGE_PREFIX):
                if payload.startswith(prefix):
                    payload = payload[len(prefix):]
                    break
            terms = re.findall(r"\w+", payload.casefold(), flags=re.UNICODE) or ["#empty"]
            ids = [int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "big") % self._vocabulary_size for term in terms[:max_length]]
            with self._torch.inference_mode():
                token_ids = self._torch.tensor(ids, dtype=self._torch.long, device=self._device)
                pooled = self._model(token_ids).mean(dim=0)
                vector = self._torch.nn.functional.normalize(pooled, p=2, dim=-1)
            vectors.append(vector.to(dtype=self._torch.float32).cpu().tolist())
        return vectors


FineTunedE5Encoder = TrainedCheckpointEncoder


class DenseIndex:
    """Immutable brute-force dense index for fixtures and reduced corpora."""

    def __init__(
        self,
        documents: Sequence[DenseDocument],
        encoder: DenseEncoder,
        config: DenseConfig = DenseConfig(),
    ) -> None:
        config.validate()
        _validate_checkpoint_preprocessing(encoder, config)
        signature = _encoder_signature(encoder)
        ordered = sorted(documents, key=lambda document: document.doc_id)
        doc_ids: list[str] = []
        corpus_digest = hashlib.sha256()
        encoded_passages: list[str] = []
        for document in ordered:
            try:
                doc_id = _clean_id(document.doc_id, "doc_id")
            except CorpusPreparationError as exc:
                raise DenseError(str(exc)) from exc
            doc_ids.append(doc_id)
            canonical_document = DenseDocument(
                doc_id=doc_id,
                title=_nfc_text(document.title, field=f"Title for {doc_id!r}"),
                text=_nfc_text(document.text, field=f"Text for {doc_id!r}"),
                language=str(document.language),
                source=str(document.source),
            )
            corpus_digest.update(_canonical_json_bytes(asdict(canonical_document)))
            corpus_digest.update(b"\n")
            encoded_passages.append(
                format_passage(canonical_document.title, canonical_document.text, config)
            )
        duplicates = sorted(
            doc_id for doc_id, count in Counter(doc_ids).items() if count > 1
        )
        if duplicates:
            raise DenseError(f"Duplicate document ID(s): {', '.join(duplicates[:5])}")
        raw_vectors = (
            encoder.encode(
                encoded_passages,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
            if encoded_passages
            else []
        )
        vectors = list(raw_vectors)
        if len(vectors) != len(doc_ids):
            raise DenseCompatibilityError(
                "Encoder returned a different number of vectors than passages"
            )
        normalized_vectors = tuple(
            _normalized_vector(
                vector,
                expected_dimension=signature["dimension"],
                context=f"Document vector for {doc_id!r}",
            )
            for doc_id, vector in zip(doc_ids, vectors, strict=True)
        )
        compatibility_hash = _compatibility_hash(signature, config)
        vector_sha256 = _vector_digest(doc_ids, normalized_vectors)
        corpus_fingerprint = corpus_digest.hexdigest()
        index_fingerprint = hashlib.sha256(
            f"{corpus_fingerprint}\0{compatibility_hash}\0{vector_sha256}".encode(
                "ascii"
            )
        ).hexdigest()
        retriever, baseline_kind, fine_tuned = _index_kind(signature)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "retriever": retriever,
            "baseline_kind": baseline_kind,
            "fine_tuned": fine_tuned,
            "encoder": signature,
            "encoder_operational_metadata": dict(encoder.metadata),
            "config": {**config.compatibility_mapping(), "batch_size": config.batch_size},
            "compatibility_hash": compatibility_hash,
            "config_hash": hashlib.sha256(_canonical_json_bytes(config.compatibility_mapping())).hexdigest(),
            "dimension": signature["dimension"],
            "document_count": len(doc_ids),
            "corpus_fingerprint": corpus_fingerprint,
            "vector_sha256": vector_sha256,
            "index_fingerprint": index_fingerprint,
            "vector_storage": "normalized_float32_json_v1",
            "search_backend": "brute_force_python",
            "chunking": False,
            "original_passage_mapping": "one_vector_per_doc_id",
            "fixture_preprocessing_note": (
                (
                    "Trainable hashed-token mean embeddings with token truncation; no E5 semantics are tested"
                    if fine_tuned else
                    "Lexical hashing ignores model-token truncation; no E5 semantics are tested"
                ) if signature["fixture_only"] else None
            ),
        }
        self._initialize(
            doc_ids=tuple(doc_ids),
            vectors=normalized_vectors,
            encoder=encoder,
            config=config,
            signature=signature,
            manifest=manifest,
        )

    def _initialize(
        self,
        *,
        doc_ids: tuple[str, ...],
        vectors: tuple[tuple[float, ...], ...],
        encoder: DenseEncoder,
        config: DenseConfig,
        signature: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        self._doc_ids = doc_ids
        self._vectors = vectors
        self._encoder = encoder
        self.config = config
        self._signature = dict(signature)
        self._manifest = dict(manifest)

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._manifest)

    def _assert_encoder_compatible(self) -> None:
        _validate_checkpoint_preprocessing(self._encoder, self.config)
        current = _encoder_signature(self._encoder)
        if current != self._signature:
            raise DenseCompatibilityError(
                "Encoder identity changed after the dense index was constructed"
            )
        current_hash = _compatibility_hash(current, self.config)
        if current_hash != self._manifest.get("compatibility_hash"):
            raise DenseCompatibilityError(
                "Encoder/preprocessing settings do not match the dense index"
            )

    def search(
        self, query_text: str, *, top_k: int = DEFAULT_TOP_K
    ) -> tuple[DenseHit, ...]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise DenseError("top_k must be a nonnegative integer")
        normalized_query = _nfc_text(query_text, field="Query text", query=True)
        if top_k == 0 or not self._doc_ids or not normalized_query.strip():
            return ()
        self._assert_encoder_compatible()
        raw = list(
            self._encoder.encode(
                [format_query(normalized_query, self.config)],
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
            )
        )
        if len(raw) != 1:
            raise DenseCompatibilityError("Encoder did not return one query vector")
        query_vector = _normalized_vector(
            raw[0],
            expected_dimension=self._signature["dimension"],
            context="Query vector",
        )
        scored: list[tuple[str, float]] = []
        for doc_id, document_vector in zip(
            self._doc_ids, self._vectors, strict=True
        ):
            score = sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector, document_vector, strict=True
                )
            )
            if not math.isfinite(score):
                raise DenseError(f"Similarity for document {doc_id!r} is non-finite")
            scored.append((doc_id, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return tuple(
            DenseHit(doc_id, rank, score)
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        )

    def search_views(
        self,
        query_views: Sequence[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        per_view_top_k: int = DEFAULT_TOP_K,
        k_view: float = 60.0,
    ) -> DenseMultiViewSearch:
        if (
            isinstance(top_k, bool)
            or isinstance(per_view_top_k, bool)
            or not isinstance(top_k, int)
            or not isinstance(per_view_top_k, int)
            or top_k < 0
            or per_view_top_k < 0
        ):
            raise DenseError("candidate limits must be nonnegative integers")
        if not math.isfinite(k_view) or k_view < 0:
            raise DenseError("k_view must be finite and nonnegative")
        unique: dict[str, str] = {}
        for view in query_views:
            normalized = _nfc_text(view, field="Query view", query=True)
            if normalized.strip() and normalized not in unique:
                unique[normalized] = normalized
        if not unique or top_k == 0 or per_view_top_k == 0:
            return DenseMultiViewSearch((), len(unique), 0, 0)
        contributions: dict[str, float] = defaultdict(float)
        intermediate_count = 0
        for view in unique.values():
            hits = self.search(view, top_k=per_view_top_k)
            intermediate_count += len(hits)
            for hit in hits:
                contributions[hit.doc_id] += 1.0 / (k_view + hit.rank)
        divisor = len(unique)
        combined = sorted(
            ((doc_id, value / divisor) for doc_id, value in contributions.items()),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        hits = tuple(
            DenseHit(doc_id, rank, score)
            for rank, (doc_id, score) in enumerate(combined, start=1)
        )
        return DenseMultiViewSearch(
            hits=hits,
            unique_view_count=len(unique),
            search_count=len(unique),
            intermediate_candidate_count=intermediate_count,
        )

    def save(
        self,
        index_dir: Path,
        *,
        overwrite: bool = False,
        build_seconds: float | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        manifest_path = index_dir / INDEX_MANIFEST_NAME
        vector_path = index_dir / VECTOR_FILE_NAME
        existing = [path for path in (manifest_path, vector_path) if path.exists()]
        if existing and not overwrite:
            raise DenseError(
                "Refusing to overwrite existing dense artifact(s): "
                + ", ".join(str(path) for path in existing)
                + "; pass --force"
            )
        if build_seconds is not None and (
            not math.isfinite(build_seconds) or build_seconds < 0
        ):
            raise DenseError("build_seconds must be finite and nonnegative")
        records = (
            {"doc_id": doc_id, "vector": list(vector)}
            for doc_id, vector in zip(self._doc_ids, self._vectors, strict=True)
        )
        _atomic_write_jsonl(vector_path, records)
        saved_manifest = dict(self._manifest)
        saved_manifest["vector_file"] = VECTOR_FILE_NAME
        if build_seconds is not None:
            saved_manifest["build_seconds"] = build_seconds
        if corpus_path is not None:
            saved_manifest["corpus_path"] = str(corpus_path)
            saved_manifest["corpus_file_sha256"] = _file_sha256(corpus_path)
        _atomic_write_json(manifest_path, saved_manifest)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        encoder: DenseEncoder,
        *,
        expected_config: DenseConfig | None = None,
    ) -> "DenseIndex":
        manifest_path = index_dir / INDEX_MANIFEST_NAME
        vector_path = index_dir / VECTOR_FILE_NAME
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DenseCompatibilityError(
                f"Could not read dense manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DenseCompatibilityError("Dense manifest must be a JSON object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise DenseCompatibilityError("Unsupported dense index schema version")
        current_signature = _encoder_signature(encoder)
        expected_retriever, expected_baseline, expected_fine_tuned = _index_kind(current_signature)
        if value.get("retriever") != expected_retriever:
            raise DenseCompatibilityError("Artifact retriever does not match encoder kind")
        if (
            value.get("baseline_kind") != expected_baseline
            or value.get("fine_tuned") is not expected_fine_tuned
            or value.get("search_backend") != "brute_force_python"
            or value.get("chunking") is not False
            or value.get("original_passage_mapping") != "one_vector_per_doc_id"
        ):
            raise DenseCompatibilityError("Dense manifest baseline or search policy is inconsistent")
        if value.get("vector_file") != VECTOR_FILE_NAME:
            raise DenseCompatibilityError("Unsupported dense vector artifact name")
        if value.get("vector_storage") != "normalized_float32_json_v1":
            raise DenseCompatibilityError("Unsupported dense vector storage format")
        config = DenseConfig.from_mapping(value.get("config", {}))
        _validate_checkpoint_preprocessing(encoder, config)
        expected_config_hash = hashlib.sha256(
            _canonical_json_bytes(config.compatibility_mapping())
        ).hexdigest()
        if value.get("config_hash") != expected_config_hash:
            raise DenseCompatibilityError("Dense configuration hash mismatch")
        if expected_config is not None:
            expected_config.validate()
            if (
                expected_config.compatibility_mapping()
                != config.compatibility_mapping()
            ):
                raise DenseCompatibilityError(
                    "Requested preprocessing configuration does not match saved index"
                )
            config = replace(config, batch_size=expected_config.batch_size)
        stored_signature = value.get("encoder")
        if not isinstance(stored_signature, dict):
            raise DenseCompatibilityError("Dense manifest has no encoder signature")
        if stored_signature != current_signature:
            raise DenseCompatibilityError(
                "Encoder model, revision, tokenizer, or dimension does not match index"
            )
        expected_compatibility = _compatibility_hash(current_signature, config)
        if value.get("compatibility_hash") != expected_compatibility:
            raise DenseCompatibilityError(
                "Dense index compatibility hash is invalid or incompatible"
            )
        if value.get("dimension") != current_signature["dimension"]:
            raise DenseCompatibilityError("Dense manifest dimension is inconsistent")
        doc_ids: list[str] = []
        vectors: list[tuple[float, ...]] = []
        try:
            for line_number, record in enumerate(iter_jsonl(vector_path), start=1):
                try:
                    doc_id = _clean_id(record.get("doc_id"), "doc_id")
                except CorpusPreparationError as exc:
                    raise DenseCompatibilityError(str(exc)) from exc
                if "vector" not in record:
                    raise DenseCompatibilityError(
                        f"Vector row {line_number} is missing vector"
                    )
                doc_ids.append(doc_id)
                vectors.append(
                    _normalized_vector(
                        record["vector"],
                        expected_dimension=current_signature["dimension"],
                        context=f"Stored vector for {doc_id!r}",
                        require_unit=True,
                    )
                )
        except (OSError, CorpusPreparationError) as exc:
            raise DenseCompatibilityError(
                f"Could not read dense vectors {vector_path}: {exc}"
            ) from exc
        if len(doc_ids) != len(set(doc_ids)):
            raise DenseCompatibilityError("Dense vector artifact has duplicate doc_ids")
        if doc_ids != sorted(doc_ids):
            raise DenseCompatibilityError("Dense vector rows are not sorted by doc_id")
        if value.get("document_count") != len(doc_ids):
            raise DenseCompatibilityError("Dense vector row count does not match manifest")
        vector_sha256 = _vector_digest(doc_ids, vectors)
        if value.get("vector_sha256") != vector_sha256:
            raise DenseCompatibilityError("Dense vector checksum mismatch")
        corpus_fingerprint = value.get("corpus_fingerprint")
        if (
            not isinstance(corpus_fingerprint, str)
            or len(corpus_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in corpus_fingerprint)
        ):
            raise DenseCompatibilityError("Dense manifest has an invalid corpus fingerprint")
        index_fingerprint = hashlib.sha256(
            f"{corpus_fingerprint}\0{expected_compatibility}\0{vector_sha256}".encode(
                "ascii"
            )
        ).hexdigest()
        if value.get("index_fingerprint") != index_fingerprint:
            raise DenseCompatibilityError("Dense index fingerprint mismatch")
        loaded = cls.__new__(cls)
        loaded._initialize(
            doc_ids=tuple(doc_ids),
            vectors=tuple(vectors),
            encoder=encoder,
            config=config,
            signature=current_signature,
            manifest=value,
        )
        return loaded


def load_corpus(path: Path) -> list[DenseDocument]:
    documents: list[DenseDocument] = []
    seen: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            doc_id = _clean_id(
                _first(record, ("doc_id", "docid", "document_id")), "doc_id"
            )
            if doc_id in seen:
                raise DenseError(f"Duplicate document ID {doc_id!r}")
            seen.add(doc_id)
            raw_text = record.get("text")
            if raw_text is None:
                raise DenseError(f"Document {doc_id!r} is missing text")
            text = _nfc_text(str(raw_text), field=f"Text for {doc_id!r}")
            title = _nfc_text(
                str(record.get("title", "")), field=f"Title for {doc_id!r}"
            )
            if not text.strip() and not title.strip():
                raise DenseError(f"Document {doc_id!r} has no encodable fields")
            language = canonical_language(str(record.get("language", "")))
            source = str(record.get("source", "unspecified"))
            documents.append(DenseDocument(doc_id, title, text, language, source))
        except (DenseError, CorpusPreparationError) as exc:
            raise DenseError(f"{path}:{line_number}: {exc}") from exc
    return documents


def load_dense_queries(path: Path, *, split: str | None = None) -> list[DenseQuery]:
    """Load original query text without BM25's lexical token-count limit."""

    if split is not None and split not in ALL_SPLITS:
        raise DenseError(f"Unknown split {split!r}")
    queries: list[DenseQuery] = []
    seen: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        try:
            query_id = _clean_id(
                _first(record, ("query_id", "queryid", "qid")), "query_id"
            )
            if query_id in seen:
                raise DenseError(f"Duplicate query ID {query_id!r}")
            seen.add(query_id)
            raw_split = record.get("split")
            query_split = str(raw_split) if raw_split is not None else None
            if split is not None and query_split != split:
                continue
            raw_text = _first(record, ("text", "query", "query_text"))
            if raw_text is None:
                raise DenseError(f"Query {query_id!r} is missing text")
            text = _nfc_text(str(raw_text), field="Query text", query=True)
            queries.append(DenseQuery(query_id, text, query_split))
        except (DenseError, CorpusPreparationError) as exc:
            raise DenseError(f"{path}:{line_number}: {exc}") from exc
    if not queries:
        suffix = f" for split {split!r}" if split else ""
        raise DenseError(f"No dense retrieval queries found in {path}{suffix}")
    return queries


def run_queries(
    index: DenseIndex,
    queries: Sequence[DenseQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
    branch: str | None = None,
) -> tuple[RetrievalResult, ...]:
    fixture_only = bool(index.manifest["encoder"]["fixture_only"])
    signature = index.manifest["encoder"]
    branch = branch or _default_branch(signature)
    if fixture_only and branch == "dense_e5_frozen":
        raise DenseError("A fixture encoder run cannot use the E5 baseline label")
    if fixture_only and "fixture" not in branch:
        raise DenseError("A fixture encoder run must use an explicit fixture label")
    if signature.get("fine_tuned") and branch == "dense_e5_frozen":
        raise DenseError("A trained encoder cannot use the frozen E5 baseline label")
    try:
        branch = _clean_id(branch, "branch")
    except CorpusPreparationError as exc:
        raise DenseError(str(exc)) from exc
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise DenseError("Duplicate query IDs in retrieval input")
    results: list[RetrievalResult] = []
    for query in sorted(queries, key=lambda item: item.query_id):
        for hit in index.search(query.text, top_k=top_k):
            results.append(
                RetrievalResult(
                    query_id=query.query_id,
                    doc_id=hit.doc_id,
                    branch=branch,
                    rank=hit.rank,
                    score=hit.score,
                )
            )
    return tuple(results)


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("e5", "fixture-hash", "e5-trained"), default="e5")
    parser.add_argument("--checkpoint", type=Path, help="verified local trained encoder checkpoint for e5-trained")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=MODEL_MAX_LENGTH)
    parser.add_argument("--fixture-dimension", type=int, default=64)
    parser.add_argument("--allow-fixture-encoder", action="store_true")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="encode and persist passages")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--index-dir", type=Path, required=True)
    build.add_argument("--force", action="store_true")
    _add_encoder_arguments(build)

    search = subparsers.add_parser("search", help="load an index and search queries")
    search.add_argument("--index-dir", type=Path, required=True)
    search.add_argument("--queries", type=Path, required=True)
    search.add_argument("--split", choices=ALL_SPLITS, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--manifest", type=Path)
    search.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    search.add_argument(
        "--branch",
        help=(
            "run label (defaults to dense_e5_frozen for E5 and "
            "dense_fixture_hash for the explicit fixture backend)"
        ),
    )
    search.add_argument("--force", action="store_true")
    _add_encoder_arguments(search)
    return parser


def _encoder_from_arguments(args: argparse.Namespace) -> DenseEncoder:
    if args.backend == "e5-trained":
        if args.checkpoint is None:
            raise DenseError("e5-trained requires --checkpoint")
        return TrainedCheckpointEncoder(args.checkpoint, device=args.device, allow_fixture=args.allow_fixture_encoder)
    if getattr(args, "checkpoint", None) is not None:
        raise DenseError("--checkpoint requires --backend e5-trained")
    if args.backend == "fixture-hash":
        return FixtureHashEncoder(
            args.fixture_dimension, allow_fixture=args.allow_fixture_encoder
        )
    if args.allow_fixture_encoder:
        raise DenseError("--allow-fixture-encoder is only valid with fixture-hash")
    return TransformersE5Encoder(
        args.model,
        args.revision,
        device=args.device,
        allow_download=args.allow_download,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = DenseConfig(max_length=args.max_length, batch_size=args.batch_size)
        config.validate()
        encoder = _encoder_from_arguments(args)
        if args.command == "build-index":
            documents = load_corpus(args.corpus)
            started = time.perf_counter()
            index = DenseIndex(documents, encoder, config)
            build_seconds = time.perf_counter() - started
            index.save(
                args.index_dir,
                overwrite=args.force,
                build_seconds=build_seconds,
                corpus_path=args.corpus,
            )
            print(
                json.dumps(
                    {
                        "index_dir": str(args.index_dir),
                        "manifest": str(args.index_dir / INDEX_MANIFEST_NAME),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.top_k <= 0:
            raise DenseError("top-k must be positive")
        index = DenseIndex.load(
            args.index_dir, encoder, expected_config=config
        )
        queries = load_dense_queries(args.queries, split=args.split)
        started = time.perf_counter()
        default_branch = _default_branch(index.manifest["encoder"])
        branch = _clean_id(args.branch or default_branch, "branch")
        results = run_queries(index, queries, top_k=args.top_k, branch=branch)
        search_seconds = time.perf_counter() - started
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "index": index.manifest,
            "index_dir": str(args.index_dir),
            "index_manifest_file_sha256": _file_sha256(args.index_dir / INDEX_MANIFEST_NAME),
            "index_vector_file_sha256": _file_sha256(args.index_dir / VECTOR_FILE_NAME),
            "queries_path": str(args.queries),
            "queries_file_sha256": _file_sha256(args.queries),
            "selected_split": args.split,
            "branch": branch,
            "top_k": args.top_k,
            "query_count": len(queries),
            "search_count": (
                sum(bool(query.text.strip()) for query in queries)
                if index.document_count else 0
            ),
            "intermediate_candidate_count": len(results),
            "result_count": len(results),
            "search_seconds": search_seconds,
            "latency_includes_view_generation": False,
            "fixture_only": bool(index.manifest["encoder"]["fixture_only"]),
            "query_encoder_operational_metadata": dict(encoder.metadata),
            "query_batch_size": config.batch_size,
        }
        write_run_artifacts(
            results,
            args.output,
            manifest,
            manifest_path,
            overwrite=args.force,
        )
    except (
        BM25Error,
        CorpusPreparationError,
        DenseError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"run": str(args.output), "manifest": str(manifest_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
