"""Exercise the real Transformers path with a random, synthetic architecture.

The local XLM-R fixture has the E5 embedding dimension but only one random
encoder layer and a tiny vocabulary. It tests pooling, autograd and persistence;
it is not pretrained E5 and provides no multilingual retrieval-quality result.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from retrieval.dense import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DenseConfig, DenseIndex, TrainedCheckpointEncoder, format_query
from tests.test_train_encoder import fixture_config, fixture_documents, fixture_qrels, fixture_records
from training.train_encoder import TransformersTrainingEncoder, prepare_encoder_training, save_trained_checkpoint, train_encoder


HAS_NEURAL_DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers", "safetensors"))


@unittest.skipUnless(HAS_NEURAL_DEPENDENCIES, "optional Torch, Transformers, Tokenizers and safetensors dependencies are unavailable")
class TransformersTrainingIntegrationTests(unittest.TestCase):
    def test_actual_transformer_autograd_masked_pooling_and_local_checkpoint_reload(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

        torch.manual_seed(31)
        vocabulary = {term: index for index, term in enumerate((
            "[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]", "query", ":", "passage",
            "earthquake", "causes", "faults", "plate", "movement", "rainfall", "clouds",
            "water", "unrelated", "history", "what", "creates", "why", "rain",
        ))}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 0), ("[SEP]", 2)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=64)
        configuration = XLMRobertaConfig(vocab_size=len(vocabulary), hidden_size=384, num_hidden_layers=1, num_attention_heads=6, intermediate_size=512, max_position_embeddings=128, pad_token_id=1, bos_token_id=0, eos_token_id=2, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        model = XLMRobertaModel(configuration)
        # Only initialization is replaced. The actual tokenizer, transformer,
        # autograd, optimizer, safetensors and local reload remain real.
        model.config._commit_hash = DEFAULT_MODEL_REVISION
        with patch("transformers.AutoModel.from_pretrained", return_value=model) as load_model, patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer:
            encoder = TransformersTrainingEncoder(device="cpu")
        load_model.assert_called_once_with(DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        load_tokenizer.assert_called_once_with(DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, local_files_only=True, trust_remote_code=False)
        self.assertEqual(encoder.dimension, 384)
        texts = [format_query("earthquake causes"), format_query("rainfall")]
        encoder.eval()
        actual = encoder.encode_training(texts, max_length=64)
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        self.assertTrue(bool((tokens["attention_mask"] == 0).any()))
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            expected = torch.nn.functional.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1), p=2, dim=1)
        self.assertEqual(tuple(actual.shape), (2, 384))
        self.assertTrue(actual.requires_grad)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(actual, dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(actual[1], encoder.encode_training([texts[1]], max_length=64)[0], atol=1e-6, rtol=1e-6))
        actual[:, 0].sum().backward()
        gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertTrue(any(bool((gradient != 0).any()) for gradient in gradients))

        config = fixture_config(epochs=2, learning_rate=0.001, max_length=64)
        data = prepare_encoder_training(fixture_records(), fixture_qrels(), fixture_documents(), config)
        before = model.embeddings.word_embeddings.weight.detach().clone()
        result = train_encoder(data, encoder, config)
        self.assertEqual(result.manifest["optimizer_updates"], 2)
        self.assertGreater(result.manifest["parameter_delta_l2"], 0)
        self.assertFalse(torch.equal(before, model.embeddings.word_embeddings.weight.detach()))
        self.assertTrue(math.isfinite(result.manifest["initial_loss"]))
        self.assertTrue(math.isfinite(result.manifest["final_loss"]))
        self.assertTrue(all(math.isfinite(step["gradient_norm"]) for step in result.history))
        self.assertEqual(result.manifest["variant_pair_count"], 1)
        result = replace(result, manifest={**result.manifest, "test_fixture": {"initialization": "random_xlm_roberta_not_pretrained_e5", "hidden_size": 384, "layers": 1, "vocabulary_size": len(vocabulary)}})
        encoder.eval()
        with torch.no_grad():
            trained_vectors = encoder.encode_training(texts, max_length=64).tolist()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "synthetic-architecture-checkpoint"
            checkpoint = save_trained_checkpoint(encoder, result, directory)
            self.assertTrue(any(name.endswith(".safetensors") for name in checkpoint["files"]))
            self.assertFalse(any(name.endswith((".bin", ".pt", ".pkl")) for name in checkpoint["files"]))
            self.assertEqual(checkpoint["metadata"]["training"]["test_fixture"]["initialization"], "random_xlm_roberta_not_pretrained_e5")
            # The original AutoModel/AutoTokenizer loaders are restored here:
            # reload the actual local model configuration and tokenizer files.
            loaded = TrainedCheckpointEncoder(directory, device="cpu")
            vectors = loaded.encode(texts, batch_size=2, max_length=64)
            self.assertTrue(torch.allclose(torch.tensor(trained_vectors), torch.tensor(vectors), atol=1e-6, rtol=1e-6))
            self.assertEqual(loaded.metadata["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
            index = DenseIndex(fixture_documents(), loaded, DenseConfig(max_length=64))
            self.assertTrue(index.manifest["fine_tuned"])
            index.save(Path(temporary) / "dense")
            reloaded = DenseIndex.load(Path(temporary) / "dense", loaded, expected_config=DenseConfig(max_length=64))
            self.assertEqual(index.search("rainfall clouds", top_k=4), reloaded.search("rainfall clouds", top_k=4))


if __name__ == "__main__":
    unittest.main()
