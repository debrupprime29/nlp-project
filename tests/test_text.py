from __future__ import annotations

import unittest

from nlp.text import (
    TextPreparationError,
    normalize_processing_text,
    protection_spans,
    tokenize_with_spans,
)


class TextContractTests(unittest.TestCase):
    def test_multiscript_offsets_reconstruct_nfc_text_and_preserve_marks(self):
        text = normalize_processing_text("cafe\u0301क्यों ভূমিকম্প 2026")
        tokens = tokenize_with_spans(text)
        self.assertEqual([token.text for token in tokens], ["café", "क्यों", "ভূমিকম্প", "2026"])
        self.assertEqual([token.script for token in tokens], ["latin", "devanagari", "bengali", "numeric"])
        self.assertTrue(all(text[token.start:token.end] == token.text for token in tokens))

    def test_protects_negation_dates_names_contacts_and_explicit_entities(self):
        text = "Delhi does not flood on 2026-10-16; don't ignore নয় না नहीं. email me@sample.org https://a.test"
        spans = protection_spans(text)
        protected = [(text[span.start:span.end], span.reasons) for span in spans]
        for value in ("Delhi", "not", "2026-10-16", "don't", "নয়", "না", "नहीं", "me@sample.org", "https://a.test"):
            self.assertTrue(any(value == piece for piece, _ in protected), (value, protected))
        self.assertEqual(protection_spans("delhi", [{"start": 0, "end": 5, "text": "delhi", "reason": "entity"}])[0].reasons, ("entity",))

    def test_explicit_spans_are_checked_in_nfc_coordinate_space(self):
        for spans in ([{"start": 0, "end": 9}], [{"start": True, "end": 2}], [{"start": 0, "end": 2, "text": "no"}]):
            with self.subTest(spans=spans), self.assertRaises(TextPreparationError):
                protection_spans("abc", spans)
        with self.assertRaises(TextPreparationError):
            protection_spans("cafe\u0301", [{"start": 0, "end": 4}])


if __name__ == "__main__":
    unittest.main()
