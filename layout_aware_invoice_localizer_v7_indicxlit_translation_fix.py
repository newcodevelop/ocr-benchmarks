#!/usr/bin/env python3
"""
Layout-aware English document localizer without a span-action classifier or LLM/VLM.

Core idea
---------
For each OCR span/line, decide one of:
    COPY / TRANSLITERATE / TRANSLATE / MIXED / ABSTAIN_COPY
using a deterministic, conservative cascade:
    1. protect identifiers/numbers/dates/amounts/codes
    2. split label-value and mixed spans
    3. use spaCy NER for entity spans
    4. translate generic document labels and sentence-like English
    5. transliterate likely names/entities
    6. abstain-copy when ambiguous

After action identification, conversion uses:
    - COPY/ABSTAIN_COPY: unchanged
    - TRANSLITERATE: AI4Bharat IndicXlit if available, otherwise mock/fallback
    - TRANSLATE: IndicTrans2 with protected placeholders
    - MIXED: recursive chunk conversion

Input OCR format
----------------
SROIE-style text files are supported by default:
    x1,y1,x2,y2,x3,y3,x4,y4,text

Example
-------
python layout_aware_invoice_localizer.py \
  --ocr-dir ./box \
  --image-dir ./img \
  --output-dir ./localized_hi \
  --target-lang hin_Deva \
  --xlit-lang hi \
  --font-path /usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf \
  --device cuda

For pipeline debugging without MT/transliteration downloads:
python layout_aware_invoice_localizer.py \
  --ocr-dir ./box --image-dir ./img --output-dir ./debug_hi \
  --mock-models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Actions and data containers
# -----------------------------------------------------------------------------


class Action(str, Enum):
    COPY = "COPY"
    TRANSLITERATE = "TRANSLITERATE"
    TRANSLATE = "TRANSLATE"
    MIXED = "MIXED"
    ABSTAIN_COPY = "ABSTAIN_COPY"


@dataclass
class Span:
    doc_id: str
    bbox8: List[int]
    text: str
    image_w: int = 1
    image_h: int = 1

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        xs = self.bbox8[0::2]
        ys = self.bbox8[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def cx(self) -> float:
        x0, _, x1, _ = self.rect
        return ((x0 + x1) / 2.0) / max(1, self.image_w)

    @property
    def cy(self) -> float:
        _, y0, _, y1 = self.rect
        return ((y0 + y1) / 2.0) / max(1, self.image_h)


@dataclass
class Decision:
    action: Action
    reason: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ConvertedSpan:
    doc_id: str
    bbox8: List[int]
    bbox_rect: List[int]
    original_text: str
    converted_text: str
    action: str
    reason: str
    confidence: float
    evidence: List[str]
    image_w: int
    image_h: int
    render_meta: Dict[str, object] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# General English document vocabulary for deciding TRANSLATE.
# This is intentionally generic, not receipt/SROIE-specific.
# -----------------------------------------------------------------------------


DOCUMENT_WORDS = {
    "account", "address", "amount", "authorized", "balance", "bank", "bill",
    "billing", "buyer", "cash", "charge", "city", "code", "company", "contact",
    "customer", "date", "description", "details", "discount", "due", "email",
    "fee", "from", "grand", "gst", "id", "invoice", "item", "method", "mobile",
    "name", "no", "number", "order", "paid", "payment", "phone", "price", "qty",
    "quantity", "rate", "receipt", "reference", "seller", "service", "shipping",
    "state", "subtotal", "supplier", "tax", "terms", "time", "tin", "to", "total",
    "transaction", "unit", "vat", "vendor", "zip", "pin", "pan", "cin",
    "summary", "operator", "trainee", "cashier", "table", "token", "user",
    "refund", "exchange", "allowed", "days", "strictly", "rounded", "rounding",
    "adjustment", "uprice", "net", "gross", "service", "sales", "person",
}

DOCUMENT_PHRASE_LEXICON_HI = {
    "invoice": "इनवॉइस",
    "receipt": "रसीद",
    "bill": "बिल",
    "date": "तारीख",
    "time": "समय",
    "invoice no": "इनवॉइस नं.",
    "invoice no.": "इनवॉइस नं.",
    "invoice number": "इनवॉइस नंबर",
    "receipt no": "रसीद नं.",
    "receipt no.": "रसीद नं.",
    "bill no": "बिल नं.",
    "order no": "ऑर्डर नं.",
    "order number": "ऑर्डर नंबर",
    "customer": "ग्राहक",
    "customer name": "ग्राहक का नाम",
    "vendor": "विक्रेता",
    "seller": "विक्रेता",
    "buyer": "खरीदार",
    "address": "पता",
    "billing address": "बिलिंग पता",
    "shipping address": "शिपिंग पता",
    "phone": "फोन",
    "mobile": "मोबाइल",
    "email": "ईमेल",
    "description": "विवरण",
    "item": "वस्तु",
    "quantity": "मात्रा",
    "qty": "मात्रा",
    "unit price": "इकाई मूल्य",
    "price": "मूल्य",
    "rate": "दर",
    "amount": "राशि",
    "total amount": "कुल राशि",
    "tax amount": "कर राशि",
    "discount amount": "छूट राशि",
    "paid amount": "भुगतान की गई राशि",
    "due amount": "देय राशि",
    "total": "कुल",
    "subtotal": "उप-कुल",
    "grand total": "कुल योग",
    "discount": "छूट",
    "tax": "कर",
    "gst": "GST",
    "vat": "VAT",
    "payment method": "भुगतान विधि",
    "balance due": "देय शेष",
    "terms and conditions": "नियम और शर्तें",
    "thank you": "धन्यवाद",
    "tax invoice": "कर इनवॉइस",
    "simplified tax invoice": "सरल कर इनवॉइस",
    "credit note": "क्रेडिट नोट",
    "creditnote": "क्रेडिट नोट",
    "cash bill": "नकद बिल",
    "cashier": "कैशियर",
    "cashier #": "कैशियर #",
    "salesperson": "सेल्सपर्सन",
    "sales person": "सेल्सपर्सन",
    "doc no": "दस्तावेज़ नं.",
    "doc no.": "दस्तावेज़ नं.",
    "ref": "संदर्भ",
    "ref.": "संदर्भ",
    "code": "कोड",
    "amt": "राशि",
    "amt rm": "राशि (RM)",
    "amt (rm)": "राशि (RM)",
    "gst summary": "GST सारांश",
    "tax code": "कर कोड",
    "gst tax": "GST/कर",
    "gst/tax": "GST/कर",
    "change": "बकाया",
    "rounding": "राउंडिंग",
    "rounding adjustment": "राउंडिंग समायोजन",
    "cash": "नकद",
    "net": "निवल",
    "gross": "सकल",
    "excluding gst": "GST छोड़कर",
    "including gst": "GST सहित",
    "sub total": "उप-कुल",
    "sub-total": "उप-कुल",
    "total qty": "कुल मात्रा",
    "total qty.": "कुल मात्रा",
    "total quantity": "कुल मात्रा",

    "summary": "सारांश",
    "ummary": "सारांश",
    "table": "टेबल",
    "token": "टोकन",
    "user": "उपयोगकर्ता",
    "operator": "ऑपरेटर",
    "trainee": "प्रशिक्षु",
    "operator trainee cashier": "प्रशिक्षु कैशियर ऑपरेटर",
    "cashier name": "कैशियर का नाम",
    "sales": "बिक्री",
    "sales invoice": "बिक्री इनवॉइस",
    "item s": "वस्तुएँ",
    "item (s)": "वस्तुएँ",
    "qty s": "मात्रा",
    "qty (s)": "मात्रा",
    "quantity s": "मात्रा",
    "u price": "इकाई मूल्य",
    "u.price": "इकाई मूल्य",
    "unit price": "इकाई मूल्य",
    "total rounded": "कुल राउंड किया गया",
    "service tax": "सेवा कर",
    "strictly no cash refund": "नकद रिफंड सख्ती से नहीं",
    "no cash refund": "नकद रिफंड नहीं",
    "cash refund": "नकद रिफंड",
    "exchange are allowed within": "बदलने की अनुमति इस अवधि के भीतर है",
    "days with receipt": "दिनों के भीतर रसीद के साथ",
    "goods sold are not returnable": "बेचा गया माल वापस नहीं होगा",
    "goods sold are not refundable": "बेचा गया माल वापस नहीं होगा",
    "goods sold are not returnable only": "बेचा गया माल वापस नहीं होगा",
    "sold are not returnable": "बेचा गया माल वापस नहीं होगा",
    "not returnable": "वापस नहीं होगा",
    "no refund": "रिफंड नहीं",
    "no exchange": "बदला नहीं जाएगा",
    "please come again": "कृपया फिर आइए",
    "thank you for shopping": "खरीदारी के लिए धन्यवाद",
    "thank you for your purchase": "खरीदारी के लिए धन्यवाद",
    "thank you for visiting us": "आने के लिए धन्यवाद",
    "for exchange within one week": "बदलने के लिए एक सप्ताह के भीतर",
}

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "to", "from", "of", "and", "or", "in", "on", "with", "by", "at",
    "this", "that", "these", "those", "please", "will", "shall", "must", "may",
    "not", "no", "once", "within", "after", "before", "your", "our",
}

COMMON_LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "corp", "corporation", "co", "company",
    "pvt", "private", "llp", "plc", "gmbh", "ag", "sa", "sarl", "sdn", "bhd",
}

COPY_ACRONYMS = {
    "GST", "VAT", "PAN", "TIN", "CIN", "IFSC", "SWIFT", "IBAN", "UPI", "SKU",
    "HSN", "SAC", "CGST", "SGST", "IGST", "USD", "INR", "EUR", "GBP", "SGD", "MYR",
    "AED", "SAR", "AUD", "CAD", "JPY", "CNY", "₹", "$", "€", "£",
}

UNIT_CODES = {
    "kg", "g", "mg", "l", "ml", "pcs", "pc", "nos", "no", "hr", "hrs", "m", "cm",
    "mm", "inch", "in", "ft", "sqft", "sqm", "gb", "mb", "kb", "tb", "w", "kw",
}


# -----------------------------------------------------------------------------
# Normalization and tokenization
# -----------------------------------------------------------------------------


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def canonical_key(text: str) -> str:
    t = normalize_space(text).lower()
    t = re.sub(r"\s+([:：])$", r"\1", t)
    t = t.strip(" \t\r\n.:;,'\"")
    return t




def compact_ocr_key(text: str) -> str:
    """Aggressive key for OCR-glued labels/sentences."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def lexicon_lookup(text: str, lexicon: Dict[str, str]) -> Optional[str]:
    """High-precision lookup for labels and common receipt/legal sentences.

    Handles normal keys, OCR-glued keys, separator variants, and frequent SROIE
    footer/legal sentence corruptions such as GOODSSOLD ARE NOTRETURNABLE.
    """
    raw = text.strip()
    if not raw:
        return None
    key = canonical_key(raw)
    if key in lexicon:
        val = lexicon[key]
        return val + (":" if raw.endswith((":", "：")) and not val.endswith((":", "：")) else "")

    spaced = re.sub(r"[/_#.-]+", " ", raw)
    key2 = canonical_key(spaced)
    if key2 in lexicon:
        val = lexicon[key2]
        return val + (":" if raw.endswith((":", "：")) and not val.endswith((":", "：")) else "")

    ck = compact_ocr_key(raw)
    compact_map = {
        "taxinvoice": "कर इनवॉइस",
        "simplifiedtaxinvoice": "सरल कर इनवॉइस",
        "creditnote": "क्रेडिट नोट",
        "invoiceno": "इनवॉइस नं.",
        "docno": "दस्तावेज़ नं.",
        "receiptno": "रसीद नं.",
        "billdate": "बिल तारीख",
        "gstsummary": "GST सारांश",
        "taxcode": "कर कोड",
        "totalqty": "कुल मात्रा",
        "qtys": "मात्रा",
        "items": "वस्तुएँ",
        "uprice": "इकाई मूल्य",
        "summary": "सारांश",
        "ummary": "सारांश",
        "totalrounded": "कुल राउंड किया गया",
        "strictlynocashrefund": "नकद रिफंड सख्ती से नहीं",
        "nocashrefund": "नकद रिफंड नहीं",
        "exchangeareallowedwithin": "बदलने की अनुमति इस अवधि के भीतर है",
        "dayswithreceipt": "दिनों के भीतर रसीद के साथ",
        "totalquantity": "कुल मात्रा",
        "subtotal": "उप-कुल",
        "grandtotal": "कुल योग",
        "goodssoldarenotreturnable": "बेचा गया माल वापस नहीं होगा",
        "goodssoldarenotreturnableonly": "बेचा गया माल वापस नहीं होगा",
        "goodssoldarenotrefundable": "बेचा गया माल वापस नहीं होगा",
        "goodsoldarenotreturnable": "बेचा गया माल वापस नहीं होगा",
        "thankyou": "धन्यवाद",
        "pleasecomeagain": "कृपया फिर आइए",
        "thankyouforyourpurchase": "खरीदारी के लिए धन्यवाद",
        "thankyouforshopping": "खरीदारी के लिए धन्यवाद",
        "thankyouforvisitingus": "आने के लिए धन्यवाद",
    }
    if ck in compact_map:
        return compact_map[ck]

    if "goodssold" in ck and ("notreturnable" in ck or "notrefundable" in ck or "returnable" in ck):
        return "बेचा गया माल वापस नहीं होगा"
    if "soldarenot" in ck and ("returnable" in ck or "refundable" in ck):
        return "बेचा गया माल वापस नहीं होगा"
    if "thank" in ck and "you" in ck:
        if "shopping" in ck or "purchase" in ck:
            return "खरीदारी के लिए धन्यवाद"
        if "visiting" in ck or "visit" in ck:
            return "आने के लिए धन्यवाद"
        return "धन्यवाद"
    if "nocashrefund" in ck or ("cashrefund" in ck and "no" in ck):
        return "नकद रिफंड नहीं"
    if "exchange" in ck and "allowed" in ck:
        return "बदलने की अनुमति है"
    if "days" in ck and "receipt" in ck:
        return "दिनों के भीतर रसीद के साथ"
    return None


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text)


def general_tokens(text: str) -> List[str]:
    """Words, protected-ish alphanumeric runs, numbers, whitespace, punctuation."""
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Z]*\d+[A-Z0-9./:_-]*|\d+(?:[.,:/-]\d+)*|\s+|[^A-Za-z\d\s]+",
        text,
    )


def is_separator(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return bool(re.fullmatch(r"[-=*_.,:/\\|~•·]+", t))


# -----------------------------------------------------------------------------
# Protected pattern detection: things that should be copied.
# -----------------------------------------------------------------------------


class ProtectedPatterns:
    def __init__(self) -> None:
        self.patterns: List[Tuple[str, re.Pattern[str]]] = [
            ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
            ("url", re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)),
            ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")),
            ("money", re.compile(r"(?<!\w)(?:[$₹€£]|INR|USD|EUR|GBP|SGD|MYR|AED|SAR|Rs\.?|RM)\s*[-+]?\d[\d,]*(?:\.\d{1,3})?\b", re.I)),
            ("date", re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")),
            ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b")),
            ("percent", re.compile(r"\b[-+]?\d+(?:\.\d+)?\s*%\b")),
            ("gstin", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", re.I)),
            ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
            ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.I)),
            ("long_id", re.compile(r"\b[A-Z]{0,6}\d{4,}[A-Z0-9./:_-]*\b", re.I)),
            ("alnum_code", re.compile(r"\b(?=[A-Z0-9./:_-]{5,}\b)(?=.*\d)(?=.*[A-Z])[A-Z0-9]+(?:[./:_-][A-Z0-9]+)+\b", re.I)),
            ("plain_number", re.compile(r"(?<!\w)[-+]?\d[\d,]*(?:\.\d+)?(?!\w)")),
        ]

    def spans(self, text: str) -> List[Tuple[int, int, str]]:
        out: List[Tuple[int, int, str]] = []
        for name, pat in self.patterns:
            for m in pat.finditer(text):
                out.append((m.start(), m.end(), name))
        out = self._merge_overlaps(out)
        return out

    @staticmethod
    def _merge_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
        merged: List[Tuple[int, int, str]] = []
        for s, e, name in spans:
            if not merged or s > merged[-1][1]:
                merged.append((s, e, name))
            else:
                ps, pe, pname = merged[-1]
                merged[-1] = (ps, max(pe, e), pname if len(pname) >= len(name) else name)
        return merged

    def is_fully_protected(self, text: str) -> bool:
        t = text.strip()
        if not t or is_separator(t):
            return True
        residual = list(t)
        for s, e, _ in self.spans(t):
            for i in range(s, e):
                residual[i] = " "
        rest = "".join(residual)
        # Ignore punctuation, whitespace, currency marks, and common unit/acronym tokens.
        rest_words = [w for w in word_tokens(rest) if w.upper() not in COPY_ACRONYMS and w.lower() not in UNIT_CODES]
        rest_nonword = re.sub(r"[\s\W_]+", "", rest, flags=re.UNICODE)
        if not rest_words and not re.search(r"[A-Za-z]", rest_nonword):
            return True
        if len(t) <= 4 and t.upper() in COPY_ACRONYMS:
            return True
        if t.lower() in UNIT_CODES:
            return True
        return False

    def has_protected_and_unprotected(self, text: str) -> bool:
        spans = self.spans(text)
        if not spans:
            return False
        residual = list(text)
        for s, e, _ in spans:
            for i in range(s, e):
                residual[i] = " "
        return bool(re.search(r"[A-Za-z]", "".join(residual)))


# -----------------------------------------------------------------------------
# spaCy NER wrapper. The rest of the pipeline still works if spaCy is missing.
# -----------------------------------------------------------------------------


class SpacyNER:
    ENTITY_TRANSLITERATE = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}
    ENTITY_COPY = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable: Sequence[str] = ("tagger", "parser", "lemmatizer", "attribute_ruler"),
        max_chars: int = 300,
    ) -> None:
        """Tiny, failure-safe spaCy wrapper.

        We only use spaCy as optional NER evidence. OCR strings can be noisy, and
        some Mac/Python/numpy/thinc combinations can crash inside spaCy's transition
        model with RecursionError. So this wrapper disables every component except
        NER when possible and, more importantly, catches runtime failures and turns
        NER off instead of killing the conversion job.
        """
        self.model_name = model_name
        self.max_chars = max_chars
        self.nlp = None
        try:
            import spacy  # type: ignore
            try:
                # `exclude` prevents unnecessary components from even loading;
                # `disable` is kept as a fallback for older spaCy versions.
                try:
                    self.nlp = spacy.load(
                        model_name,
                        exclude=["tagger", "parser", "lemmatizer", "attribute_ruler", "textcat"],
                    )
                except TypeError:
                    self.nlp = spacy.load(model_name, disable=list(disable))
                # Keep only tok2vec/transformer + ner if the model exposes pipe management.
                if hasattr(self.nlp, "pipe_names"):
                    for pipe in list(self.nlp.pipe_names):
                        if pipe not in {"tok2vec", "transformer", "ner"}:
                            try:
                                self.nlp.disable_pipe(pipe)
                            except Exception:
                                pass
                self.nlp.max_length = max(self.nlp.max_length, max_chars + 1000)
            except Exception as e:
                warnings.warn(
                    f"Could not load spaCy model {model_name!r}; NER evidence disabled. "
                    f"Install with: python -m spacy download {model_name}. Error: {e}"
                )
                self.nlp = None
        except Exception as e:
            warnings.warn(f"spaCy is not installed or failed to import; NER evidence disabled. Error: {e}")
            self.nlp = None

    def ents(self, text: str):
        if self.nlp is None or not text.strip():
            return []
        # Long OCR-glued strings are exactly where spaCy is least useful and most fragile.
        if len(text) > self.max_chars:
            return []
        try:
            return list(self.nlp(text).ents)
        except RecursionError as e:
            warnings.warn(f"spaCy NER hit RecursionError and will be disabled for this run: {e}")
            self.nlp = None
            return []
        except Exception as e:
            warnings.warn(f"spaCy NER failed on one OCR span; continuing without NER for it. Error: {e}")
            return []

    def whole_text_entity_label(self, text: str) -> Optional[str]:
        t = text.strip()
        ents = self.ents(t)
        for ent in ents:
            if ent.start_char <= 1 and ent.end_char >= len(t) - 1:
                return ent.label_
        return None

    def entity_spans(self, text: str) -> List[Tuple[int, int, str]]:
        return [(ent.start_char, ent.end_char, ent.label_) for ent in self.ents(text)]


class NoOpNER:
    ENTITY_TRANSLITERATE = SpacyNER.ENTITY_TRANSLITERATE
    ENTITY_COPY = SpacyNER.ENTITY_COPY

    def ents(self, text: str):
        return []

    def whole_text_entity_label(self, text: str) -> Optional[str]:
        return None

    def entity_spans(self, text: str) -> List[Tuple[int, int, str]]:
        return []


# -----------------------------------------------------------------------------
# Decision cascade
# -----------------------------------------------------------------------------


class ActionResolver:
    def __init__(
        self,
        ner: Optional[SpacyNER] = None,
        protected: Optional[ProtectedPatterns] = None,
        label_lexicon: Optional[Dict[str, str]] = None,
    ) -> None:
        self.ner = ner or SpacyNER()
        self.protected = protected or ProtectedPatterns()
        self.label_lexicon = label_lexicon or DOCUMENT_PHRASE_LEXICON_HI

    def decide(self, text: str, span: Optional[Span] = None) -> Decision:
        t = normalize_space(text)
        evidence: List[str] = []

        if not t or is_separator(t):
            return Decision(Action.COPY, "empty_or_separator", 1.0, ["separator"])

        # Highest priority: fixed document labels and common receipt/legal sentences.
        # This prevents MT from turning "TAX INVOICE" into things like "कर निवेश",
        # and catches OCR-glued strings such as "GOODSSOLD ARE NOTRETURNABLE".
        if lexicon_lookup(t, self.label_lexicon) is not None:
            return Decision(Action.TRANSLATE, "lexicon_label_or_sentence", 0.99, ["lexicon"])

        if self.protected.is_fully_protected(t):
            return Decision(Action.COPY, "fully_protected_pattern", 1.0, ["protected_pattern"])

        if self.is_label_value_line(t):
            return Decision(Action.MIXED, "label_value_pattern", 0.98, ["label_value"])

        if self.protected.has_protected_and_unprotected(t):
            return Decision(Action.MIXED, "protected_plus_text", 0.96, ["protected_plus_unprotected"])

        ent_label = self.ner.whole_text_entity_label(t) if self.ner else None
        if ent_label in SpacyNER.ENTITY_COPY:
            return Decision(Action.COPY, f"spacy_entity_copy_{ent_label}", 0.95, [f"NER:{ent_label}"])
        if ent_label in SpacyNER.ENTITY_TRANSLITERATE:
            # If it is also a generic document label, translation wins.
            if not self.looks_like_document_label(t):
                return Decision(Action.TRANSLITERATE, f"spacy_entity_transliterate_{ent_label}", 0.90, [f"NER:{ent_label}"])

        if self.looks_like_document_label(t):
            return Decision(Action.TRANSLATE, "generic_document_label", 0.90, ["document_vocab"])

        if self.looks_like_sentence(t):
            return Decision(Action.TRANSLATE, "sentence_like_text", 0.82, ["sentence_like"])

        if self.looks_like_proper_name(t):
            return Decision(Action.TRANSLITERATE, "proper_name_like_without_ner", 0.70, ["title_case_or_legal_suffix"])

        if self.looks_like_code_or_acronym(t):
            return Decision(Action.COPY, "code_or_acronym", 0.92, ["code_or_acronym"])

        return Decision(Action.ABSTAIN_COPY, "ambiguous_or_unknown", 0.50, ["default_abstain"])

    @staticmethod
    def is_label_value_line(text: str) -> bool:
        # Generic key-value patterns: "Label: value", "Label # value", "Label    value".
        # Do NOT treat a bare internal "NO" as a separator; that caused sentences like
        # "STRICTLY NO CASH REFUND" to be split as label/value and then mistransliterated.
        if re.match(r"^\s*[A-Za-z][A-Za-z0-9 ./&()%-]{0,45}\s*(?::|：|#)\s*\S+", text, re.I):
            return True
        if re.match(r"^\s*(?:invoice|receipt|bill|order|doc|ref|reference|customer|table|token|cashier)\s+no\.?\s*:?\s*\S+", text, re.I):
            return True
        if re.match(r"^\s*[A-Za-z][A-Za-z0-9 ./&()%-]{1,45}\s{2,}\S+", text):
            return True
        return False

    def looks_like_document_label(self, text: str) -> bool:
        if lexicon_lookup(text, self.label_lexicon) is not None:
            return True
        key = canonical_key(text)
        words = [w.lower() for w in word_tokens(text)]
        if not words or len(words) > 6:
            return False
        if any(w in DOCUMENT_WORDS for w in words):
            # Avoid treating company-like phrases as labels.
            if any(w in COMMON_LEGAL_SUFFIXES for w in words):
                return False
            # Labels rarely contain many digits.
            digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
            return digit_ratio < 0.25
        return False

    @staticmethod
    def looks_like_sentence(text: str) -> bool:
        words = [w.lower() for w in word_tokens(text)]
        if len(words) < 4:
            return False
        if sum(ch.isdigit() for ch in text) / max(1, len(text)) > 0.30:
            return False
        function_hits = sum(w in FUNCTION_WORDS for w in words)
        has_sentence_punct = bool(re.search(r"[.!?]$", text.strip()))
        has_verbish = bool(re.search(r"\b(is|are|was|were|be|been|being|sold|paid|due|received|provided|return(?:ed|able)?|exchange(?:d|able)?|retain|contact|visit|thank)\b", text, re.I))
        return function_hits >= 1 or has_sentence_punct or has_verbish

    @staticmethod
    def looks_like_proper_name(text: str) -> bool:
        words = word_tokens(text)
        if not words or len(words) > 7:
            return False
        lower = [w.lower() for w in words]
        # Single OCR noise/acronym tokens should not be treated as names.
        if len(words) == 1:
            up0 = words[0].upper()
            if up0 in COPY_ACRONYMS or up0 in ENTITY_ACRONYM_COPY or len(up0) <= 2:
                return False
        if any(w in DOCUMENT_WORDS for w in lower):
            return False
        if any(w in COMMON_LEGAL_SUFFIXES for w in lower):
            return True
        titleish = sum(1 for w in words if w[:1].isupper() and not w.isupper())
        if titleish >= max(1, math.ceil(len(words) * 0.6)):
            return True
        # All-caps names are common in OCR, but only trust if short and alphabetic.
        if len(words) <= 4 and text.upper() == text and sum(ch.isdigit() for ch in text) == 0:
            return True
        return False

    @staticmethod
    def looks_like_code_or_acronym(text: str) -> bool:
        t = text.strip()
        if t.upper() in COPY_ACRONYMS:
            return True
        if re.fullmatch(r"[A-Z]{1,5}", t):
            return True
        if re.fullmatch(r"[A-Z0-9./:_-]{4,}", t, re.I) and any(ch.isdigit() for ch in t):
            return True
        return False


# -----------------------------------------------------------------------------
# Transliteration and translation backends
# -----------------------------------------------------------------------------


class MockTranslator:
    def __init__(self, label_lexicon: Optional[Dict[str, str]] = None) -> None:
        self.label_lexicon = label_lexicon or {}

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        out = []
        for t in texts:
            val = lexicon_lookup(t, self.label_lexicon)
            out.append(val if val is not None else f"[TGT:{t}]")
        return out


class IndicTrans2Translator:
    def __init__(
        self,
        model_name: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
        device: str = "cpu",
        max_length: int = 256,
        label_lexicon: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        self.max_length = max_length
        self.cache: Dict[str, str] = {}
        self.label_lexicon = label_lexicon or {}

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Install translation dependencies: torch transformers sentencepiece sacremoses.") from e

        try:
            try:
                from IndicTransToolkit.processor import IndicProcessor  # type: ignore
            except Exception:
                from IndicTransToolkit import IndicProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Could not import IndicTransToolkit. Install AI4Bharat IndicTrans2 toolkit, "
                "or use --mock-models for debugging."
            ) from e

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.IndicProcessor = IndicProcessor
        self.ip = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        results: List[Optional[str]] = [None] * len(texts)
        missing: List[str] = []
        missing_idx: List[int] = []

        for i, src in enumerate(texts):
            src_norm = src.strip()
            key = canonical_key(src_norm)
            if key in self.label_lexicon:
                results[i] = self.label_lexicon[key]
            elif src_norm in self.cache:
                results[i] = self.cache[src_norm]
            else:
                missing.append(src_norm)
                missing_idx.append(i)

        if missing:
            batch = self.ip.preprocess_batch(missing, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
            inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    use_cache=False,
                    min_length=0,
                    max_length=self.max_length,
                    num_beams=5,
                    num_return_sequences=1,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated = self.ip.postprocess_batch(decoded, lang=self.tgt_lang)
            for idx, src, tgt in zip(missing_idx, missing, translated):
                tgt = trim_translation(tgt)
                self.cache[src] = tgt
                results[idx] = tgt

        return [r or "" for r in results]


class CopyTransliterator:
    """No-op transliterator. Useful when the target language has no rule fallback."""

    def transliterate_word(self, word: str) -> str:
        return word

    def transliterate_phrase(self, text: str) -> str:
        return text


class RuleBasedHindiTransliterator:
    """Small dependency-free Latin→Devanagari fallback.

    This is not as good as IndicXlit, but it removes the hard dependency on the
    transliteration model. It is intentionally conservative: short acronyms,
    IDs, units, and alphanumeric codes are copied by should_copy_token().
    """

    COMMON = {
        "home": "होम", "master": "मास्टर", "hardware": "हार्डवेयर",
        "electrical": "इलेक्ट्रिकल", "electric": "इलेक्ट्रिक",
        "trading": "ट्रेडिंग", "gallery": "गैलरी", "lightroom": "लाइटरूम",
        "light": "लाइट", "room": "रूम", "sam": "सैम", "angel": "एंजेल",
        "angela": "एंजेला", "station": "स्टेशन", "cashier": "कैशियर",
        "jalan": "जलान", "jln": "जलान", "bandar": "बंदर", "bukit": "बुकित",
        "raja": "राजा", "setia": "सेतिया", "alam": "आलम", "klang": "क्लांग",
        "selangor": "सेलंगोर", "malaysia": "मलेशिया", "kuala": "कुआला",
        "lumpur": "लुम्पुर", "sri": "श्री", "muda": "मुदा", "shah": "शाह",
        "taman": "तमन", "park": "पार्क", "street": "स्ट्रीट", "road": "रोड",
        "shop": "शॉप", "store": "स्टोर", "restaurant": "रेस्टोरेंट",
        "kitchen": "किचन", "mobile": "मोबाइल", "fax": "फैक्स", "tel": "टेल",
        "company": "कंपनी", "reg": "रजि", "no": "नं", "invoice": "इनवॉइस",
        "credit": "क्रेडिट", "note": "नोट", "bill": "बिल", "date": "डेट",
        "cover": "कवर", "name": "नेम", "bank": "बैंक", "cash": "कैश",
    }

    # Longest-first replacements for English-ish chunks.
    CHUNKS = [
        ("tion", "शन"), ("sion", "शन"), ("ing", "िंग"), ("ware", "वेयर"),
        ("ght", "ट"), ("ck", "क"), ("ph", "फ"), ("sh", "श"), ("ch", "च"),
        ("th", "थ"), ("dh", "ध"), ("kh", "ख"), ("gh", "घ"), ("bh", "भ"),
        ("ai", "ै"), ("ay", "े"), ("ee", "ी"), ("ea", "ी"), ("oo", "ू"),
        ("ou", "ाउ"), ("ow", "ाउ"), ("au", "ौ"), ("oi", "ॉइ"),
    ]
    VOWELS_INDEP = {
        "a": "अ", "e": "ए", "i": "इ", "o": "ओ", "u": "उ",
    }
    VOWEL_SIGNS = {
        "a": "", "e": "े", "i": "ि", "o": "ो", "u": "ु",
    }
    CONS = {
        "b": "ब", "c": "क", "d": "द", "f": "फ", "g": "ग", "h": "ह",
        "j": "ज", "k": "क", "l": "ल", "m": "म", "n": "न", "p": "प",
        "q": "क", "r": "र", "s": "स", "t": "ट", "v": "व", "w": "व",
        "x": "क्स", "y": "य", "z": "ज़",
    }

    def __init__(self) -> None:
        self.cache: Dict[str, str] = {}

    def transliterate_word(self, word: str) -> str:
        if not word:
            return word
        if word in self.cache:
            return self.cache[word]
        if should_copy_token(word):
            self.cache[word] = word
            return word

        lower = word.lower().strip("'")
        if lower in self.COMMON:
            out = self.COMMON[lower]
            self.cache[word] = out
            return out

        out = self._transliterate_ascii_word(lower)
        # Preserve original if rule output is basically empty or too weird.
        if not out:
            out = word
        self.cache[word] = out
        return out

    def _transliterate_ascii_word(self, w: str) -> str:
        # A compact syllable approximation: consonant + following vowel sign,
        # standalone vowels, plus a few chunk replacements.
        out: List[str] = []
        i = 0
        while i < len(w):
            ch = w[i]
            if not ch.isalpha():
                out.append(ch)
                i += 1
                continue

            # Try consonant clusters / common chunks first.
            matched = False
            for src, dst in self.CHUNKS:
                if w.startswith(src, i):
                    out.append(dst)
                    i += len(src)
                    matched = True
                    break
            if matched:
                continue

            if ch in self.CONS:
                base = self.CONS[ch]
                # Attach a vowel sign if the next char is a simple vowel.
                if i + 1 < len(w) and w[i + 1] in self.VOWEL_SIGNS:
                    out.append(base + self.VOWEL_SIGNS[w[i + 1]])
                    i += 2
                else:
                    out.append(base)
                    i += 1
                continue

            if ch in self.VOWELS_INDEP:
                out.append(self.VOWELS_INDEP[ch])
                i += 1
                continue

            out.append(ch)
            i += 1
        return "".join(out)

    def transliterate_phrase(self, text: str) -> str:
        return transliterate_phrase_tokenwise(text, self, force_alpha=True)


class MockTransliterator:
    def transliterate_word(self, word: str) -> str:
        return f"‹{word}›"

    def transliterate_phrase(self, text: str) -> str:
        return transliterate_phrase_tokenwise(text, self, force_alpha=True)


class IndicXlitTransliterator:
    def __init__(self, lang_code: str = "hi", beam_width: int = 10) -> None:
        self.lang_code = lang_code
        self.cache: Dict[str, str] = {}
        self.engine = None
        self.fallback = RuleBasedHindiTransliterator() if lang_code == "hi" else CopyTransliterator()
        self.backend_status = "indicxlit_uninitialized"
        try:
            from ai4bharat.transliteration import XlitEngine  # type: ignore
            try:
                self.engine = XlitEngine(lang2use=lang_code, beam_width=beam_width, rescore=False)
            except TypeError:
                try:
                    self.engine = XlitEngine(lang2use=lang_code, beam_width=beam_width)
                except TypeError:
                    self.engine = XlitEngine(lang_code, beam_width=beam_width)
            self.backend_status = "indicxlit"
            print(f"[IndicXlit] initialized successfully for lang={lang_code}")
        except Exception as e:
            self.backend_status = "rule_fallback" if lang_code == "hi" else "copy_fallback"
            warnings.warn(
                "ai4bharat-transliteration failed to initialize; falling back to "
                f"{self.backend_status}. Original error: {type(e).__name__}: {e}"
            )

    def transliterate_word(self, word: str) -> str:
        if not word:
            return word
        if word in self.cache:
            return self.cache[word]
        # Use the RELAXED transliteration-span policy here. The old code used
        # should_copy_token(), which copies all short all-caps words; that is why
        # TAN / WOON / YANN stayed verbatim even when the action was TRANSLITERATE.
        if should_copy_token_in_transliterate_span(word):
            self.cache[word] = word
            return word
        if self.engine is None:
            val = self.fallback.transliterate_word(word)
            self.cache[word] = val
            return val
        # IndicXlit behaves much better on receipt OCR all-caps names when lowercased.
        src = word.lower() if word.isupper() else word
        try:
            try:
                out = self.engine.translit_word(src, topk=1)
            except TypeError:
                out = self.engine.translit_word(src)
            val = self._extract_top(out, word)
        except Exception as e:
            warnings.warn(f"IndicXlit failed for token {word!r}: {type(e).__name__}: {e}; using rule fallback.")
            val = self.fallback.transliterate_word(word)
        self.cache[word] = val
        return val

    def _extract_top(self, out, fallback: str) -> str:
        if isinstance(out, dict):
            if self.lang_code in out and out[self.lang_code]:
                return str(out[self.lang_code][0])
            for v in out.values():
                if isinstance(v, list) and v:
                    return str(v[0])
                if isinstance(v, str):
                    return v
        if isinstance(out, list) and out:
            return str(out[0])
        if isinstance(out, str):
            return out
        return fallback

    def transliterate_phrase(self, text: str) -> str:
        return transliterate_phrase_tokenwise(text, self, force_alpha=True)


def build_transliterator(backend: str, target_lang: str, xlit_lang: str):
    """Build transliterator without requiring IndicXlit by default."""
    backend = backend.lower()
    if backend == "copy":
        return CopyTransliterator()
    if backend == "mock":
        return MockTransliterator()
    if backend == "indicxlit":
        return IndicXlitTransliterator(lang_code=xlit_lang)
    if backend == "rule":
        if target_lang == "hin_Deva" or xlit_lang == "hi":
            return RuleBasedHindiTransliterator()
        warnings.warn(
            f"Rule-based transliteration is currently implemented only for Hindi/Devanagari. "
            f"Using copy fallback for target_lang={target_lang!r}, xlit_lang={xlit_lang!r}."
        )
        return CopyTransliterator()
    raise ValueError(f"Unknown transliteration backend: {backend}")


# These are genuinely acronym/code-like tokens that should stay Latin even inside
# a TRANSLITERATE span.  Do NOT use a blanket "all uppercase <= 4 chars" rule here,
# because many merchant/person names in OCR are all-caps short words, e.g.
# TAN WOON YANN.  The older code copied all of those verbatim.
ENTITY_ACRONYM_COPY = {
    "MR", "MRS", "MS", "DR", "CO", "CORP", "INC", "LTD", "LLC", "LLP", "PVT",
    "SDN", "BHD", "PTE", "PLC", "LL", "KG", "ML", "PCS", "PC",
    "GST", "VAT", "SST", "PAN", "TIN", "CIN", "TEL", "FAX", "RM",
}


def should_copy_token(tok: str) -> bool:
    """Strict copy policy for protected/mixed spans.

    This is intentionally conservative for product lines and IDs.  It is NOT used
    unchanged for full TRANSLITERATE spans, because all-caps names would otherwise
    be copied instead of transliterated.
    """
    key = tok.strip()
    if not key:
        return True
    if key.upper() in COPY_ACRONYMS:
        return True
    if key.lower() in UNIT_CODES:
        return True
    if len(key) <= 4 and key.upper() == key and re.fullmatch(r"[A-Z]+", key):
        return True
    if re.fullmatch(r"[A-Z]*\d+[A-Z0-9./:_-]*", key, re.I):
        return True
    return False


def should_copy_token_in_transliterate_span(tok: str) -> bool:
    """Looser copy policy for spans whose *whole action* is TRANSLITERATE.

    Here, short all-caps alphabetic words should usually be names/places and should
    be transliterated, not copied.  We only copy true acronyms/legal suffixes,
    single-letter initials, units/currency, and alphanumeric codes.
    """
    key = tok.strip()
    if not key:
        return True
    up = key.upper()

    if up in ENTITY_ACRONYM_COPY or up in COPY_ACRONYMS:
        return True
    if key.lower() in UNIT_CODES:
        return True
    if len(key) == 1 and key.isalpha():
        return True
    if re.fullmatch(r"[A-Z]*\d+[A-Z0-9./:_-]*", key, re.I):
        return True
    return False


def transliterate_phrase_tokenwise(text: str, transliterator, force_alpha: bool = False) -> str:
    parts: List[str] = []
    for tok in general_tokens(text):
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok):
            copy_it = should_copy_token_in_transliterate_span(tok) if force_alpha else should_copy_token(tok)
            parts.append(tok if copy_it else transliterator.transliterate_word(tok))
        else:
            parts.append(tok)
    return "".join(parts)


# -----------------------------------------------------------------------------
# Masked conversion
# -----------------------------------------------------------------------------


def trim_translation(text: str) -> str:
    t = text.strip()
    t = re.split(r"[.·]{8,}|[-_=]{8,}", t, maxsplit=1)[0].strip()
    return re.sub(r"\s+", " ", t)


class MaskedConverter:
    def __init__(
        self,
        resolver: ActionResolver,
        translator,
        transliterator,
        label_lexicon: Optional[Dict[str, str]] = None,
    ) -> None:
        self.resolver = resolver
        self.translator = translator
        self.transliterator = transliterator
        self.label_lexicon = label_lexicon or {}
        self.protected = resolver.protected
        self.ner = resolver.ner

    def convert_span(self, span: Span) -> ConvertedSpan:
        decision = self.resolver.decide(span.text, span)
        converted = self.convert_text(span.text, decision=decision, depth=0)
        return ConvertedSpan(
            doc_id=span.doc_id,
            bbox8=span.bbox8,
            bbox_rect=list(span.rect),
            original_text=span.text,
            converted_text=converted,
            action=decision.action.value,
            reason=decision.reason,
            confidence=decision.confidence,
            evidence=decision.evidence,
            image_w=span.image_w,
            image_h=span.image_h,
        )

    def convert_text(self, text: str, decision: Optional[Decision] = None, depth: int = 0) -> str:
        if depth > 3:
            return text
        t = text.strip()
        decision = decision or self.resolver.decide(t)

        if decision.action in {Action.COPY, Action.ABSTAIN_COPY}:
            return t
        if decision.action == Action.TRANSLITERATE:
            return self.transliterator.transliterate_phrase(t)
        if decision.action == Action.TRANSLATE:
            return self.translate_with_masks(t)
        if decision.action == Action.MIXED:
            return self.convert_mixed(t, depth=depth + 1)
        return t

    def convert_mixed(self, text: str, depth: int = 1) -> str:
        # First handle obvious label-value lines.
        split = split_label_value(text)
        if split is not None:
            label, sep, value = split
            label_out = self.convert_text(label, self.resolver.decide(label), depth=depth)
            value_out = self.convert_value(value, depth=depth)
            return f"{label_out}{sep}{value_out}"

        # Otherwise chunk by protected spans and NER spans, then recursively convert residual text.
        masks = self.collect_mask_spans(text)
        if not masks:
            # Fall back to token-wise conservative conversion.
            return self.convert_tokenwise_mixed(text, depth=depth)

        pieces: List[str] = []
        cur = 0
        for s, e, kind in masks:
            if s > cur:
                residual = text[cur:s]
                pieces.append(self.convert_residual_piece(residual, depth=depth))
            chunk = text[s:e]
            if kind.startswith("NER:"):
                label = kind.split(":", 1)[1]
                if label in SpacyNER.ENTITY_TRANSLITERATE:
                    pieces.append(self.transliterator.transliterate_phrase(chunk))
                else:
                    pieces.append(chunk)
            else:
                pieces.append(chunk)
            cur = e
        if cur < len(text):
            pieces.append(self.convert_residual_piece(text[cur:], depth=depth))
        return "".join(pieces).strip()

    def convert_value(self, value: str, depth: int = 1) -> str:
        d = self.resolver.decide(value)
        if d.action == Action.TRANSLATE:
            # Values are often names/product/address-like. Be conservative.
            ent = self.ner.whole_text_entity_label(value) if self.ner else None
            if ent in SpacyNER.ENTITY_TRANSLITERATE:
                return self.transliterator.transliterate_phrase(value)
        return self.convert_text(value, d, depth=depth)

    def convert_residual_piece(self, text: str, depth: int = 1) -> str:
        if not text.strip():
            return text
        d = self.resolver.decide(text)
        if d.action == Action.ABSTAIN_COPY:
            # In mixed spans, residual label-ish words should still translate.
            if self.resolver.looks_like_document_label(text):
                return self.translate_with_masks(text)
            return text
        return self.convert_text(text, d, depth=depth)

    def convert_tokenwise_mixed(self, text: str, depth: int = 1) -> str:
        parts: List[str] = []
        buffer: List[str] = []

        def flush_buffer():
            if not buffer:
                return
            phrase = "".join(buffer)
            buffer.clear()
            parts.append(self.convert_residual_piece(phrase, depth=depth))

        for tok in general_tokens(text):
            if not tok.strip():
                buffer.append(tok)
            elif should_copy_token(tok) or self.protected.is_fully_protected(tok):
                flush_buffer()
                parts.append(tok)
            else:
                buffer.append(tok)
        flush_buffer()
        return "".join(parts).strip()

    def translate_with_masks(self, text: str) -> str:
        # Lexicon shortcut for generic labels. Hindi built-in is used only when target is Hindi or user supplied it.
        val = lexicon_lookup(text, self.label_lexicon)
        if val is not None:
            return val

        masked, placeholder_map = self.mask_protected_and_entities(text)
        try:
            out = self.translator.translate_many([masked])[0]
            out = trim_translation(out)
        except Exception as e:
            warnings.warn(f"Translation failed for {text!r}: {e}. Falling back to mixed conversion.")
            return self.convert_tokenwise_mixed(text)

        # Exact unmask. If placeholders were mangled by MT, fall back conservatively.
        missing = [ph for ph in placeholder_map if ph not in out]
        if missing:
            return self.convert_tokenwise_mixed(text)
        for ph, val in placeholder_map.items():
            out = out.replace(ph, val)
        return out

    def mask_protected_and_entities(self, text: str) -> Tuple[str, Dict[str, str]]:
        spans = self.collect_mask_spans(text)
        if not spans:
            return text, {}
        placeholders: Dict[str, str] = {}
        out: List[str] = []
        cur = 0
        idx = 0
        for s, e, kind in spans:
            if s < cur:
                continue
            out.append(text[cur:s])
            ph = f"__PH{idx}__"
            raw = text[s:e]
            if kind.startswith("NER:"):
                label = kind.split(":", 1)[1]
                if label in SpacyNER.ENTITY_TRANSLITERATE:
                    val = self.transliterator.transliterate_phrase(raw)
                else:
                    val = raw
            else:
                val = raw
            placeholders[ph] = val
            out.append(ph)
            cur = e
            idx += 1
        out.append(text[cur:])
        return "".join(out), placeholders

    def collect_mask_spans(self, text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        spans.extend(self.protected.spans(text))
        if self.ner:
            for s, e, label in self.ner.entity_spans(text):
                if label in SpacyNER.ENTITY_TRANSLITERATE or label in SpacyNER.ENTITY_COPY:
                    spans.append((s, e, f"NER:{label}"))
        return merge_labeled_spans(spans)


def merge_labeled_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not spans:
        return []
    # Prefer longer/earlier spans; protected spans and NER spans are both masks.
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[Tuple[int, int, str]] = []
    for s, e, label in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e, label))
        else:
            ps, pe, plabel = merged[-1]
            if e > pe:
                merged[-1] = (ps, e, plabel)
    return merged


def split_label_value(text: str) -> Optional[Tuple[str, str, str]]:
    # "Invoice No: INV-123", "Date    12/05/2024", "Customer # C001".
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ./&()%-]{0,45}?)(\s*(?:[:：]|#)\s*)(.+?)\s*$", text)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3).strip()
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ./&()%-]{1,45}?)(\s{2,})(.+?)\s*$", text)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3).strip()
    return None


# -----------------------------------------------------------------------------
# OCR IO
# -----------------------------------------------------------------------------


def parse_sroie_ocr_file(path: Path, image_w: int = 1, image_h: int = 1) -> List[Span]:
    spans: List[Span] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n\r")
            if not raw.strip():
                continue
            parts = raw.split(",", 8)
            if len(parts) < 9:
                warnings.warn(f"Skipping malformed line {line_no} in {path}: {raw[:100]}")
                continue
            try:
                coords = [int(round(float(x))) for x in parts[:8]]
            except ValueError:
                warnings.warn(f"Skipping non-numeric bbox line {line_no} in {path}: {raw[:100]}")
                continue
            text = parts[8].strip()
            spans.append(Span(doc_id=path.stem, bbox8=coords, text=text, image_w=image_w, image_h=image_h))
    return spans


def find_ocr_files(ocr_dir: Path) -> List[Path]:
    if ocr_dir.is_file():
        return [ocr_dir]
    files = sorted(p for p in ocr_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt OCR files found under {ocr_dir}")
    return files


def find_image_for_stem(image_dir: Optional[Path], stem: str) -> Optional[Path]:
    if image_dir is None or not image_dir.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for ext in exts:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in image_dir.rglob("*"):
        if p.is_file() and p.stem == stem and p.suffix.lower() in exts:
            return p
    return None


def load_spans_for_file(ocr_file: Path, image_dir: Optional[Path]) -> Tuple[List[Span], Tuple[int, int], Optional[Path]]:
    img_path = find_image_for_stem(image_dir, ocr_file.stem)
    if img_path:
        with Image.open(img_path) as im:
            w, h = im.size
    else:
        tmp = parse_sroie_ocr_file(ocr_file, 1, 1)
        max_x = max([max(s.bbox8[0::2]) for s in tmp] + [100])
        max_y = max([max(s.bbox8[1::2]) for s in tmp] + [100])
        w, h = int(max_x + 20), int(max_y + 20)
    return parse_sroie_ocr_file(ocr_file, w, h), (w, h), img_path


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Kohinoor Devanagari.ttc",
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Supplemental/Kohinoor Devanagari.ttc",
    "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
]

FALLBACK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def resolve_font_path(font_path: Optional[str]) -> str:
    if font_path:
        p = Path(font_path)
        if not p.exists():
            raise FileNotFoundError(f"Font path does not exist: {font_path}")
        return str(p)
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("No suitable font found automatically. Pass --font-path.")


def resolve_fallback_font_path(font_path: Optional[str], primary: str) -> str:
    # For mixed Latin+Devanagari receipts, using a different fallback font often
    # creates ugly baseline/size mismatches. Default to one font; only use a
    # separate fallback when the user explicitly passes --fallback-font-path.
    if font_path:
        p = Path(font_path)
        if not p.exists():
            raise FileNotFoundError(f"Fallback font path does not exist: {font_path}")
        return str(p)
    return primary


def use_fallback_font_for_char(ch: str) -> bool:
    # Keep ASCII IDs/codes visually stable using fallback Latin font.
    return ord(ch) < 128


def font_runs(text: str) -> List[Tuple[bool, str]]:
    runs: List[Tuple[bool, str]] = []
    cur_flag: Optional[bool] = None
    cur: List[str] = []
    for ch in text:
        flag = use_fallback_font_for_char(ch)
        if ch.isspace() and cur_flag is not None:
            flag = cur_flag
        if cur and flag != cur_flag:
            runs.append((bool(cur_flag), "".join(cur)))
            cur = []
        cur.append(ch)
        cur_flag = flag
    if cur:
        runs.append((bool(cur_flag), "".join(cur)))
    return runs


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def text_bbox_mixed(draw: ImageDraw.ImageDraw, text: str, target_font: ImageFont.FreeTypeFont, fallback_font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    widths: List[int] = []
    heights: List[int] = []
    for line in text.split("\n"):
        line_w = 0
        line_h = 0
        for fallback, run in font_runs(line):
            font = fallback_font if fallback else target_font
            rw, rh = text_bbox(draw, run, font)
            line_w += rw
            line_h = max(line_h, rh)
        widths.append(line_w)
        heights.append(line_h)
    return max(widths or [0]), sum(heights or [0])


def draw_text_mixed(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, target_font: ImageFont.FreeTypeFont, fallback_font: ImageFont.FreeTypeFont, fill: int = 0) -> None:
    x0, y = xy
    for line in text.split("\n"):
        x = x0
        line_h = 0
        for fallback, run in font_runs(line):
            font = fallback_font if fallback else target_font
            rw, rh = text_bbox(draw, run, font)
            draw.text((x, y), run, font=font, fill=fill)
            x += rw
            line_h = max(line_h, rh)
        y += line_h


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, target_font: ImageFont.FreeTypeFont, fallback_font: ImageFont.FreeTypeFont, max_w: int) -> str:
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    lines: List[str] = []
    cur = ""
    for tok in tokens:
        cand = tok if not cur else cur + " " + tok
        w, _ = text_bbox_mixed(draw, cand, target_font, fallback_font)
        if w <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    rect: Tuple[int, int, int, int],
    font_path: str,
    fallback_font_path: str,
    min_font_size: int = 7,
    allow_wrap: bool = True,
    font_scale: float = 1.0,
) -> Dict[str, object]:
    x0, y0, x1, y1 = rect
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    start_size = max(min_font_size, int(round(box_h * font_scale)))

    for size in range(start_size, min_font_size - 1, -1):
        target_font = ImageFont.truetype(font_path, size=size)
        fallback_font = ImageFont.truetype(fallback_font_path, size=size)
        tw, th = text_bbox_mixed(draw, text, target_font, fallback_font)
        if tw <= box_w and th <= box_h:
            y = y0 + max(0, (box_h - th) // 2)
            draw_text_mixed(draw, (x0, y), text, target_font, fallback_font)
            return {"font_size": size, "fit_policy": "same_box_shrink", "overflow": False, "rendered_rect": [x0, y, x0 + tw, y + th]}

    if allow_wrap:
        for size in range(start_size, min_font_size - 1, -1):
            target_font = ImageFont.truetype(font_path, size=size)
            fallback_font = ImageFont.truetype(fallback_font_path, size=size)
            wrapped = wrap_text_to_width(draw, text, target_font, fallback_font, box_w)
            tw, th = text_bbox_mixed(draw, wrapped, target_font, fallback_font)
            if tw <= box_w and th <= box_h:
                y = y0 + max(0, (box_h - th) // 2)
                draw_text_mixed(draw, (x0, y), wrapped, target_font, fallback_font)
                return {"font_size": size, "fit_policy": "wrapped", "overflow": False, "rendered_rect": [x0, y, x0 + tw, y + th]}

    size = min_font_size
    target_font = ImageFont.truetype(font_path, size=size)
    fallback_font = ImageFont.truetype(fallback_font_path, size=size)
    tw, th = text_bbox_mixed(draw, text, target_font, fallback_font)
    draw_text_mixed(draw, (x0, y0), text, target_font, fallback_font)
    return {"font_size": size, "fit_policy": "overflow_min_font", "overflow": True, "rendered_rect": [x0, y0, x0 + tw, y0 + th]}


def render_document(records: List[ConvertedSpan], size: Tuple[int, int], font_path: str, fallback_font_path: str, out_path: Path, min_font_size: int, allow_wrap: bool, font_scale: float) -> List[ConvertedSpan]:
    canvas = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(canvas)
    enriched: List[ConvertedSpan] = []
    for rec in records:
        rect = tuple(int(v) for v in rec.bbox_rect)
        span_allow_wrap = allow_wrap and rec.action in {Action.TRANSLATE.value, Action.TRANSLITERATE.value, Action.MIXED.value}
        meta = draw_fitted_text(
            draw,
            rec.converted_text,
            rect,
            font_path,
            fallback_font_path,
            min_font_size=min_font_size,
            allow_wrap=span_allow_wrap,
            font_scale=font_scale,
        )
        rec.render_meta = meta
        enriched.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return enriched


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------


def save_sroie_style_txt(records: List[ConvertedSpan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            coords = ",".join(str(int(x)) for x in rec.bbox8)
            f.write(f"{coords},{rec.converted_text}\n")


def rec_to_dict(rec: ConvertedSpan) -> Dict[str, object]:
    return {
        "doc_id": rec.doc_id,
        "bbox8": rec.bbox8,
        "bbox_rect": rec.bbox_rect,
        "original_text": rec.original_text,
        "converted_text": rec.converted_text,
        "action": rec.action,
        "reason": rec.reason,
        "confidence": rec.confidence,
        "evidence": rec.evidence,
        "image_w": rec.image_w,
        "image_h": rec.image_h,
        **rec.render_meta,
    }


def save_json(records: List[ConvertedSpan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([rec_to_dict(r) for r in records], f, ensure_ascii=False, indent=2)


def load_label_lexicon(path: Optional[str], target_lang: str) -> Dict[str, str]:
    lexicon: Dict[str, str] = {}
    if target_lang == "hin_Deva":
        lexicon.update(DOCUMENT_PHRASE_LEXICON_HI)
    if path:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            user_lex = json.load(f)
        # Accept either {"invoice": "..."} or {"hin_Deva": {"invoice": "..."}}
        if target_lang in user_lex and isinstance(user_lex[target_lang], dict):
            user_lex = user_lex[target_lang]
        for k, v in user_lex.items():
            lexicon[canonical_key(k)] = str(v)
    return {canonical_key(k): v for k, v in lexicon.items()}


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Classifier-free, layout-aware English document localizer.")
    ap.add_argument("--ocr-dir", type=str, required=True, help="SROIE-style OCR txt file or directory of txt files.")
    ap.add_argument("--image-dir", type=str, default=None, help="Optional image directory. Used for canvas size.")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional number of documents to process.")
    ap.add_argument("--seed", type=int, default=42)

    # Source is fixed to English by design, but keeping src-lang visible helps IndicTrans.
    ap.add_argument("--src-lang", type=str, default="eng_Latn", help="Source language code for IndicTrans. Default: English.")
    ap.add_argument("--target-lang", type=str, default="hin_Deva", help="IndicTrans target code, e.g. hin_Deva, ben_Beng, tam_Taml.")
    ap.add_argument("--xlit-lang", type=str, default="hi", help="IndicXlit language code, e.g. hi, bn, ta.")
    ap.add_argument("--translation-model", type=str, default="ai4bharat/indictrans2-en-indic-1B")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mock-models", action="store_true", help="Use mock translation and mock transliteration for debugging.")
    ap.add_argument(
        "--transliteration-backend",
        type=str,
        default="rule",
        choices=["copy", "rule", "indicxlit", "mock"],
        help=(
            "Transliteration backend. Default 'rule' gives dependency-free Hindi-script transliteration without requiring any model. Use 'copy' to preserve names/brands/addresses verbatim. "
            "Use 'rule' for a dependency-free rough Hindi transliteration, or 'indicxlit' only if installed."
        ),
    )
    ap.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    ap.add_argument("--no-spacy", action="store_true", help="Disable spaCy NER evidence.")
    ap.add_argument("--label-lexicon", type=str, default=None, help="Optional JSON phrase lexicon for target labels.")

    ap.add_argument("--font-path", type=str, default=None)
    ap.add_argument("--fallback-font-path", type=str, default=None)
    ap.add_argument("--min-font-size", type=int, default=7)
    ap.add_argument("--font-scale", type=float, default=0.82)
    ap.add_argument("--no-wrap", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    ocr_path = Path(args.ocr_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    out_dir = Path(args.output_dir)
    out_img_dir = out_dir / "images"
    out_txt_dir = out_dir / "bbox_txt"
    out_json_dir = out_dir / "metadata"

    label_lexicon = load_label_lexicon(args.label_lexicon, args.target_lang)

    ner = NoOpNER() if args.no_spacy else SpacyNER(args.spacy_model)
    protected = ProtectedPatterns()
    resolver = ActionResolver(ner=ner, protected=protected, label_lexicon=label_lexicon)

    if args.mock_models:
        translator = MockTranslator(label_lexicon=label_lexicon)
        transliterator = build_transliterator(args.transliteration_backend, args.target_lang, args.xlit_lang)
    else:
        translator = IndicTrans2Translator(
            model_name=args.translation_model,
            src_lang=args.src_lang,
            tgt_lang=args.target_lang,
            device=args.device,
            label_lexicon=label_lexicon,
        )
        transliterator = build_transliterator(args.transliteration_backend, args.target_lang, args.xlit_lang)

    converter = MaskedConverter(
        resolver=resolver,
        translator=translator,
        transliterator=transliterator,
        label_lexicon=label_lexicon,
    )

    font_path = resolve_font_path(args.font_path)
    fallback_font_path = resolve_fallback_font_path(args.fallback_font_path, font_path)
    print(f"Target language: {args.target_lang}")
    print(f"Transliteration language: {args.xlit_lang}")
    print(f"Transliteration backend: {args.transliteration_backend}")
    print(f"Font: {font_path}")
    print(f"Fallback font: {fallback_font_path}")

    files = find_ocr_files(ocr_path)
    if args.limit is not None:
        files = files[: args.limit]

    summary_docs: List[Dict[str, object]] = []
    action_counts = {a.value: 0 for a in Action}
    total_spans = 0
    total_overflow = 0

    for ocr_file in tqdm(files, desc="Localizing documents"):
        spans, size, img_path = load_spans_for_file(ocr_file, image_dir)
        converted = [converter.convert_span(s) for s in spans]
        enriched = render_document(
            converted,
            size=size,
            font_path=font_path,
            fallback_font_path=fallback_font_path,
            out_path=out_img_dir / f"{ocr_file.stem}.png",
            min_font_size=args.min_font_size,
            allow_wrap=not args.no_wrap,
            font_scale=args.font_scale,
        )
        save_sroie_style_txt(enriched, out_txt_dir / f"{ocr_file.stem}.txt")
        save_json(enriched, out_json_dir / f"{ocr_file.stem}.json")

        n_overflow = sum(1 for r in enriched if r.render_meta.get("overflow"))
        for r in enriched:
            action_counts[r.action] = action_counts.get(r.action, 0) + 1
        total_spans += len(enriched)
        total_overflow += n_overflow
        summary_docs.append({
            "doc_id": ocr_file.stem,
            "source_ocr_file": str(ocr_file),
            "source_image_file": str(img_path) if img_path else None,
            "output_image_file": str(out_img_dir / f"{ocr_file.stem}.png"),
            "num_spans": len(enriched),
            "num_overflow_spans": n_overflow,
            "overflow_rate": n_overflow / max(1, len(enriched)),
        })

    summary = {
        "source_language": args.src_lang,
        "target_language": args.target_lang,
        "xlit_language": args.xlit_lang,
        "ocr_dir": str(ocr_path),
        "image_dir": str(image_dir) if image_dir else None,
        "output_dir": str(out_dir),
        "num_documents": len(files),
        "total_spans": total_spans,
        "action_counts": action_counts,
        "total_overflow_spans": total_overflow,
        "overall_overflow_rate": total_overflow / max(1, total_spans),
        "documents": summary_docs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Images:   {out_img_dir}")
    print(f"BBox txt: {out_txt_dir}")
    print(f"JSON:     {out_json_dir}")
    print(f"Summary:  {out_dir / 'summary.json'}")
    print(f"Overflow: {total_overflow}/{total_spans} = {summary['overall_overflow_rate']:.2%}")
    print(f"Actions:  {action_counts}")


if __name__ == "__main__":
    main()
