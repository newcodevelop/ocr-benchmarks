#!/usr/bin/env python3
"""
WildReceipt benchmark inference for OCR-native models with semantic grounding.

This file combines the WildReceipt data/evaluation pipeline from
``benchmark_wr_grounded.py`` with the working OCR-native adapters from
``benchmark_sroie_grounded_ocr.py``.

Supported OCR-native backends:

1. DeepSeek-OCR
   - Loaded with in-process ``vllm.LLM``.
   - Called through ``LLM.generate`` with a PIL image and
     ``<image>\\nFree OCR.``.
   - Uses ``NGramPerReqLogitsProcessor`` unless explicitly disabled.

2. Mistral OCR
   - Called through the hosted ``client.ocr.process`` API.
   - Reads page markdown as the model's raw transcription.

Neither model natively returns the 24-field WildReceipt JSON schema.  Therefore
the first-stage adapter performs an explicit and auditable conversion:

    raw model OCR text
      -> cleaned OCR lines
      -> fuzzy alignment to label-free WildReceipt OCR geometry
      -> 24 field-conditioned primitive span proposals
      -> list-valued draft JSON
      -> the unchanged OCR grounding and crop-refinement pipeline

The resulting "base" metric measures OCR-native transcription plus deterministic
semantic proposal heuristics.  It must not be described as native structured
JSON extraction by DeepSeek-OCR or Mistral OCR.

Anti-leakage rule:

    Semantic labels are used only to build evaluation ground truth.
    Alignment, proposals, grounding, and crop refinement receive text+bbox only.
"""

import argparse
import atexit
import base64
import binascii
import csv
import difflib
import json
import os
import re
import shlex
import subprocess
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from glob import glob
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_BASELINES_DIR = os.path.dirname(SCRIPT_DIR)

import os

os.environ["HF_HOME"] = "/workspace/hf_cache"

# WildReceipt defines a key and a value class for each semantic concept.  These
# canonical names are deliberately stable even if class_list.txt uses spaces,
# capitalization, abbreviations, or "sub_total" instead of "subtotal".
WR_FIELDS: Tuple[str, ...] = (
    "store_name_key",
    "store_name_value",
    "store_address_key",
    "store_address_value",
    "telephone_key",
    "telephone_value",
    "date_key",
    "date_value",
    "time_key",
    "time_value",
    "product_item_key",
    "product_item_value",
    "product_quantity_key",
    "product_quantity_value",
    "product_price_key",
    "product_price_value",
    "subtotal_key",
    "subtotal_value",
    "tax_key",
    "tax_value",
    "tips_key",
    "tips_value",
    "total_key",
    "total_value",
)

DEFAULT_FIELDS = ",".join(WR_FIELDS)

# User-facing descriptions are shared by the full-image prompt and crop
# verifier.  Explicit descriptions matter for the fine distinction between
# subtotal, tax, tips, and final total.
FIELD_DESCRIPTIONS: Dict[str, str] = {
    "store_name_key": "the printed key/label introducing the store name",
    "store_name_value": "the merchant or store name value",
    "store_address_key": "the printed key/label introducing the store address",
    "store_address_value": "the merchant or store address value",
    "telephone_key": "the printed telephone/phone key or label",
    "telephone_value": "the merchant telephone number value",
    "date_key": "the printed date key or label",
    "date_value": "the transaction date value",
    "time_key": "the printed time key or label",
    "time_value": "the transaction time value",
    "product_item_key": "the column key/header for product or item name",
    "product_item_value": "a purchased product or item name",
    "product_quantity_key": "the column key/header for product quantity",
    "product_quantity_value": "a purchased product quantity value",
    "product_price_key": "the column key/header for product price",
    "product_price_value": "a purchased product price value",
    "subtotal_key": "the subtotal key or label",
    "subtotal_value": "the subtotal amount value before final adjustments",
    "tax_key": "the tax/VAT/GST key or label",
    "tax_value": "the tax/VAT/GST amount value",
    "tips_key": "the tip/gratuity key or label",
    "tips_value": "the tip/gratuity amount value",
    "total_key": "the final total/payable key or label",
    "total_value": "the final payable total amount value",
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen25vl-7b-instruct": {
        "label": "Qwen2.5-VL-7B-Instruct",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "served_model_name": "qwen25vl-7b-instruct",
        "output_slug": "qwen25vl_7b_instruct",
        "vllm_server_args": "--limit-mm-per-prompt '{\"image\": 1}'",
    },
    "internvl25-8b": {
        "label": "InternVL2_5-8B",
        "model": "OpenGVLab/InternVL2_5-8B",
        "served_model_name": "internvl25-8b",
        "output_slug": "internvl25_8b",
        "vllm_server_args": "--trust-remote-code --limit-mm-per-prompt '{\"image\": 1}'",
    },
    "qwen25vl-3b-instruct": {
        "label": "Qwen2.5-VL-3B-Instruct",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "served_model_name": "qwen25vl-3b-instruct",
        "output_slug": "qwen25vl_3b_instruct",
        "vllm_server_args": "--limit-mm-per-prompt '{\"image\": 1}'",
    },
    "granite41-4b": {
        "label": "Granite 4.1-4B",
        "model": "ibm-granite/granite-vision-4.1-4b",
        "served_model_name": "granite-vision-41-4b",
        "output_slug": "granite_vision_41_4b",
        "vllm_server_args": "--trust-remote-code --limit-mm-per-prompt '{\"image\": 1}'",
    },
    "numarkdown-8b-thinking": {
        "label": "NuMarkdown-8B-Thinking",
        "model": "numind/NuMarkdown-8B-Thinking",
        "served_model_name": "numarkdown-8b-thinking",
        "output_slug": "numarkdown_8b_thinking",
        "vllm_server_args": "--trust-remote-code --limit-mm-per-prompt '{\"image\": 1}'",
    },
    "deepseek-ocr": {
        "label": "DeepSeek OCR",
        "model": "deepseek-ai/DeepSeek-OCR",
        "served_model_name": "deepseek-ocr",
        "output_slug": "deepseek_ocr",
        "vllm_server_args": "--trust-remote-code --limit-mm-per-prompt '{\"image\": 1}'",
    },
    "mistral-ocr-latest": {
        "label": "mistral-ocr-latest",
        "model": "mistral-ocr-latest",
        "served_model_name": "mistral-ocr-latest",
        "output_slug": "mistral_ocr_latest",
        "spawn_vllm_supported": False,
        "vllm_server_args": "",
    },
    "pixtral-12b-2409": {
        "label": "Pixtral-12B-2409",
        "model": "mistralai/Pixtral-12B-2409",
        "served_model_name": "pixtral-12b-2409",
        "output_slug": "pixtral_12b_2409",
        "vllm_server_args": "--limit-mm-per-prompt '{\"image\": 1}'",
    },
    "granite-vision-3.3-2b": {
        "label": "Granite Vision 3.3-2B",
        "model": "ibm-granite/granite-vision-3.3-2b",
        "served_model_name": "granite-vision-3.3-2b",
        "output_slug": "granite_vision_33_2b",
        "vllm_server_args": "--limit-mm-per-prompt '{\"image\": 1}'",
    },
}

MODEL_PRESET_ALIASES = {
    "qwen2.5-vl-7b-instruct": "qwen25vl-7b-instruct",
    "qwen25vl-7b": "qwen25vl-7b-instruct",
    "internvl2_5-8b": "internvl25-8b",
    "internvl2.5-8b": "internvl25-8b",
    "qwen2.5-vl-3b-instruct": "qwen25vl-3b-instruct",
    "qwen25vl-3b": "qwen25vl-3b-instruct",
    "granite-4.1-4b": "granite41-4b",
    "granite4.1-4b": "granite41-4b",
    "nuMarkdown-8b-thinking": "numarkdown-8b-thinking",
    "deepseekocr": "deepseek-ocr",
    "mistral_ocr_latest": "mistral-ocr-latest",
    "pixtral": "pixtral-12b-2409",
}


def canonical_model_preset_name(name: str) -> str:
    key = str(name or "").strip()
    lowered = key.lower()
    if key in MODEL_PRESETS:
        return key
    if lowered in MODEL_PRESETS:
        return lowered
    if key in MODEL_PRESET_ALIASES:
        return MODEL_PRESET_ALIASES[key]
    if lowered in MODEL_PRESET_ALIASES:
        return MODEL_PRESET_ALIASES[lowered]
    supported = ", ".join(sorted(MODEL_PRESETS.keys()))
    raise ValueError(f"Unknown --model_preset {name!r}. Supported presets: {supported}")


def print_model_presets() -> None:
    print("Available model presets:")
    for key in sorted(MODEL_PRESETS.keys()):
        preset = MODEL_PRESETS[key]
        spawn_note = "" if preset.get("spawn_vllm_supported", True) else " [external/OpenAI-compatible endpoint]"
        print(f"  {key}: {preset['label']} -> {preset['model']}{spawn_note}")


def apply_model_preset(args: argparse.Namespace) -> None:
    model_overridden = args.model is not None
    served_name_overridden = args.served_model_name is not None
    server_args_overridden = args.vllm_server_args is not None

    args.model_preset = canonical_model_preset_name(args.model_preset)
    preset = MODEL_PRESETS[args.model_preset]

    native_requested = args.ocr_native_adapter == "on" or (
        args.ocr_native_adapter == "auto"
        and args.model_preset in ("deepseek-ocr", "mistral-ocr-latest")
    )

    if (
        args.spawn_vllm
        and not preset.get("spawn_vllm_supported", True)
        and not model_overridden
        and not native_requested
    ):
        raise ValueError(
            f"--model_preset {args.model_preset!r} is not a local vLLM checkpoint preset. "
            "Use --no_spawn_vllm with an OpenAI-compatible endpoint, or pass --model with a local vLLM-served checkpoint."
        )
    if native_requested and args.model_preset in ("deepseek-ocr", "mistral-ocr-latest"):
        args.spawn_vllm = False

    if not model_overridden:
        args.model = preset["model"]
    if not served_name_overridden:
        args.served_model_name = preset["served_model_name"]
    if not server_args_overridden:
        args.vllm_server_args = preset.get("vllm_server_args", "")

    slug = preset["output_slug"]
    args.model_label = preset["label"]
    if args.output_json is None:
        args.output_json = f"WildReceipt_{slug}_grounded_ocr_results.json"
    if args.output_csv is None:
        args.output_csv = f"WildReceipt_{slug}_grounded_ocr_results.csv"
    if args.grounding_csv is None:
        args.grounding_csv = f"WildReceipt_{slug}_grounded_ocr_details.csv"
    if args.vllm_log is None:
        args.vllm_log = f"vllm_{slug}_wildreceipt_grounded_server.log"


def _prompt_schema(grounded: bool) -> str:
    """Build the large schema programmatically so prompt and EVAL_FIELDS agree."""
    if grounded:
        example = {field: [{"value": "", "bbox": [0, 0, 0, 0]}] for field in WR_FIELDS}
    else:
        example = {field: [] for field in WR_FIELDS}
    return json.dumps(example, indent=2)


def _field_catalog() -> str:
    """Explain all labels without relying on terse snake_case names alone."""
    return "\n".join(f"- {field}: {FIELD_DESCRIPTIONS[field]}" for field in WR_FIELDS)


# Every field is a list because WildReceipt contains repeated product rows and,
# occasionally, multiple telephone/address fragments.  A single-item field is
# represented by a one-element list; an absent field is an empty list.
GROUNDED_WR_PROMPT = f"""
You are an OCR extraction and grounding system.

Extract every WildReceipt semantic field from this receipt image.
Return ONLY valid JSON in this exact top-level schema:

{_prompt_schema(grounded=True)}

Field meanings:
{_field_catalog()}

Rules:
- Do not explain and do not use markdown.
- Every field must be a JSON list.
- Each occurrence must be {{"value": "exact visible text", "bbox": [x1, y1, x2, y2]}}.
- Preserve repeated product items, quantities, and prices as separate list entries.
- Put repeated entries in visual reading order: top-to-bottom, then left-to-right.
- Use the smallest visible evidence box for each occurrence.
- Coordinates are pixel coordinates in the provided image.
- Keep key/label text separate from its value.
- Distinguish subtotal, tax, tips, and final total using surrounding labels and layout.
- Use exact visible OCR text; do not normalize, calculate, infer, or hallucinate values.
- Use [] when a field is absent.
""".strip()


FLAT_WR_PROMPT = f"""
You are an OCR extraction system.

Extract every WildReceipt semantic field from this receipt image.
Return ONLY valid JSON in this exact top-level schema:

{_prompt_schema(grounded=False)}

Field meanings:
{_field_catalog()}

Rules:
- Do not explain and do not use markdown.
- Every field must be a JSON list of exact visible strings.
- Preserve repeated product items, quantities, and prices as separate entries.
- Put repeated entries in visual reading order: top-to-bottom, then left-to-right.
- Keep key/label text separate from its value.
- Distinguish subtotal, tax, tips, and final total using surrounding labels and layout.
- Use exact visible OCR text; do not normalize, calculate, infer, or hallucinate values.
- Use [] when a field is absent.
""".strip()


# The two legacy prompt-style flags are retained for command compatibility.
# On WildReceipt they intentionally use the same complete flat task instead of
# reverting to SROIE's four-field/line-item schema.
VLLM_PROMPT = FLAT_WR_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM-supported VLM/OCR models on WildReceipt with OCR-nudged semantic grounding."
    )
    parser.add_argument(
        "--model_preset",
        default=os.environ.get("MODEL_PRESET", "deepseek-ocr"),
        help=(
            "Model preset key. Supported canonical keys: "
            + ", ".join(sorted(MODEL_PRESETS.keys()))
            + ". Common aliases are also accepted."
        ),
    )
    parser.add_argument(
        "--list_model_presets",
        action="store_true",
        help="Print available model presets and exit.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME"),
        help="Override the checkpoint/API model name resolved from --model_preset.",
    )
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT"),
        help="OpenAI-compatible chat completions URL. Defaults to http://HOST:PORT/v1/chat/completions.",
    )
    parser.add_argument(
        "--spawn_vllm",
        dest="spawn_vllm",
        action="store_true",
        default=os.environ.get("SPAWN_VLLM", "1") == "1",
        help="Spawn a local vLLM OpenAI-compatible server before inference.",
    )
    parser.add_argument(
        "--no_spawn_vllm",
        dest="spawn_vllm",
        action="store_false",
        help="Use an already-running vLLM/OpenAI-compatible server at --endpoint.",
    )
    parser.add_argument("--vllm_bin", default=os.environ.get("VLLM_BIN", "vllm"))
    parser.add_argument(
        "--served_model_name",
        default=os.environ.get("SERVED_MODEL_NAME"),
        help="Override the served model name passed to vLLM and used in chat payloads.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
    )
    parser.add_argument("--dtype", default=os.environ.get("VLLM_DTYPE"))
    parser.add_argument("--max_model_len", type=int, default=os.environ.get("MAX_MODEL_LEN"))
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
    )
    parser.add_argument(
        "--vllm_server_args",
        default=os.environ.get("VLLM_SERVER_ARGS"),
        help="Override preset extra arguments appended to `vllm serve`, parsed with shell-like quoting.",
    )
    parser.add_argument(
        "--vllm_start_timeout",
        type=float,
        default=float(os.environ.get("VLLM_START_TIMEOUT", "600")),
    )
    parser.add_argument(
        "--vllm_log",
        default=os.environ.get("VLLM_LOG"),
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=os.environ.get("HF_CACHE_DIR", os.path.join(os.getcwd(), ".cache", "huggingface")),
        help="Hugging Face cache directory used by the spawned vLLM server.",
    )
    parser.add_argument(
        "--runtime_cache_dir",
        default=os.environ.get("RUNTIME_CACHE_DIR"),
        help="Torch/Triton/vLLM runtime compilation cache directory used by the spawned vLLM server.",
    )
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get("WILDRECEIPT_ROOT", os.path.join(OCR_BASELINES_DIR, "WildReceipt")),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default=os.environ.get("WILDRECEIPT_SPLIT", "test"),
        help="Official WildReceipt split to evaluate. 'all' concatenates train and test.",
    )
    parser.add_argument(
        "--annotation_file",
        action="append",
        default=None,
        help=(
            "Explicit JSON/JSONL annotation file. Repeat for multiple files. "
            "When omitted, --split resolves train.txt and/or test.txt below --dataset_root."
        ),
    )
    parser.add_argument(
        "--class_list",
        default=os.environ.get("WILDRECEIPT_CLASS_LIST"),
        help="Path to official class_list.txt. Defaults to DATASET_ROOT/class_list.txt.",
    )
    parser.add_argument(
        "--img_dir",
        default=os.environ.get("IMG_DIR"),
        help=(
            "Optional image base directory. Relative annotation file_name values are first "
            "resolved below this directory, then below --dataset_root."
        ),
    )
    parser.add_argument(
        "--ocr_source",
        choices=("annotations", "dir"),
        default=os.environ.get("OCR_SOURCE", "annotations"),
        help=(
            "annotations strips semantic labels and uses released text+bboxes as OCR; "
            "dir loads independent OCR predictions from --ocr_dir."
        ),
    )
    parser.add_argument(
        "--ocr_dir",
        default=os.environ.get("OCR_DIR"),
        help="Directory containing external OCR .txt/.json files when --ocr_source dir is selected.",
    )
    parser.add_argument(
        "--output_json",
        default=os.environ.get("OUTPUT_JSON"),
    )
    parser.add_argument(
        "--output_csv",
        default=os.environ.get("OUTPUT_CSV"),
    )
    parser.add_argument(
        "--grounding_csv",
        default=os.environ.get("GROUNDING_CSV"),
    )
    parser.add_argument("--eval_fields", default=os.environ.get("EVAL_FIELDS", DEFAULT_FIELDS))
    parser.add_argument("--num_samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "1")))
    parser.add_argument("--full_dataset", action="store_true", help="Process all images instead of --num_samples.")

    parser.add_argument(
        "--image_mode",
        choices=("vllm_resize", "full"),
        default=os.environ.get("IMAGE_MODE", "vllm_resize"),
        help="vllm_resize sends a resized PNG; full sends original compressed bytes.",
    )
    parser.add_argument(
        "--resize_image",
        dest="resize_image",
        action="store_true",
        default=None,
        help="Resize before sending. Overrides --image_mode when set.",
    )
    parser.add_argument(
        "--no_resize_image",
        dest="resize_image",
        action="store_false",
        help="Send original compressed bytes. Overrides --image_mode when set.",
    )
    parser.add_argument("--max_image_size", type=int, default=int(os.environ.get("MAX_IMAGE_SIZE", "1024")))
    parser.add_argument(
        "--prompt_style",
        choices=("grounded", "flat", "vllm", "vllm_raw"),
        default=os.environ.get("PROMPT_STYLE", "grounded"),
        help=(
            "grounded asks for occurrence values+bboxes; flat asks for occurrence "
            "value lists; vllm/vllm_raw retain transport compatibility while using "
            "the complete WildReceipt flat task."
        ),
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", "4096")),
        help="WildReceipt has 24 fields and repeated product rows, so 4096 is the practical default.",
    )
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--top_p", type=float, default=os.environ.get("TOP_P"))
    parser.add_argument("--top_k", type=int, default=os.environ.get("TOP_K"))
    parser.add_argument("--min_p", type=float, default=os.environ.get("MIN_P"))
    parser.add_argument("--frequency_penalty", type=float, default=os.environ.get("FREQUENCY_PENALTY"))
    parser.add_argument("--presence_penalty", type=float, default=os.environ.get("PRESENCE_PENALTY"))
    parser.add_argument("--repetition_penalty", type=float, default=os.environ.get("REPETITION_PENALTY"))
    parser.add_argument("--stop", action="append", default=None, help="Stop sequence. Can be repeated.")
    parser.add_argument(
        "--guided_json",
        action="store_true",
        default=os.environ.get("GUIDED_JSON", "0") == "1",
        help="Ask vLLM-compatible endpoints to constrain output to JSON schema.",
    )
    parser.add_argument(
        "--extra_body_json",
        default=os.environ.get("EXTRA_BODY_JSON"),
        help="Raw JSON object merged into the request payload for endpoint-specific vLLM options.",
    )
    parser.add_argument(
        "--ocr_native_adapter",
        choices=("auto", "on", "off"),
        default=os.environ.get("OCR_NATIVE_ADAPTER", "auto"),
        help=(
            "Use native OCR invocation. auto enables it for deepseek-ocr and "
            "mistral-ocr-latest; off retains the generic structured-chat path."
        ),
    )
    parser.add_argument(
        "--deepseek_ocr_prompt",
        default=os.environ.get("DEEPSEEK_OCR_PROMPT", "<image>\nFree OCR."),
        help="Native DeepSeek-OCR prompt. The image marker must remain in the prompt.",
    )
    parser.add_argument(
        "--mistral_api_key",
        default=None,
        help="Mistral OCR API key. Environment variables are checked when omitted.",
    )
    parser.add_argument(
        "--mistral_api_key_file",
        default=os.environ.get("MISTRAL_API_KEY_FILE"),
        help="Optional file containing the Mistral OCR API key.",
    )
    parser.add_argument(
        "--mistral_image_format",
        choices=("JPEG", "PNG"),
        default=os.environ.get("MISTRAL_IMAGE_FORMAT", "JPEG"),
    )
    parser.add_argument(
        "--mistral_retry_delay",
        type=float,
        default=float(os.environ.get("MISTRAL_RETRY_DELAY", "5.0")),
    )
    parser.add_argument(
        "--disable_ngram_processor",
        action="store_true",
        default=os.environ.get("DISABLE_NGRAM_PROCESSOR", "0") == "1",
        help="Disable DeepSeek-OCR's NGramPerReqLogitsProcessor for debugging.",
    )
    parser.add_argument("--ngram_size", type=int, default=int(os.environ.get("NGRAM_SIZE", "30")))
    parser.add_argument(
        "--ngram_window_size",
        type=int,
        default=int(os.environ.get("NGRAM_WINDOW_SIZE", "90")),
    )
    parser.add_argument(
        "--deepseek_skip_special_tokens",
        action="store_true",
        default=os.environ.get("DEEPSEEK_SKIP_SPECIAL_TOKENS", "0") == "1",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        default=os.environ.get("ENFORCE_EAGER", "0") == "1",
        help="Pass enforce_eager=True to the in-process DeepSeek-OCR vLLM engine.",
    )
    parser.add_argument(
        "--native_alignment_threshold",
        type=float,
        default=float(os.environ.get("NATIVE_ALIGNMENT_THRESHOLD", "0.45")),
        help="Minimum fuzzy score for mapping a native OCR line to label-free layout OCR.",
    )
    parser.add_argument(
        "--native_max_field_instances",
        type=int,
        default=int(os.environ.get("NATIVE_MAX_FIELD_INSTANCES", "20")),
        help="Maximum draft occurrences proposed for a repeated WildReceipt field.",
    )
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("HTTP_TIMEOUT", "180")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("HTTP_RETRIES", "2")))
    parser.add_argument("--sleep", type=float, default=float(os.environ.get("REQUEST_SLEEP", "0")))

    # OCR-grounding settings.
    parser.add_argument(
        "--disable_grounding",
        action="store_true",
        help="Run only VLM extraction/evaluation; skip OCR-nudged grounding.",
    )
    parser.add_argument(
        "--primitive_max_window",
        type=int,
        default=int(os.environ.get("PRIMITIVE_MAX_WINDOW", "5")),
        help="Maximum number of consecutive OCR lines considered as a primitive candidate block.",
    )
    parser.add_argument(
        "--local_context_lines",
        type=int,
        default=int(os.environ.get("LOCAL_CONTEXT_LINES", "5")),
        help="Number of OCR lines around the VLM bbox to search during local refinement.",
    )
    parser.add_argument(
        "--evidence_high_threshold",
        type=float,
        default=float(os.environ.get("EVIDENCE_HIGH_THRESHOLD", "0.70")),
        help="Value-match score above which a VLM bbox is treated as grounded.",
    )
    parser.add_argument(
        "--evidence_partial_threshold",
        type=float,
        default=float(os.environ.get("EVIDENCE_PARTIAL_THRESHOLD", "0.30")),
        help="Value-match score above which local refinement is attempted before global recovery.",
    )
    parser.add_argument(
        "--primitive_threshold",
        type=float,
        default=float(os.environ.get("PRIMITIVE_THRESHOLD", "0.25")),
        help="Minimum field-conditioned OCR-span score for primitive proposal when VLM value is empty.",
    )
    parser.add_argument(
        "--verify_crops",
        action="store_true",
        default=os.environ.get("VERIFY_CROPS", "0") == "1",
        help="After OCR grounding/proposal, call the VLM on the crop for verification/extraction.",
    )
    parser.add_argument(
        "--refine_crop_values",
        dest="refine_crop_values",
        action="store_true",
        default=os.environ.get("REFINE_CROP_VALUES", "1") == "1",
        help="When --verify_crops is enabled, use positive crop-level VLM values as final field values.",
    )
    parser.add_argument(
        "--no_refine_crop_values",
        dest="refine_crop_values",
        action="store_false",
        help="Run crop verification but keep OCR/VLM grounded values unchanged.",
    )
    parser.add_argument(
        "--max_crop_verifications",
        type=int,
        default=int(os.environ.get("MAX_CROP_VERIFICATIONS", "999999")),
        help=(
            "Cap total crop-level VLM calls across the run. WildReceipt can have many "
            "instances per receipt, so use this to control runtime/cost."
        ),
    )
    parser.add_argument(
        "--crop_padding",
        type=int,
        default=int(os.environ.get("CROP_PADDING", "4")),
        help="Pixel padding around grounded bbox before optional crop verification.",
    )
    parser.add_argument(
        "--max_primitive_instances",
        type=int,
        default=int(os.environ.get("MAX_PRIMITIVE_INSTANCES", "8")),
        help=(
            "Maximum non-overlapping OCR proposals for an entirely omitted field. "
            "This mainly bounds repeated product-row recovery."
        ),
    )

    args = parser.parse_args()
    if args.list_model_presets:
        print_model_presets()
        raise SystemExit(0)

    apply_model_preset(args)

    if args.full_dataset:
        args.num_samples = None
    if args.class_list is None:
        args.class_list = os.path.join(args.dataset_root, "class_list.txt")
    if args.img_dir is None:
        args.img_dir = args.dataset_root
    if args.ocr_source == "dir" and args.ocr_dir is None:
        args.ocr_dir = os.path.join(args.dataset_root, "ocr_pred")
    if args.resize_image is None:
        args.resize_image = args.image_mode == "vllm_resize"
    else:
        args.image_mode = "vllm_resize" if args.resize_image else "full"

    for name in ("top_p", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, float(value))
    if args.top_k is not None:
        args.top_k = int(args.top_k)
    if args.max_model_len is not None:
        args.max_model_len = int(args.max_model_len)
    if args.endpoint is None:
        args.endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    if args.primitive_max_window < 1:
        raise ValueError("--primitive_max_window must be >= 1")
    if args.max_primitive_instances < 1:
        raise ValueError("--max_primitive_instances must be >= 1")
    return args


ARGS = parse_args()

def configure_local_cache(args):
    hf_cache_dir = args.hf_cache_dir or "/workspace/hf_cache"
    runtime_cache_dir = args.runtime_cache_dir or "/workspace/runtime_cache"

    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, "hub"), exist_ok=True)
    os.makedirs(runtime_cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "hub")

    os.environ["XDG_CACHE_HOME"] = runtime_cache_dir
    os.environ["TORCH_HOME"] = os.path.join(runtime_cache_dir, "torch")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(runtime_cache_dir, "triton")
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(runtime_cache_dir, "vllm")
    os.environ["TMPDIR"] = os.path.join(runtime_cache_dir, "tmp")

    for key in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "TMPDIR",
    ]:
        os.makedirs(os.environ[key], exist_ok=True)


ARGS = parser.parse_args()
configure_local_cache(ARGS)




EVAL_FIELDS = [field.strip() for field in ARGS.eval_fields.split(",") if field.strip()]
input_token_lengths: List[int] = []
output_token_lengths: List[int] = []
crop_verification_calls = 0
VLLM_PROCESS: Optional[subprocess.Popen] = None
VLLM_LOG_HANDLE: Optional[Any] = None
DEEPSEEK_LLM: Optional[Any] = None
DEEPSEEK_SAMPLING_PARAMS: Optional[Any] = None
MISTRAL_CLIENT: Optional[Any] = None


# -----------------------------------------------------------------------------
# Normalization and evaluation
# -----------------------------------------------------------------------------


def normalize_space(text: Any) -> str:
    text = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_for_match(text: Any) -> str:
    text = normalize_space(text)
    text = text.replace("|", " ")
    text = re.sub(r"[^a-z0-9./:&%+\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_field(field: str, value: Any) -> str:
    text = normalize_space(value)
    # All monetary *value* categories receive the same currency normalization.
    # Key fields such as "total_key" remain textual because "TOTAL" is itself
    # the target annotation there.
    if field in {
        "product_price_value",
        "subtotal_value",
        "tax_value",
        "tips_value",
        "total_value",
    }:
        text = text.replace(",", "")
        text = re.sub(r"\b(rm|rp|idr|usd|rs|myr|inr|eur|gbp|aud|cad)\b", "", text)
        text = re.sub(r"[^0-9.\-]+", "", text)
        if text.endswith(".") and text.count(".") == 1:
            text = text[:-1]
    return text.strip()


def metric_tokens(text: Any) -> List[str]:
    normalized = normalize_for_match(text)
    tokens = []
    for token in normalized.split():
        cleaned = token.strip(".,:;")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def token_multiset_counts(gt_text: str, pred_text: str) -> Tuple[int, int, int]:
    gt_counts = Counter(metric_tokens(gt_text))
    pred_counts = Counter(metric_tokens(pred_text))
    vocab = set(gt_counts) | set(pred_counts)
    tp = sum(min(gt_counts[t], pred_counts[t]) for t in vocab)
    fp = sum(max(0, pred_counts[t] - gt_counts[t]) for t in vocab)
    fn = sum(max(0, gt_counts[t] - pred_counts[t]) for t in vocab)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def safe_percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------


def resize_image(image: Any, max_size: int) -> Any:
    width, height = image.size
    scale = max_size / max(width, height)
    if scale >= 1:
        return image
    return image.resize((int(width * scale), int(height * scale)))


def image_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def jpeg_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    try:
        with open(image_path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                marker_start = f.read(1)
                while marker_start and marker_start != b"\xff":
                    marker_start = f.read(1)
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                marker_value = marker[0]
                if marker_value in (0xD8, 0xD9):
                    continue
                length = struct.unpack(">H", f.read(2))[0]
                if 0xC0 <= marker_value <= 0xCF and marker_value not in (0xC4, 0xC8, 0xCC):
                    frame = f.read(5)
                    height = struct.unpack(">H", frame[1:3])[0]
                    width = struct.unpack(">H", frame[3:5])[0]
                    return width, height
                f.seek(length - 2, os.SEEK_CUR)
    except Exception:
        return None


def png_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    try:
        with open(image_path, "rb") as f:
            header = f.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", header[16:24])
    except Exception:
        return None


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    extension = os.path.splitext(image_path)[1].lower()
    if Image is not None:
        try:
            with Image.open(image_path) as im:
                return im.size
        except Exception:
            pass
    if extension in (".jpg", ".jpeg"):
        return jpeg_dimensions(image_path)
    if extension == ".png":
        return png_dimensions(image_path)
    return None


def read_png_chunks(data: bytes) -> List[Tuple[bytes, bytes]]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    chunks: List[Tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        chunks.append((chunk_type, chunk_data))
        offset += 12 + length
        if chunk_type == b"IEND":
            break
    return chunks


def png_unfilter_scanlines(raw: bytes, width: int, height: int, bpp: int) -> List[bytearray]:
    stride = width * bpp
    rows: List[bytearray] = []
    offset = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        recon = bytearray(stride)
        for i, value in enumerate(row):
            left = recon[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                recon[i] = value
            elif filter_type == 1:
                recon[i] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[i] = (value + up) & 0xFF
            elif filter_type == 3:
                recon[i] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                recon[i] = (value + predictor) & 0xFF
            else:
                raise ValueError(f"unsupported PNG filter: {filter_type}")
        rows.append(recon)
        previous = recon
    return rows


def png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)


def resized_image_file_to_data_url(image_path: str, max_size: int) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    if Image is None:
        raise RuntimeError("Pillow is required for --image_mode vllm_resize")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = resize_image(image, max_size)
    return image_to_data_url(image), original_size, image.size


def full_image_file_to_data_url(image_path: str) -> Tuple[str, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    extension = os.path.splitext(image_path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        extension, "application/octet-stream"
    )
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    image_size = get_image_size(image_path)
    return f"data:{mime};base64,{encoded}", image_size, image_size


def crop_image_to_data_url(
    image_path: str,
    bbox: Sequence[float],
    padding: int = 4,
    max_size: int = 1024,
) -> Tuple[str, Tuple[int, int], List[int]]:
    if Image is None:
        raise RuntimeError("Pillow is required for crop verification")
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = sanitize_bbox(bbox, (width, height)) or [0, 0, width, height]
        x1 = max(0, int(round(x1)) - padding)
        y1 = max(0, int(round(y1)) - padding)
        x2 = min(width, int(round(x2)) + padding)
        y2 = min(height, int(round(y2)) + padding)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
        crop = image.crop((x1, y1, x2, y2))
        crop = resize_image(crop, max_size)
        return image_to_data_url(crop), crop.size, [x1, y1, x2, y2]


def crop_image_to_pil(
    image_path: str,
    bbox: Sequence[float],
    padding: int = 4,
    max_size: int = 1024,
) -> Tuple[Any, Tuple[int, int], List[int]]:
    """Return a PIL crop for OCR-native backends that do not accept data URLs."""
    if Image is None:
        raise RuntimeError("Pillow is required for OCR-native crop refinement")
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = sanitize_bbox(bbox, (width, height)) or [0, 0, width, height]
        x1 = max(0, int(round(x1)) - padding)
        y1 = max(0, int(round(y1)) - padding)
        x2 = min(width, int(round(x2)) + padding)
        y2 = min(height, int(round(y2)) + padding)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
        crop = resize_image(image.crop((x1, y1, x2, y2)), max_size)
        return crop.copy(), crop.size, [x1, y1, x2, y2]


# -----------------------------------------------------------------------------
# Prompting and vLLM calls
# -----------------------------------------------------------------------------


def prompt_text() -> str:
    if ARGS.prompt_style == "grounded":
        return GROUNDED_WR_PROMPT
    if ARGS.prompt_style == "flat":
        return FLAT_WR_PROMPT
    if ARGS.prompt_style == "vllm_raw":
        return f"""
<|system|>
You are a helpful OCR extraction system.

<|user|>
<image>

{VLLM_PROMPT}

<|assistant|>
""".strip()
    return VLLM_PROMPT


def crop_verification_prompt(field: str) -> str:
    readable = FIELD_DESCRIPTIONS.get(field, field)
    concept = field.removesuffix("_key").removesuffix("_value")
    role = "key/label" if field.endswith("_key") else "value"
    field_rules = f"""
- Return only the {role} for the {concept.replace('_', ' ')} field.
- Do not merge the printed key/label with its value.
- A subtotal, tax, tips, product price, and final total are different fields.
- For product fields, return only the one occurrence visible in this crop.
""".strip()
    return f"""
You are verifying a cropped region from a receipt/invoice.

Target field: {readable}

Return ONLY valid JSON:
{{
  "supports_field": true,
  "value": ""
}}

Rules:
- If the crop contains the target field, supports_field must be true and value must be the exact visible text for that field.
- If the crop does not contain the target field, supports_field must be false and value must be "".
- If the crop contains multiple fields, return only the target field value.
{field_rules}
- Do not explain.
- Do not add markdown.
""".strip()


def is_deepseek_ocr_model() -> bool:
    return ARGS.model_preset == "deepseek-ocr" or "deepseek-ocr" in str(ARGS.model).lower()


def is_mistral_ocr_model() -> bool:
    return (
        ARGS.model_preset == "mistral-ocr-latest"
        or str(ARGS.model).lower() == "mistral-ocr-latest"
    )


def use_ocr_native_adapter() -> bool:
    if ARGS.ocr_native_adapter == "on":
        return True
    if ARGS.ocr_native_adapter == "off":
        return False
    return is_deepseek_ocr_model() or is_mistral_ocr_model()


def build_messages(image_url: str, text: str) -> List[Dict[str, Any]]:
    if ARGS.prompt_style == "vllm_raw":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text},
                ],
            }
        ]
    return [
        {"role": "system", "content": "You are a helpful OCR extraction system."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ],
        },
    ]


def wr_guided_json_schema() -> Dict[str, Any]:
    """Return a JSON schema matching list-valued WildReceipt predictions."""
    if ARGS.prompt_style == "grounded":
        occurrence_schema = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["value", "bbox"],
            "additionalProperties": False,
        }
        field_schema = {"type": "array", "items": occurrence_schema}
        return {
            "type": "object",
            "properties": {field: field_schema for field in EVAL_FIELDS},
            "required": EVAL_FIELDS,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            field: {"type": "array", "items": {"type": "string"}} for field in EVAL_FIELDS
        },
        "required": EVAL_FIELDS,
        "additionalProperties": False,
    }


def optional_generation_params() -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name in ("top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
        value = getattr(ARGS, name)
        if value is not None:
            params[name] = value
    if ARGS.stop:
        params["stop"] = ARGS.stop
    if ARGS.guided_json:
        params["guided_json"] = wr_guided_json_schema()
    if ARGS.extra_body_json:
        try:
            extra = json.loads(ARGS.extra_body_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--extra_body_json must be a valid JSON object: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("--extra_body_json must decode to a JSON object")
        params.update(extra)
    return params


def request_model_name() -> str:
    return ARGS.served_model_name or ARGS.model


def vllm_models_url() -> str:
    return f"http://{ARGS.host}:{ARGS.port}/v1/models"


def wait_for_vllm_ready() -> None:
    deadline = time.time() + ARGS.vllm_start_timeout
    last_error = ""
    while time.time() < deadline:
        if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited early with code {VLLM_PROCESS.returncode}. "
                f"Check log: {ARGS.vllm_log}"
            )
        try:
            with urllib.request.urlopen(vllm_models_url(), timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting {ARGS.vllm_start_timeout:.0f}s for vLLM at {vllm_models_url()}. "
        f"Last error: {last_error}. Check log: {ARGS.vllm_log}"
    )


def stop_vllm_server() -> None:
    global VLLM_PROCESS, VLLM_LOG_HANDLE
    if VLLM_PROCESS is not None and VLLM_PROCESS.poll() is None:
        VLLM_PROCESS.terminate()
        try:
            VLLM_PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:
            VLLM_PROCESS.kill()
            VLLM_PROCESS.wait(timeout=20)
    VLLM_PROCESS = None
    if VLLM_LOG_HANDLE is not None:
        VLLM_LOG_HANDLE.close()
        VLLM_LOG_HANDLE = None


def start_vllm_server() -> None:
    global VLLM_PROCESS, VLLM_LOG_HANDLE
    if not ARGS.spawn_vllm:
        print(f"Using existing vLLM/OpenAI-compatible endpoint: {ARGS.endpoint}")
        return

    if VLLM_PROCESS is not None:
        return

    cmd = [
        ARGS.vllm_bin,
        "serve",
        ARGS.model,
        "--host",
        ARGS.host,
        "--port",
        str(ARGS.port),
        "--tensor-parallel-size",
        str(ARGS.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(ARGS.gpu_memory_utilization),
    ]
    if ARGS.served_model_name:
        cmd.extend(["--served-model-name", ARGS.served_model_name])
    if ARGS.dtype:
        cmd.extend(["--dtype", ARGS.dtype])
    if ARGS.max_model_len is not None:
        cmd.extend(["--max-model-len", str(ARGS.max_model_len)])
    if ARGS.vllm_server_args:
        cmd.extend(shlex.split(ARGS.vllm_server_args))

    print("Starting vLLM server:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"vLLM log: {ARGS.vllm_log}")
    print(f"Hugging Face cache: {ARGS.hf_cache_dir}")

    env = os.environ.copy()
    if ARGS.hf_cache_dir:
        os.makedirs(ARGS.hf_cache_dir, exist_ok=True)
        env["HF_HOME"] = ARGS.hf_cache_dir
        env["HF_HUB_CACHE"] = os.path.join(ARGS.hf_cache_dir, "hub")
        env["TRANSFORMERS_CACHE"] = os.path.join(ARGS.hf_cache_dir, "hub")
    runtime_cache_dir = ARGS.runtime_cache_dir or os.path.join(ARGS.hf_cache_dir, "runtime")
    if runtime_cache_dir:
        os.makedirs(runtime_cache_dir, exist_ok=True)
        env["XDG_CACHE_HOME"] = runtime_cache_dir
        env["TORCH_HOME"] = os.path.join(runtime_cache_dir, "torch")
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(runtime_cache_dir, "torchinductor")
        env["TRITON_CACHE_DIR"] = os.path.join(runtime_cache_dir, "triton")
        env["CUDA_CACHE_PATH"] = os.path.join(runtime_cache_dir, "cuda")
        env["VLLM_CACHE_ROOT"] = os.path.join(runtime_cache_dir, "vllm")
        env["TMPDIR"] = os.path.join(runtime_cache_dir, "tmp")
        for path in (
            env["TORCH_HOME"],
            env["TORCHINDUCTOR_CACHE_DIR"],
            env["TRITON_CACHE_DIR"],
            env["CUDA_CACHE_PATH"],
            env["VLLM_CACHE_ROOT"],
            env["TMPDIR"],
        ):
            os.makedirs(path, exist_ok=True)
        print(f"Runtime cache: {runtime_cache_dir}")

    VLLM_LOG_HANDLE = open(ARGS.vllm_log, "w", encoding="utf-8")
    VLLM_PROCESS = subprocess.Popen(
        cmd,
        stdout=VLLM_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    atexit.register(stop_vllm_server)
    wait_for_vllm_ready()
    print(f"vLLM server is ready at {ARGS.endpoint}")


def describe_http_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"HTTP Error {exc.code}: {exc.reason}. Response body: {body[:2000]}"
    return str(exc)


def call_vllm(image_url: str, text_prompt: str) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": request_model_name(),
        "messages": build_messages(image_url, text_prompt),
        "max_tokens": ARGS.max_tokens,
        "temperature": ARGS.temperature,
    }
    payload.update(optional_generation_params())
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ARGS.endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[Exception] = None
    for attempt in range(ARGS.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=ARGS.timeout) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            usage = response_json.get("usage") or {}
            if isinstance(usage.get("prompt_tokens"), int):
                input_token_lengths.append(usage["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                output_token_lengths.append(usage["completion_tokens"])
            content = response_json["choices"][0]["message"].get("content") or ""
            return content.strip(), response_json
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < ARGS.retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed: {describe_http_error(last_error)}")


def run_vllm_image(image_path: str) -> Tuple[str, Dict[str, Any], Optional[Tuple[int, int]], Optional[Tuple[int, int]], str]:
    if ARGS.resize_image:
        image_url, original_size, sent_size = resized_image_file_to_data_url(image_path, ARGS.max_image_size)
        image_target = str(ARGS.max_image_size)
        print(f"Image Size Used: {sent_size} (resized from {original_size}; target max={ARGS.max_image_size})")
    else:
        image_url, original_size, sent_size = full_image_file_to_data_url(image_path)
        image_target = "full"
        print(f"Image Size Used: {sent_size or 'original'}")
    print(f"Image URL payload length: {len(image_url)} characters")
    raw_output, response_json = call_vllm(image_url, prompt_text())
    return raw_output, response_json, original_size, sent_size, image_target


def pil_to_base64(image: Any, fmt: str = "JPEG") -> Tuple[str, str]:
    """Encode a PIL image for the hosted Mistral OCR document API."""
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    media_type = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return encoded, media_type


def load_deepseek_ocr_vllm() -> None:
    """Initialize the working in-process DeepSeek-OCR inference path."""
    global DEEPSEEK_LLM, DEEPSEEK_SAMPLING_PARAMS
    if DEEPSEEK_LLM is not None and DEEPSEEK_SAMPLING_PARAMS is not None:
        return

    print("\nLOADING DEEPSEEK-OCR WITH IN-PROCESS VLLM...\n")
    print(f"MODEL_NAME = {ARGS.model}")
    from vllm import LLM, SamplingParams

    logits_processors = None
    if not ARGS.disable_ngram_processor:
        try:
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

            logits_processors = [NGramPerReqLogitsProcessor]
            print("Using DeepSeek-OCR NGramPerReqLogitsProcessor.")
        except Exception as exc:
            raise ImportError(
                "Could not import DeepSeek-OCR NGramPerReqLogitsProcessor. "
                "Use a compatible vLLM build or pass --disable_ngram_processor. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    llm_kwargs: Dict[str, Any] = {
        "model": ARGS.model,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "gpu_memory_utilization": ARGS.gpu_memory_utilization,
        "tensor_parallel_size": ARGS.tensor_parallel_size,
        "dtype": ARGS.dtype or "auto",
        "enforce_eager": ARGS.enforce_eager,
        "trust_remote_code": True,
    }
    if ARGS.max_model_len is not None:
        llm_kwargs["max_model_len"] = ARGS.max_model_len
    if logits_processors is not None:
        llm_kwargs["logits_processors"] = logits_processors

    extra_args = None
    if not ARGS.disable_ngram_processor:
        extra_args = {
            "ngram_size": ARGS.ngram_size,
            "window_size": ARGS.ngram_window_size,
            "whitelist_token_ids": {128821, 128822},
        }

    print("HF_HOME =", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE =", os.environ.get("HF_HUB_CACHE"))
    print("TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
    DEEPSEEK_LLM = LLM(**llm_kwargs)
    DEEPSEEK_SAMPLING_PARAMS = SamplingParams(
        temperature=ARGS.temperature,
        max_tokens=ARGS.max_tokens,
        extra_args=extra_args,
        skip_special_tokens=ARGS.deepseek_skip_special_tokens,
    )
    print("DEEPSEEK-OCR MODEL LOADED SUCCESSFULLY\n")


def resolve_mistral_api_key() -> Optional[str]:
    if ARGS.mistral_api_key:
        return str(ARGS.mistral_api_key).strip()
    for env_name in (
        "MISTRAL_API_KEY",
        "MISTRALAI_API_KEY",
        "MISTRAL_API_TOKEN",
        "MISTRAL_KEY",
    ):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    if ARGS.mistral_api_key_file:
        with open(ARGS.mistral_api_key_file, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
        if value:
            return value
    return None


def build_mistral_client() -> None:
    global MISTRAL_CLIENT
    if MISTRAL_CLIENT is not None:
        return
    api_key = resolve_mistral_api_key()
    if not api_key:
        raise RuntimeError(
            "Mistral OCR requires authentication. Provide --mistral_api_key, "
            "--mistral_api_key_file, or MISTRAL_API_KEY."
        )
    try:
        from mistralai.client import Mistral
    except Exception:
        from mistralai import Mistral
    MISTRAL_CLIENT = Mistral(api_key=api_key)


def initialize_ocr_backend() -> None:
    if is_deepseek_ocr_model():
        load_deepseek_ocr_vllm()
    elif is_mistral_ocr_model():
        build_mistral_client()
    else:
        raise ValueError("OCR-native mode supports only DeepSeek-OCR and Mistral OCR.")


def call_deepseek_ocr_image(image: Any) -> Tuple[str, Dict[str, Any]]:
    initialize_ocr_backend()
    assert DEEPSEEK_LLM is not None
    assert DEEPSEEK_SAMPLING_PARAMS is not None
    outputs = DEEPSEEK_LLM.generate(
        [
            {
                "prompt": ARGS.deepseek_ocr_prompt,
                "multi_modal_data": {"image": image},
            }
        ],
        DEEPSEEK_SAMPLING_PARAMS,
        use_tqdm=False,
    )
    raw_output = ""
    if outputs and getattr(outputs[0], "outputs", None):
        raw_output = outputs[0].outputs[0].text or ""
    return raw_output, {
        "backend": "deepseek_ocr_vllm_inprocess",
        "prompt": ARGS.deepseek_ocr_prompt,
        "num_outputs": len(outputs or []),
    }


def call_mistral_ocr_image(image: Any) -> Tuple[str, Dict[str, Any]]:
    initialize_ocr_backend()
    if MISTRAL_CLIENT is None:
        raise RuntimeError("Mistral OCR client was not initialized")
    encoded, media_type = pil_to_base64(image, fmt=ARGS.mistral_image_format)
    last_error: Optional[Exception] = None
    for attempt in range(ARGS.retries + 1):
        try:
            response = MISTRAL_CLIENT.ocr.process(
                model=ARGS.model,
                document={
                    "type": "image_url",
                    "image_url": f"data:{media_type};base64,{encoded}",
                },
            )
            text = "\n\n".join(
                page.markdown
                for page in (getattr(response, "pages", None) or [])
                if getattr(page, "markdown", None)
            )
            return text, {
                "backend": "mistral_ocr_client",
                "model": ARGS.model,
                "num_pages": len(getattr(response, "pages", None) or []),
            }
        except Exception as exc:
            last_error = exc
            if attempt < ARGS.retries:
                print(
                    f"Mistral OCR failed ({attempt + 1}/{ARGS.retries + 1}): "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(ARGS.mistral_retry_delay)
    raise RuntimeError(
        f"Mistral OCR failed after {ARGS.retries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def call_ocr_native_image(image: Any) -> Tuple[str, Dict[str, Any]]:
    if is_deepseek_ocr_model():
        return call_deepseek_ocr_image(image)
    if is_mistral_ocr_model():
        return call_mistral_ocr_image(image)
    raise RuntimeError("Unsupported OCR-native model")


def clean_native_ocr_line(line: str) -> str:
    """Remove markdown decoration while retaining recognized receipt text."""
    text = str(line or "").strip()
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip("|").strip()
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\t")


def native_ocr_lines(text: str) -> List[str]:
    """Convert OCR markdown/text into semantic-sized recognition segments.

    WildReceipt annotates keys and values separately, while OCR engines often
    emit ``TOTAL 12.34`` as one line. Recognizable key prefixes are split from
    their trailing text using only the model transcription.
    """
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        raw_cells = raw_line.split("|") if "|" in raw_line else [raw_line]
        for raw_cell in raw_cells:
            line = clean_native_ocr_line(raw_cell)
            if not line or re.fullmatch(r"[:|+\-= _]+", line):
                continue
            key_value = re.match(
                r"^(grand\s+total|sub\s*total|subtotal|total|tax|vat|gst|sst|"
                r"tip|tips|gratuity|date|time|tel(?:ephone)?|phone|contact|"
                r"qty|quantity|price|amount|item|description)"
                r"\s*[:#-]?\s+(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if key_value:
                lines.append(key_value.group(1).strip())
                lines.append(key_value.group(2).strip())
            else:
                lines.append(line)
    return lines


def align_native_ocr_to_layout(
    raw_text: str,
    layout_ocr_lines: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Align model text to label-free geometry without copying layout text."""
    recognized = native_ocr_lines(raw_text)
    candidate_spans = generate_candidate_spans(
        layout_ocr_lines,
        max_window=min(3, max(1, ARGS.primitive_max_window)),
    )
    used_layout_indices: set = set()
    aligned: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for native_index, native_line in enumerate(recognized):
        available = [
            span
            for span in candidate_spans
            if not (set(span.get("line_indices") or []) & used_layout_indices)
        ]
        if not available:
            diagnostics.append(
                {"native_index": native_index, "text": native_line, "matched": False}
            )
            continue
        best = max(
            available,
            key=lambda span: fuzzy_similarity(native_line, span.get("text", "")),
        )
        score = fuzzy_similarity(native_line, best.get("text", ""))
        if score < ARGS.native_alignment_threshold:
            diagnostics.append(
                {
                    "native_index": native_index,
                    "text": native_line,
                    "matched": False,
                    "best_score": score,
                    "best_layout_text": best.get("text", ""),
                }
            )
            continue
        layout_indices = list(best.get("line_indices") or [])
        aligned.append(
            {
                "idx": native_index,
                "text": native_line,
                "bbox": best.get("bbox"),
                "raw": {
                    "native_text": native_line,
                    "layout_text": best.get("text", ""),
                    "alignment_score": score,
                    "layout_line_indices": layout_indices,
                },
            }
        )
        used_layout_indices.update(layout_indices)
        diagnostics.append(
            {
                "native_index": native_index,
                "text": native_line,
                "matched": True,
                "score": score,
                "layout_text": best.get("text", ""),
                "bbox": best.get("bbox"),
            }
        )
    return sort_ocr_lines(aligned), diagnostics


def semantic_draft_from_aligned_ocr(
    aligned_lines: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply WildReceipt field heuristics to OCR-native text occurrences."""
    draft: Dict[str, List[Dict[str, Any]]] = {field: [] for field in EVAL_FIELDS}
    for field in EVAL_FIELDS:
        if field in {
            "product_item_value",
            "product_quantity_value",
            "product_price_value",
        }:
            max_instances = ARGS.native_max_field_instances
        elif field == "store_address_value":
            max_instances = min(6, ARGS.native_max_field_instances)
        elif field.endswith("_key"):
            max_instances = 1
        else:
            max_instances = min(3, ARGS.native_max_field_instances)
        proposals = best_primitive_spans(
            aligned_lines,
            field=field,
            max_window=ARGS.primitive_max_window,
            max_instances=max_instances,
        )
        draft[field] = [
            {"value": str(span.get("text") or "").strip(), "bbox": span.get("bbox")}
            for span in proposals
            if str(span.get("text") or "").strip()
        ]
    return draft


def run_ocr_native_image(
    image_path: str,
    layout_ocr_lines: Sequence[Dict[str, Any]],
) -> Tuple[
    str,
    Dict[str, Any],
    Optional[Tuple[int, int]],
    Optional[Tuple[int, int]],
    str,
    str,
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """Run OCR, align to geometry, create draft JSON, and retain diagnostics."""
    if Image is None:
        raise RuntimeError("Pillow is required for OCR-native inference")
    with Image.open(image_path) as im:
        original = im.convert("RGB")
        original_size = original.size
        image = resize_image(original, ARGS.max_image_size) if ARGS.resize_image else original
        sent_size = image.size
        if image is original:
            image = original.copy()
        image_target = str(ARGS.max_image_size) if ARGS.resize_image else "full"

    raw_text, response = call_ocr_native_image(image)
    aligned, diagnostics = align_native_ocr_to_layout(raw_text, layout_ocr_lines)
    draft = semantic_draft_from_aligned_ocr(aligned)
    response["native_line_count"] = len(native_ocr_lines(raw_text))
    response["aligned_line_count"] = len(aligned)
    response["semantic_adapter"] = "fuzzy_layout_alignment_plus_field_heuristics"
    return (
        json.dumps(draft, ensure_ascii=False),
        response,
        original_size,
        sent_size,
        image_target,
        raw_text,
        draft,
        diagnostics,
    )


def native_crop_field_value(raw_text: str, field: str) -> str:
    """Select field-like text from an already grounded crop transcription."""
    lines = native_ocr_lines(raw_text)
    if not lines:
        return ""
    pseudo_lines = [
        {"idx": index, "text": text, "bbox": [0.0, float(index), 1.0, float(index + 1)]}
        for index, text in enumerate(lines)
    ]
    best = best_primitive_span(
        pseudo_lines,
        field=field,
        max_window=min(ARGS.primitive_max_window, len(pseudo_lines)),
    )
    return str((best or {}).get("text") or " ".join(lines)).strip()


# -----------------------------------------------------------------------------
# JSON parsing and field extraction
# -----------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def find_balanced_json(text: str) -> Optional[str]:
    text = strip_code_fences(text)
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        stack: List[str] = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    break
                last = stack.pop()
                if (last == "{" and c != "}") or (last == "[" and c != "]"):
                    break
                if not stack:
                    return text[start : i + 1]
    return None


def parse_model_json(raw_output: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    cleaned = find_balanced_json(raw_output) or strip_code_fences(raw_output)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None, isinstance(parsed, dict), cleaned
    except Exception:
        return None, False, cleaned


def parse_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("bbox", "box", "bounding_box", "coordinates"):
            if key in value:
                return parse_bbox(value[key])
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) >= 4:
            return [float(n) for n in nums[:4]]
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(x, (int, float, str)) for x in value):
            try:
                return [float(x) for x in value]
            except Exception:
                return None
        if len(value) >= 8:
            try:
                nums = [float(x) for x in value[:8]]
                xs = nums[0::2]
                ys = nums[1::2]
                return [min(xs), min(ys), max(xs), max(ys)]
            except Exception:
                return None
        # Point list: [[x,y], [x,y], ...]
        if value and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value):
            try:
                xs = [float(p[0]) for p in value]
                ys = [float(p[1]) for p in value]
                return [min(xs), min(ys), max(xs), max(ys)]
            except Exception:
                return None
    return None


def sanitize_bbox(bbox: Any, image_size: Optional[Tuple[int, int]]) -> Optional[List[float]]:
    parsed = parse_bbox(bbox)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed[:4]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    if image_size is not None:
        width, height = image_size
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
    return [x1, y1, x2, y2]


def map_bbox_from_sent_to_original(
    bbox: Optional[List[float]],
    sent_size: Optional[Tuple[int, int]],
    original_size: Optional[Tuple[int, int]],
) -> Optional[List[float]]:
    if bbox is None:
        return None
    if sent_size is None or original_size is None:
        return bbox
    sent_w, sent_h = sent_size
    orig_w, orig_h = original_size
    if sent_w <= 0 or sent_h <= 0:
        return bbox
    sx = orig_w / sent_w
    sy = orig_h / sent_h
    mapped = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    return sanitize_bbox(mapped, original_size)


def extract_field_prediction(
    parsed: Optional[Dict[str, Any]],
    field: str,
    sent_size: Optional[Tuple[int, int]],
    original_size: Optional[Tuple[int, int]],
) -> Tuple[str, Optional[List[float]]]:
    if not isinstance(parsed, dict):
        return "", None

    value: Any = ""
    bbox: Any = None

    obj = parsed.get(field)
    if isinstance(obj, dict):
        for key in ("value", "text", "answer", "prediction", "extracted_value"):
            if key in obj:
                value = obj.get(key)
                break
        bbox = obj.get("bbox") or obj.get("box") or obj.get("bounding_box")
    elif obj is not None:
        value = obj

    # Some models return separate bbox maps.
    for bbox_key in ("bboxes", "boxes", "field_bboxes", "grounding"):
        maybe_map = parsed.get(bbox_key)
        if bbox is None and isinstance(maybe_map, dict) and field in maybe_map:
            bbox = maybe_map[field]

    parsed_bbox = sanitize_bbox(bbox, sent_size)
    mapped_bbox = map_bbox_from_sent_to_original(parsed_bbox, sent_size, original_size)
    return str(value or "").strip(), mapped_bbox


def extract_field_predictions(
    parsed: Optional[Dict[str, Any]],
    field: str,
    sent_size: Optional[Tuple[int, int]],
    original_size: Optional[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Normalize a model's many possible field encodings into occurrences.

    The requested format is a list, but real models commonly return one string,
    one ``{"value", "bbox"}`` object, a list of strings, or a list of objects.
    Accepting those variants prevents JSON-shape brittleness from being confused
    with semantic extraction failure.
    """
    if not isinstance(parsed, dict):
        return []

    raw_field = parsed.get(field, [])
    if raw_field is None or raw_field == "":
        return []
    raw_items = raw_field if isinstance(raw_field, list) else [raw_field]

    # Some models place all boxes in a parallel top-level map.  We use the box at
    # the same occurrence index only when the occurrence itself has no bbox.
    parallel_boxes: Any = None
    for bbox_key in ("bboxes", "boxes", "field_bboxes", "grounding"):
        candidate = parsed.get(bbox_key)
        if isinstance(candidate, dict) and field in candidate:
            parallel_boxes = candidate[field]
            break
    if parallel_boxes is not None and not isinstance(parallel_boxes, list):
        parallel_boxes = [parallel_boxes]

    occurrences: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        value: Any = ""
        bbox: Any = None
        if isinstance(item, dict):
            for key in ("value", "text", "answer", "prediction", "extracted_value"):
                if key in item:
                    value = item.get(key)
                    break
            bbox = item.get("bbox") or item.get("box") or item.get("bounding_box")
        else:
            value = item

        if bbox is None and isinstance(parallel_boxes, list) and index < len(parallel_boxes):
            bbox = parallel_boxes[index]

        text = str(value or "").strip()
        if not text:
            continue
        parsed_bbox = sanitize_bbox(bbox, sent_size)
        mapped_bbox = map_bbox_from_sent_to_original(parsed_bbox, sent_size, original_size)
        occurrences.append({"value": text, "bbox": mapped_bbox})

    return occurrences


# -----------------------------------------------------------------------------
# WildReceipt annotation loading
# -----------------------------------------------------------------------------


def canonical_wr_label(label: Any) -> Optional[str]:
    """Map official/converted class names to this script's stable field names."""
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    replacements = {
        "str": "store",
        "addr": "address",
        "tel": "telephone",
        "phone": "telephone",
        "prod": "product",
        "qty": "quantity",
        "sub_total": "subtotal",
        "tip": "tips",
    }
    parts = [replacements.get(part, part) for part in text.split("_") if part]
    normalized = "_".join(parts).replace("sub_total", "subtotal")

    # The token-level replacement above already converts the official "addr"
    # abbreviation without accidentally rewriting the prefix of "address".
    normalized = normalized.replace("product_name", "product_item")
    normalized = normalized.replace("item_name", "product_item")

    if normalized in {"other", "others", "ignore", "background", "none"}:
        return None
    if normalized in WR_FIELDS:
        return normalized
    return None


def load_class_map(path: str) -> Dict[int, Optional[str]]:
    """Read numeric WildReceipt labels from the dataset's own class list.

    We intentionally do not embed a guessed numeric ordering.  Original and
    converted WildReceipt packages may shift IDs (for example, by inserting an
    ignore/background class).  Reading class_list.txt makes the mapping explicit
    and auditable.
    """
    if not os.path.exists(path):
        # A direct extraction of the official tarball can introduce one or two
        # nested "wildreceipt" directories.  Search only for the expected file
        # name and choose the shallowest match; explicit --class_list still wins.
        matches = glob(
            os.path.join(ARGS.dataset_root, "**", os.path.basename(path)),
            recursive=True,
        )
        if matches:
            path = min(matches, key=lambda candidate: candidate.count(os.sep))
            ARGS.class_list = path
        else:
            raise FileNotFoundError(
                f"WildReceipt class list not found: {path}. Pass --class_list explicitly."
            )

    mapping: Dict[int, Optional[str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Accepted examples include "1 Store_name_value",
            # "1,Store_name_value", and "Store_name_value 1".
            match = re.match(r"^\s*(\d+)\s*[,:\t ]+\s*(.+?)\s*$", line)
            if match:
                label_id = int(match.group(1))
                label_name = match.group(2)
            else:
                match = re.match(r"^\s*(.+?)\s*[,:\t ]+\s*(\d+)\s*$", line)
                if not match:
                    raise ValueError(
                        f"Cannot parse class_list entry at {path}:{line_number}: {line!r}"
                    )
                label_name = match.group(1)
                label_id = int(match.group(2))
            mapping[label_id] = canonical_wr_label(label_name)

    recognized = {field for field in mapping.values() if field is not None}
    missing = [field for field in WR_FIELDS if field not in recognized]
    if missing:
        raise ValueError(
            "class_list.txt does not map every WildReceipt field. "
            f"Missing canonical fields: {missing}. Parsed mapping: {mapping}"
        )
    return mapping


def annotation_files_from_args() -> List[str]:
    """Resolve explicit annotation files or the official train/test split."""
    if ARGS.annotation_file:
        return [os.path.abspath(path) for path in ARGS.annotation_file]
    files: List[str] = []
    if ARGS.split in ("train", "all"):
        files.append(os.path.join(ARGS.dataset_root, "train.txt"))
    if ARGS.split in ("test", "all"):
        files.append(os.path.join(ARGS.dataset_root, "test.txt"))
    resolved: List[str] = []
    for path in files:
        if os.path.exists(path):
            resolved.append(path)
            continue
        matches = glob(
            os.path.join(ARGS.dataset_root, "**", os.path.basename(path)),
            recursive=True,
        )
        resolved.append(
            min(matches, key=lambda candidate: candidate.count(os.sep))
            if matches
            else path
        )
    return resolved


def read_annotation_documents(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Read either JSON-lines files or a JSON array of document records."""
    documents: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"WildReceipt annotation file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read().strip()
        if not content:
            continue

        if content.startswith("["):
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON array in {path}")
            records = parsed
        else:
            records = []
            for line_number, line in enumerate(content.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

        for record in records:
            if not isinstance(record, dict):
                continue
            copied = dict(record)
            copied["_annotation_file"] = path
            documents.append(copied)
    return documents


def resolve_image_path(record: Dict[str, Any]) -> str:
    """Resolve file_name across official and commonly converted layouts."""
    file_name = str(record.get("file_name") or record.get("image") or "").strip()
    if not file_name:
        raise ValueError("WildReceipt record is missing file_name")
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name

    annotation_dir = os.path.dirname(str(record.get("_annotation_file") or ""))
    candidates = [
        os.path.join(ARGS.img_dir, file_name),
        os.path.join(ARGS.dataset_root, file_name),
        os.path.join(annotation_dir, file_name),
        os.path.join(ARGS.img_dir, "image_files", os.path.basename(file_name)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Cannot resolve image {file_name!r}. Checked: {candidates}. "
        "Pass --img_dir if images were moved."
    )


def annotation_bbox(annotation: Dict[str, Any]) -> Optional[List[float]]:
    """Convert a quadrilateral or xyxy annotation into an axis-aligned bbox."""
    raw = annotation.get("box") or annotation.get("bbox") or annotation.get("points")
    if isinstance(raw, dict):
        raw = raw.get("bbox") or raw.get("box") or raw.get("points")
    if not isinstance(raw, (list, tuple)):
        return None

    # A flat 8-number polygon is the official WildReceipt representation.
    if len(raw) >= 8 and all(isinstance(value, (int, float)) for value in raw[:8]):
        xs = [float(value) for value in raw[:8:2]]
        ys = [float(value) for value in raw[1:8:2]]
        return sanitize_bbox([min(xs), min(ys), max(xs), max(ys)], None)

    # Converted datasets may store four [x, y] points.
    if len(raw) >= 4 and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in raw[:4]):
        xs = [float(point[0]) for point in raw[:4]]
        ys = [float(point[1]) for point in raw[:4]]
        return sanitize_bbox([min(xs), min(ys), max(xs), max(ys)], None)

    return sanitize_bbox(raw, None)


def record_instances(
    record: Dict[str, Any],
    class_map: Dict[int, Optional[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Create evaluation instances and label-free OCR lines from one record."""
    grouped: Dict[str, List[Dict[str, Any]]] = {field: [] for field in EVAL_FIELDS}
    ocr_lines: List[Dict[str, Any]] = []

    annotations = record.get("annotations") or record.get("instances") or []
    if not isinstance(annotations, list):
        raise ValueError("WildReceipt record annotations must be a list")

    for source_index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            continue
        text = str(annotation.get("text") or annotation.get("transcription") or "").strip()
        bbox = annotation_bbox(annotation)
        if not text or bbox is None:
            continue

        # This object is the only representation passed into grounding.  Notice
        # that it deliberately contains no label/class/category member.
        ocr_lines.append(
            {
                "idx": source_index,
                "text": text,
                "bbox": bbox,
                "raw": {"text": text, "bbox": bbox},
            }
        )

        raw_label = annotation.get("label", annotation.get("category_id"))
        if isinstance(raw_label, str) and not raw_label.strip().isdigit():
            field = canonical_wr_label(raw_label)
        else:
            try:
                field = class_map.get(int(raw_label))
            except (TypeError, ValueError):
                field = None
        if field in grouped:
            grouped[field].append({"value": text, "bbox": bbox, "source_index": source_index})

    # Both GT lists and OCR lines use one deterministic visual order.  This
    # prevents annotation serialization order from affecting exact match.
    ocr_lines = sort_ocr_lines(ocr_lines)
    for field in grouped:
        grouped[field] = sorted(
            grouped[field],
            key=lambda item: (
                (item["bbox"][1] + item["bbox"][3]) / 2.0,
                item["bbox"][0],
            ),
        )
    return grouped, ocr_lines


# -----------------------------------------------------------------------------
# OCR loading and line geometry
# -----------------------------------------------------------------------------


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [sanitize_bbox(b, None) for b in boxes]
    valid = [b for b in valid if b is not None]
    if not valid:
        return None
    xs1 = [b[0] for b in valid]
    ys1 = [b[1] for b in valid]
    xs2 = [b[2] for b in valid]
    ys2 = [b[3] for b in valid]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def bbox_area(bbox: Sequence[float]) -> float:
    b = sanitize_bbox(bbox, None)
    if b is None:
        return 0.0
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    aa = sanitize_bbox(a, None)
    bb = sanitize_bbox(b, None)
    if aa is None or bb is None:
        return 0.0
    x1 = max(aa[0], bb[0])
    y1 = max(aa[1], bb[1])
    x2 = min(aa[2], bb[2])
    y2 = min(aa[3], bb[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def line_intersects_box(line: Dict[str, Any], box: Sequence[float], min_frac: float = 0.10) -> bool:
    line_box = line.get("bbox")
    if not line_box:
        return False
    inter = intersection_area(line_box, box)
    if inter <= 0:
        return False
    return inter / max(1.0, bbox_area(line_box)) >= min_frac


def lines_inside_box(ocr_lines: Sequence[Dict[str, Any]], box: Optional[Sequence[float]]) -> List[Dict[str, Any]]:
    if box is None:
        return []
    return [line for line in ocr_lines if line_intersects_box(line, box)]


def sort_ocr_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(line: Dict[str, Any]) -> Tuple[float, float]:
        x1, y1, x2, y2 = line["bbox"]
        return ((y1 + y2) / 2.0, x1)

    return sorted(lines, key=key)


def parse_sroie_box_line(raw_line: str, idx: int) -> Optional[Dict[str, Any]]:
    line = raw_line.rstrip("\n")
    if not line.strip():
        return None
    parts = line.split(",")
    if len(parts) >= 9:
        try:
            coords = [float(x.strip()) for x in parts[:8]]
            text = ",".join(parts[8:]).strip()
            xs = coords[0::2]
            ys = coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            return {"idx": idx, "text": text, "bbox": bbox, "raw": line}
        except Exception:
            pass

    # Fallback: try tab-separated bbox and text.
    tab_parts = line.split("\t")
    if len(tab_parts) >= 2:
        nums = re.findall(r"-?\d+(?:\.\d+)?", tab_parts[0])
        if len(nums) >= 4:
            bbox = [float(n) for n in nums[:4]]
            text = "\t".join(tab_parts[1:]).strip()
            return {"idx": idx, "text": text, "bbox": bbox, "raw": line}

    return None


def load_ocr_lines(ocr_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(ocr_path):
        return []
    lines: List[Dict[str, Any]] = []
    with open(ocr_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
    if not content:
        return []

    # Optional JSON OCR support.
    if content.startswith("{") or content.startswith("["):
        try:
            parsed = json.loads(content)
            items: Iterable[Any]
            if isinstance(parsed, dict):
                items = parsed.get("lines") or parsed.get("ocr") or parsed.get("words") or []
            else:
                items = parsed
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or item.get("value") or "").strip()
                # annotation_bbox accepts xyxy, flat quadrilaterals, and point
                # lists, making external OCR JSON as flexible as dataset input.
                bbox = annotation_bbox(item)
                if text and bbox:
                    lines.append({"idx": idx, "text": text, "bbox": bbox, "raw": item})
            return sort_ocr_lines(lines)
        except Exception:
            pass

    for idx, raw_line in enumerate(content.splitlines()):
        parsed_line = parse_sroie_box_line(raw_line, idx)
        if parsed_line and parsed_line.get("text"):
            lines.append(parsed_line)
    return sort_ocr_lines(lines)


def line_span_text(lines: Sequence[Dict[str, Any]]) -> str:
    return " ".join(str(line.get("text") or "").strip() for line in lines if str(line.get("text") or "").strip())


def make_span(lines: Sequence[Dict[str, Any]], start: int, end: int) -> Dict[str, Any]:
    selected = list(lines[start : end + 1])
    return {
        "start": start,
        "end": end,
        "line_indices": [line["idx"] for line in selected],
        "text": line_span_text(selected),
        "bbox": union_boxes([line["bbox"] for line in selected]),
        "lines": selected,
    }


def generate_candidate_spans(
    lines: Sequence[Dict[str, Any]],
    max_window: int,
    allowed_range: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    if not lines:
        return []
    n = len(lines)
    if allowed_range is None:
        left, right = 0, n - 1
    else:
        left = max(0, allowed_range[0])
        right = min(n - 1, allowed_range[1])
    spans: List[Dict[str, Any]] = []
    for i in range(left, right + 1):
        for j in range(i, min(right, i + max_window - 1) + 1):
            span = make_span(lines, i, j)
            if span["text"].strip():
                spans.append(span)
    return spans


# -----------------------------------------------------------------------------
# OCR evidence scoring
# -----------------------------------------------------------------------------


def char_ngrams(text: str, n: int = 3) -> set:
    text = re.sub(r"\s+", " ", normalize_for_match(text))
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_for_match(a).split())
    tb = set(normalize_for_match(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_recall(candidate: str, target: str) -> float:
    c = Counter(normalize_for_match(candidate).split())
    t = Counter(normalize_for_match(target).split())
    if not t:
        return 0.0
    hit = sum(min(c[k], v) for k, v in t.items())
    total = sum(t.values())
    return hit / total if total else 0.0


def fuzzy_similarity(candidate: str, target: str) -> float:
    c = normalize_for_match(candidate)
    t = normalize_for_match(target)
    if not c or not t:
        return 0.0
    if c == t:
        return 1.0
    seq = difflib.SequenceMatcher(None, c, t).ratio()
    jac = token_jaccard(c, t)
    rec = token_recall(c, t)
    ng_c = char_ngrams(c)
    ng_t = char_ngrams(t)
    ng = len(ng_c & ng_t) / len(ng_c | ng_t) if ng_c and ng_t else 0.0
    contains = 1.0 if (t in c or c in t) else 0.0
    return max(0.0, min(1.0, 0.30 * seq + 0.25 * jac + 0.30 * rec + 0.10 * ng + 0.05 * contains))


def has_date_pattern(text: str) -> bool:
    t = normalize_for_match(text)
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}\s*,?\s*\d{2,4}\b",
    ]
    return any(re.search(p, t) for p in patterns)


def money_amounts(text: str) -> List[str]:
    t = normalize_space(text)
    patterns = [
        r"\b(?:rm|myr|usd|rs|inr|rp|eur|gbp|aud|cad|\$|€|£)?\s*\d{1,8}(?:[,.]\d{2})\b",
        r"\b\d{1,8}(?:[,.]\d{2})\s*(?:rm|myr|usd|rs|inr|rp|eur|gbp|aud|cad)?\b",
    ]
    found: List[str] = []
    for p in patterns:
        found.extend(re.findall(p, t, flags=re.IGNORECASE))
    return found


def field_concept(field: str) -> str:
    return field.removesuffix("_key").removesuffix("_value")


def field_is_key(field: str) -> bool:
    return field.endswith("_key")


FIELD_MARKERS: Dict[str, Tuple[str, ...]] = {
    "store_name": ("store", "merchant", "company", "restaurant", "market"),
    "store_address": ("address", "addr", "street", "road", "avenue"),
    "telephone": ("tel", "telephone", "phone", "contact"),
    "date": ("date", "dated"),
    "time": ("time",),
    "product_item": ("item", "description", "product", "article"),
    "product_quantity": ("qty", "quantity", "quan"),
    "product_price": ("price", "amount", "unit price"),
    "subtotal": ("subtotal", "sub total", "net"),
    "tax": ("tax", "vat", "gst", "sst"),
    "tips": ("tip", "tips", "gratuity", "service charge"),
    "total": ("total", "grand total", "amount due", "balance due", "payable"),
}


def marker_hits(text: str, concept: str) -> int:
    return sum(
        1
        for marker in FIELD_MARKERS.get(concept, ())
        if re.search(rf"\b{re.escape(marker)}\b", text)
    )


def contamination_penalty(text: str, field: str) -> float:
    t = normalize_for_match(text)
    concept = field_concept(field)
    penalty = 0.0

    # Key fields should be short labels, whereas values should generally not
    # include a competing field's label.  The penalties are intentionally soft:
    # value-to-OCR fuzzy matching remains the primary grounding mechanism.
    if field_is_key(field):
        if len(t.split()) > 5:
            penalty += 0.15
        if money_amounts(t):
            penalty += 0.25
    elif marker_hits(t, concept):
        penalty += 0.08

    competing = sum(
        marker_hits(t, other)
        for other in FIELD_MARKERS
        if other != concept
    )
    penalty += min(0.30, 0.06 * competing)

    if concept in {"store_name", "store_address"} and money_amounts(t):
        penalty += 0.20
    if concept == "store_address" and len(money_amounts(t)) >= 2:
        penalty += 0.20
    if concept in {"date", "time", "telephone"} and money_amounts(t):
        penalty += 0.12
    if concept == "total" and any(word in t for word in ("change", "cash", "tender")):
        penalty += 0.25
    if concept == "total" and marker_hits(t, "subtotal"):
        penalty += 0.20
    if concept in {"subtotal", "tax", "tips"} and marker_hits(t, "total") and not marker_hits(t, concept):
        penalty += 0.15
    return min(0.75, penalty)


def field_semantic_score(text: str, field: str, start: int = 0, num_lines: int = 1, total_lines: int = 1) -> float:
    """Heuristic field-conditioned OCR-span score for primitive proposals.

    This is deliberately one lightweight OCR nudge, not a second VLM. Replace this
    function with a learned text classifier later if you want a stronger paper setup.
    """
    raw = str(text or "")
    t = normalize_for_match(raw)
    if not t:
        return 0.0

    tokens = t.split()
    score = 0.0
    concept = field_concept(field)

    if field_is_key(field):
        # Key categories are usually compact lexical labels.  The marker score
        # is concept-specific, so "TAX" cannot become a "total_key" proposal.
        score += min(0.85, 0.45 * marker_hits(t, concept))
        if len(tokens) <= 4:
            score += 0.10
        if concept in {"product_item", "product_quantity", "product_price"}:
            # Product headers tend to appear before the receipt's middle body.
            relative_position = start / max(1, total_lines)
            if 0.15 <= relative_position <= 0.65:
                score += 0.05

    elif concept == "store_address":
        address_markers = [
            "jln",
            "jalan",
            "road",
            "rd",
            "street",
            "st",
            "ave",
            "avenue",
            "taman",
            "tmn",
            "lorong",
            "lrg",
            "persiaran",
            "no",
            "lot",
            "block",
            "floor",
            "shah",
            "alam",
            "kuala",
            "lumpur",
            "selangor",
            "malaysia",
            "boulevard",
            "drive",
            "lane",
            "suite",
            "city",
            "state",
            "zip",
        ]
        address_marker_count = sum(
            1 for marker in address_markers if re.search(rf"\b{re.escape(marker)}\b", t)
        )
        score += min(0.40, 0.10 * address_marker_count)
        if re.search(r"\b\d{5}\b", t):
            score += 0.25
        if re.search(r"\b\d+[a-z]?([/-]\d+[a-z]?)*\b", t):
            score += 0.15
        if "," in raw or num_lines >= 2:
            score += 0.10
        if len(tokens) >= 4:
            score += 0.10
        # Small top prior only, not a hard template.
        if total_lines > 0 and start / max(1, total_lines) < 0.35:
            score += 0.05

    elif concept == "store_name":
        company_markers = [
            "co",
            "company",
            "trading",
            "enterprise",
            "sdn",
            "bhd",
            "ltd",
            "limited",
            "restaurant",
            "restoran",
            "mart",
            "market",
            "store",
            "supermarket",
            "hardware",
            "pharmacy",
            "shop",
            "services",
        ]
        company_marker_count = sum(
            1 for marker in company_markers if re.search(rf"\b{re.escape(marker)}\b", t)
        )
        score += min(0.45, 0.15 * company_marker_count)
        letters = [c for c in raw if c.isalpha()]
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        if uppercase_ratio > 0.70 and len(tokens) == 1 and len(tokens[0]) >= 4:
            score += 0.20
        elif uppercase_ratio > 0.55 and len(tokens) >= 2:
            score += 0.20
        if 1 <= len(tokens) <= 8:
            score += 0.10
        if total_lines > 0 and start / max(1, total_lines) < 0.25:
            score += 0.10

    elif concept == "telephone":
        digits = re.sub(r"\D", "", raw)
        if 7 <= len(digits) <= 16:
            score += 0.65
        if re.search(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]\d{3,4}", raw):
            score += 0.20
        if marker_hits(t, "telephone"):
            score += 0.08

    elif concept == "date":
        if has_date_pattern(t):
            score += 0.75
        if any(m in t for m in ["date", "dated"]):
            score += 0.12
        if len(tokens) <= 8:
            score += 0.08

    elif concept == "time":
        if re.search(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d(?:[:.][0-5]\d)?\s*(?:am|pm)?\b", t):
            score += 0.78
        if marker_hits(t, "time"):
            score += 0.10
        if len(tokens) <= 6:
            score += 0.07

    elif concept == "product_item":
        letter_count = sum(character.isalpha() for character in raw)
        digit_count = sum(character.isdigit() for character in raw)
        if letter_count >= 3:
            score += 0.35
        if letter_count > digit_count:
            score += 0.15
        relative_position = start / max(1, total_lines)
        if 0.15 <= relative_position <= 0.80:
            score += 0.12
        if not money_amounts(t):
            score += 0.10

    elif concept == "product_quantity":
        if re.fullmatch(r"\s*\d{1,3}\s*", raw):
            score += 0.65
        elif re.search(r"\b\d+(?:[.,]\d+)?\s*[xX]\b|\b[xX]\s*\d+(?:[.,]\d+)?\b", raw):
            score += 0.65
        if len(tokens) <= 3:
            score += 0.12

    elif concept in {"product_price", "subtotal", "tax", "tips", "total"}:
        if marker_hits(t, concept):
            score += 0.35
        amounts = money_amounts(t)
        if amounts:
            score += 0.40
        # When a key and amount share a candidate window, prefer the natural
        # receipt order "TOTAL 12.34" over the boundary-crossing "9.99 TOTAL".
        markers = FIELD_MARKERS.get(concept, ())
        if amounts and any(
            re.search(
                rf"\b{re.escape(marker)}\b.*\d",
                raw,
                flags=re.IGNORECASE,
            )
            for marker in markers
        ):
            score += 0.12
        if re.search(r"\b(rm|myr|usd|rs|inr|rp|eur|gbp|aud|cad)\b|[$€£]", raw, flags=re.IGNORECASE):
            score += 0.10
        if concept in {"subtotal", "tax", "tips", "total"} and total_lines > 0 and start / max(1, total_lines) > 0.45:
            score += 0.05
        if len(amounts) > 3:
            score -= 0.10

    else:
        # This branch should be rare, but retaining it makes custom --eval_fields
        # degrade gracefully instead of crashing.
        if len(tokens) >= 1:
            score += 0.20

    score -= contamination_penalty(t, field)
    # Penalize overly long primitive spans; they are rarely minimal evidence.
    if len(tokens) > 25:
        score -= min(0.25, (len(tokens) - 25) / 100.0)
    return max(0.0, min(1.0, score))


def score_span_for_value(span: Dict[str, Any], field: str, value: str) -> float:
    normalized_target = normalize_field(field, value)
    normalized_candidate = normalize_field(field, span.get("text", ""))
    return fuzzy_similarity(normalized_candidate, normalized_target)


def best_value_span(
    ocr_lines: Sequence[Dict[str, Any]],
    field: str,
    value: str,
    max_window: int,
    allowed_range: Optional[Tuple[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    spans = generate_candidate_spans(ocr_lines, max_window=max_window, allowed_range=allowed_range)
    if not spans:
        return None
    best = max(spans, key=lambda s: score_span_for_value(s, field, value))
    best["score"] = score_span_for_value(best, field, value)
    return best


def best_primitive_span(
    ocr_lines: Sequence[Dict[str, Any]],
    field: str,
    max_window: int,
) -> Optional[Dict[str, Any]]:
    spans = generate_candidate_spans(ocr_lines, max_window=max_window)
    if not spans:
        return None
    total_lines = len(ocr_lines)
    for span in spans:
        span["score"] = field_semantic_score(
            span.get("text", ""),
            field=field,
            start=int(span.get("start", 0)),
            num_lines=int(span.get("end", 0)) - int(span.get("start", 0)) + 1,
            total_lines=total_lines,
        )
    best = max(spans, key=lambda s: s["score"])
    return prune_boundary_lines(best, ocr_lines, field, score_mode="primitive")


def primitive_minimum_score(field: str) -> float:
    """Return a precision-oriented minimum for field-only OCR proposal.

    A universal 0.25 threshold was reasonable for four almost-always-present
    SROIE fields.  It is not reasonable for 24 sparse WildReceipt categories:
    the same unlabeled ``12.34`` could otherwise populate time, quantity, price,
    subtotal, tax, tips, and total simultaneously.  These minima still use the
    same semantic scorer; they merely reflect how distinctive each field is.
    """
    concept = field_concept(field)
    if field_is_key(field):
        return max(ARGS.primitive_threshold, 0.45)
    thresholds = {
        "store_name": 0.35,
        "store_address": 0.40,
        "telephone": 0.60,
        "date": 0.70,
        "time": 0.75,
        "product_item": 0.72,
        "product_quantity": 0.72,
        "product_price": 0.45,
        "subtotal": 0.65,
        "tax": 0.65,
        "tips": 0.65,
        "total": 0.65,
    }
    return max(ARGS.primitive_threshold, thresholds.get(concept, 0.50))


def best_primitive_spans(
    ocr_lines: Sequence[Dict[str, Any]],
    field: str,
    max_window: int,
    max_instances: int,
) -> List[Dict[str, Any]]:
    """Return high-scoring, non-overlapping primitive proposals.

    SROIE needed at most one proposal per field.  WildReceipt product fields
    repeat, so selecting only one candidate would artificially cap recall.  We
    rank every contiguous OCR window, prune its boundary, and greedily retain
    spans that do not reuse an already-selected OCR line.
    """
    spans = generate_candidate_spans(ocr_lines, max_window=max_window)
    total_lines = len(ocr_lines)
    for span in spans:
        span["score"] = field_semantic_score(
            span.get("text", ""),
            field=field,
            start=int(span.get("start", 0)),
            num_lines=int(span.get("end", 0)) - int(span.get("start", 0)) + 1,
            total_lines=total_lines,
        )

    selected: List[Dict[str, Any]] = []
    used_lines: set = set()
    seen_text: set = set()
    minimum_score = primitive_minimum_score(field)
    for candidate in sorted(spans, key=lambda item: item["score"], reverse=True):
        if candidate["score"] < minimum_score:
            break
        pruned = prune_boundary_lines(candidate, ocr_lines, field, score_mode="primitive")
        pruned["score"] = field_semantic_score(
            pruned.get("text", ""),
            field=field,
            start=int(pruned.get("start", 0)),
            num_lines=int(pruned.get("end", 0)) - int(pruned.get("start", 0)) + 1,
            total_lines=total_lines,
        )
        indices = set(pruned.get("line_indices") or [])
        normalized_text = normalize_for_match(pruned.get("text", ""))
        if (
            pruned["score"] < minimum_score
            or not indices
            or indices & used_lines
            or not normalized_text
            or normalized_text in seen_text
        ):
            continue
        selected.append(pruned)
        used_lines.update(indices)
        seen_text.add(normalized_text)
        if len(selected) >= max_instances:
            break

    # Restore visual order after confidence-ranked selection so predictions and
    # ground truth use the same deterministic occurrence order.
    return sorted(
        selected,
        key=lambda item: (
            item.get("bbox", [0, 0, 0, 0])[1],
            item.get("bbox", [0, 0, 0, 0])[0],
        ),
    )


def prune_boundary_lines(
    span: Dict[str, Any],
    ocr_lines: Sequence[Dict[str, Any]],
    field: str,
    score_mode: str = "primitive",
    value: str = "",
) -> Dict[str, Any]:
    """Remove unnecessary boundary lines when score does not decrease."""
    if not span:
        return span
    current = dict(span)
    total_lines = len(ocr_lines)

    def span_score(s: Dict[str, Any]) -> float:
        if score_mode == "value":
            return score_span_for_value(s, field, value)
        return field_semantic_score(
            s.get("text", ""),
            field=field,
            start=int(s.get("start", 0)),
            num_lines=int(s.get("end", 0)) - int(s.get("start", 0)) + 1,
            total_lines=total_lines,
        )

    current["score"] = span_score(current)
    improved = True
    while improved and int(current["end"]) > int(current["start"]):
        improved = False
        candidates: List[Dict[str, Any]] = []
        start = int(current["start"])
        end = int(current["end"])
        if start + 1 <= end:
            candidates.append(make_span(ocr_lines, start + 1, end))
        if start <= end - 1:
            candidates.append(make_span(ocr_lines, start, end - 1))
        if not candidates:
            break
        for cand in candidates:
            cand["score"] = span_score(cand)
        best = max(candidates, key=lambda s: s["score"])
        # Allow tiny numerical loss to remove obvious boundary contamination.
        if best["score"] >= current["score"] - 1e-6:
            current = best
            improved = True
    return current


# -----------------------------------------------------------------------------
# Grounding logic
# -----------------------------------------------------------------------------


def verify_crop_with_vlm(image_path: str, field: str, bbox: Optional[Sequence[float]]) -> Dict[str, Any]:
    global crop_verification_calls
    if not ARGS.verify_crops or bbox is None:
        return {"attempted": False, "supports_field": None, "value": "", "raw_output": ""}
    if crop_verification_calls >= ARGS.max_crop_verifications:
        return {"attempted": False, "skipped_reason": "max_crop_verifications_reached", "supports_field": None, "value": "", "raw_output": ""}
    crop_verification_calls += 1
    try:
        if use_ocr_native_adapter():
            crop_image, crop_size, crop_box = crop_image_to_pil(
                image_path,
                bbox,
                padding=ARGS.crop_padding,
                max_size=ARGS.max_image_size,
            )
            raw, _response = call_ocr_native_image(crop_image)
            value = native_crop_field_value(raw, field)
            supports = bool(value)
        else:
            crop_url, crop_size, crop_box = crop_image_to_data_url(
                image_path, bbox, padding=ARGS.crop_padding, max_size=ARGS.max_image_size
            )
            raw, _response = call_vllm(crop_url, crop_verification_prompt(field))
            parsed, ok, _cleaned = parse_model_json(raw)
            supports = None
            value = ""
            if ok and isinstance(parsed, dict):
                supports = parsed.get("supports_field")
                if isinstance(supports, str):
                    supports = supports.strip().lower() in ("true", "yes", "1")
                elif supports is not None:
                    supports = bool(supports)
                value = str(parsed.get("value") or "").strip()
        return {
            "attempted": True,
            "supports_field": supports,
            "value": value,
            "raw_output": raw,
            "crop_size": crop_size,
            "crop_box_with_padding": crop_box,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "supports_field": None,
            "value": "",
            "raw_output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def line_position_range_for_box(
    ocr_lines: Sequence[Dict[str, Any]],
    box: Optional[Sequence[float]],
    context: int,
) -> Optional[Tuple[int, int]]:
    if box is None:
        return None
    indices = [i for i, line in enumerate(ocr_lines) if line_intersects_box(line, box)]
    if not indices:
        return None
    return (min(indices) - context, max(indices) + context)


def ground_field(
    image_path: str,
    field: str,
    vlm_value: str,
    vlm_bbox: Optional[List[float]],
    ocr_lines: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ground a single field using the one OCR-nudge mechanism."""
    value = str(vlm_value or "").strip()
    details: Dict[str, Any] = {
        "field": field,
        "vlm_value": value,
        "vlm_bbox": vlm_bbox,
        "final_value": value,
        "final_bbox": None,
        "evidence_text": "",
        "evidence_score": 0.0,
        "status": "not_grounded",
        "source": "vlm" if value else "primitive_ocr_span",
        "pre_crop_final_value": value,
        "crop_verification": {"attempted": False, "supports_field": None, "value": "", "raw_output": ""},
    }

    if ARGS.disable_grounding:
        details["status"] = "grounding_disabled"
        return details

    if not ocr_lines:
        details["status"] = "no_ocr_available"
        return details

    max_window = ARGS.primitive_max_window

    # Case 1: VLM generated a field value. Use OCR to verify/refine/recover WHERE.
    if value:
        inside_lines = lines_inside_box(ocr_lines, vlm_bbox)
        inside_text = line_span_text(inside_lines)
        inside_score = fuzzy_similarity(
            normalize_field(field, inside_text),
            normalize_field(field, value),
        )
        details["vlm_bbox_evidence_text"] = inside_text
        details["vlm_bbox_evidence_score"] = inside_score

        chosen_span: Optional[Dict[str, Any]] = None

        if vlm_bbox is not None and inside_score >= ARGS.evidence_high_threshold and inside_lines:
            # The VLM bbox is already evidence-supported. Use OCR-line union as the minimal evidence region.
            chosen_span = {
                "text": inside_text,
                "bbox": union_boxes([line["bbox"] for line in inside_lines]),
                "lines": inside_lines,
                "score": inside_score,
                "start": min((line["idx"] for line in inside_lines), default=0),
                "end": max((line["idx"] for line in inside_lines), default=0),
                "line_indices": [line["idx"] for line in inside_lines],
            }
            details["status"] = "grounded_from_vlm_bbox"

        elif vlm_bbox is not None and inside_score >= ARGS.evidence_partial_threshold:
            # The VLM bbox is close but incomplete/contaminated. Search a local band around it.
            allowed = line_position_range_for_box(ocr_lines, vlm_bbox, ARGS.local_context_lines)
            local_best = best_value_span(ocr_lines, field, value, max_window=max_window, allowed_range=allowed)
            if local_best is not None:
                local_best = prune_boundary_lines(local_best, ocr_lines, field, score_mode="value", value=value)
                local_best["score"] = score_span_for_value(local_best, field, value)
            chosen_span = local_best
            details["status"] = "locally_refined_from_vlm_bbox"

        else:
            # The VLM bbox is absent or way off. Search the whole OCR text for evidence supporting the value.
            global_best = best_value_span(ocr_lines, field, value, max_window=max_window)
            if global_best is not None:
                global_best = prune_boundary_lines(global_best, ocr_lines, field, score_mode="value", value=value)
                global_best["score"] = score_span_for_value(global_best, field, value)
            chosen_span = global_best
            details["status"] = "globally_recovered_from_vlm_value"

        if chosen_span is not None:
            details["final_bbox"] = chosen_span.get("bbox")
            details["evidence_text"] = chosen_span.get("text", "")
            details["evidence_score"] = float(chosen_span.get("score", 0.0))
            details["evidence_line_indices"] = chosen_span.get("line_indices", [])

        # If even global recovery has almost no evidence, explicitly mark as ungrounded.
        if details["evidence_score"] < ARGS.evidence_partial_threshold:
            details["status"] = "ungrounded_vlm_value"

    # Case 2: VLM generated nothing. Use primitive OCR-span proposal.
    else:
        primitive = best_primitive_span(ocr_lines, field, max_window=max_window)
        if primitive is not None:
            details["final_bbox"] = primitive.get("bbox")
            details["evidence_text"] = primitive.get("text", "")
            details["evidence_score"] = float(primitive.get("score", 0.0))
            details["evidence_line_indices"] = primitive.get("line_indices", [])
            if details["evidence_score"] >= ARGS.primitive_threshold:
                details["status"] = "primitive_ocr_proposal"
                details["final_value"] = primitive.get("text", "")
            else:
                details["status"] = "primitive_low_confidence"
                details["final_value"] = ""
        else:
            details["status"] = "primitive_failed_no_candidate"

    # Optional crop-level VLM verification/extraction.
    if details.get("final_bbox") is not None:
        details["pre_crop_final_value"] = details.get("final_value", "")
        crop_check = verify_crop_with_vlm(image_path, field, details["final_bbox"])
        details["crop_verification"] = crop_check
        crop_value = str(crop_check.get("value") or "").strip()
        if crop_check.get("supports_field") is True and crop_value:
            if ARGS.refine_crop_values:
                details["final_value"] = crop_value
                if value:
                    details["status"] = details["status"] + "_crop_refined"
                else:
                    details["status"] = "primitive_ocr_proposal_vlm_verified"
            elif not value:
                details["status"] = "primitive_ocr_proposal_vlm_verified_not_applied"
            else:
                details["status"] = details["status"] + "_crop_verified_not_applied"
        elif value and crop_check.get("supports_field") is False:
            details["status"] = details["status"] + "_crop_rejected"

    return details


def ground_field_instances(
    image_path: str,
    field: str,
    predicted: Sequence[Dict[str, Any]],
    ocr_lines: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the original grounding state machine to every field occurrence."""
    grounded: List[Dict[str, Any]] = []
    used_line_indices: set = set()

    if predicted:
        for occurrence_index, occurrence in enumerate(predicted):
            # Duplicate textual values are common in quantity and price columns.
            # Removing OCR lines consumed by earlier occurrences prevents every
            # identical prediction from grounding to the first matching row.
            available_lines = [
                line for line in ocr_lines if line.get("idx") not in used_line_indices
            ]
            details = ground_field(
                image_path=image_path,
                field=field,
                vlm_value=str(occurrence.get("value") or ""),
                vlm_bbox=occurrence.get("bbox"),
                ocr_lines=available_lines,
            )
            details["occurrence_index"] = occurrence_index
            grounded.append(details)
            used_line_indices.update(details.get("evidence_line_indices") or [])
        return grounded

    if ARGS.disable_grounding:
        return []

    # The VLM omitted the entire field.  Propose one or more non-overlapping OCR
    # regions and pass each through the SAME crop-verification/refinement path.
    proposals = best_primitive_spans(
        ocr_lines,
        field=field,
        max_window=ARGS.primitive_max_window,
        # Only row-level product values are expected to repeat heavily.  Capping
        # every other omitted field at one primitive avoids manufacturing many
        # speculative totals, dates, or label keys from a single receipt.
        max_instances=(
            ARGS.max_primitive_instances
            if field in {
                "product_item_value",
                "product_quantity_value",
                "product_price_value",
            }
            else 1
        ),
    )
    for occurrence_index, proposal in enumerate(proposals):
        details = ground_field(
            image_path=image_path,
            field=field,
            vlm_value="",
            vlm_bbox=None,
            ocr_lines=[
                line
                for line in proposal.get("lines", [])
                if line.get("idx") not in used_line_indices
            ],
        )
        # ground_field's empty-value branch recomputes the best primitive over
        # this proposal's lines, retaining all original thresholds and crop logic.
        details["occurrence_index"] = occurrence_index
        details["source"] = "primitive_ocr_span"
        grounded.append(details)
        used_line_indices.update(details.get("evidence_line_indices") or [])
    return grounded


def occurrence_values(instances: Sequence[Dict[str, Any]], key: str = "final_value") -> List[str]:
    """Extract non-empty values from normalized occurrence dictionaries."""
    values: List[str] = []
    for instance in instances:
        value = str(instance.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def normalize_instance_values(field: str, values: Sequence[Any]) -> List[str]:
    """Normalize each occurrence while preserving multiplicity and order."""
    return [
        normalized
        for normalized in (normalize_field(field, value) for value in values)
        if normalized
    ]


def instances_as_metric_text(values: Sequence[Any]) -> str:
    """Flatten occurrences only for token P/R/F1; exact match stays list-wise."""
    return " ".join(str(value or "").strip() for value in values if str(value or "").strip())


def serialize_fields(values: Dict[str, Any], fields: List[str]) -> str:
    rows = []
    for field in fields:
        raw_values = values.get(field, [])
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        rows.append(f"{field}: {normalize_instance_values(field, raw_values)}")
    return "\n".join(rows)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


def main() -> None:
    unknown_fields = [field for field in EVAL_FIELDS if field not in WR_FIELDS]
    if unknown_fields:
        raise ValueError(
            f"Unknown --eval_fields entries: {unknown_fields}. Supported fields: {list(WR_FIELDS)}"
        )

    class_map = load_class_map(ARGS.class_list)
    annotation_paths = annotation_files_from_args()
    records = read_annotation_documents(annotation_paths)
    if ARGS.num_samples is not None:
        records = records[: ARGS.num_samples]

    print(f"\nTOTAL IMAGES TO PROCESS: {len(records)}\n")
    print(f"Model preset: {ARGS.model_preset} ({ARGS.model_label})")
    print(f"Model name  : {ARGS.model}")
    print(f"Served name : {request_model_name()}")
    print(f"WildReceipt split: {ARGS.split}")
    print(f"Annotation files: {annotation_paths}")
    print(f"Class list: {ARGS.class_list}")
    print(f"Evaluated fields ({len(EVAL_FIELDS)}): {', '.join(EVAL_FIELDS)}")
    print(f"OCR source: {ARGS.ocr_source}")
    if ARGS.ocr_source == "dir":
        print(f"OCR directory: {ARGS.ocr_dir}")
    else:
        print("OCR label isolation: semantic labels are stripped before grounding")
    print(f"Grounding enabled: {not ARGS.disable_grounding}")
    print(f"Crop verification: {ARGS.verify_crops}")
    print(f"Crop value refinement: {ARGS.refine_crop_values}\n")
    if use_ocr_native_adapter():
        print("OCR-native adapter: enabled")
        print("Semantic draft: native OCR -> fuzzy layout alignment -> field heuristics")
        if is_deepseek_ocr_model():
            print(f"DeepSeek OCR prompt: {ARGS.deepseek_ocr_prompt!r}")
        elif is_mistral_ocr_model():
            print("Mistral OCR backend: client.ocr.process")
        print()

    if records and use_ocr_native_adapter():
        print("Using OCR-native backend directly; no HTTP vLLM server will be spawned.\n")
        initialize_ocr_backend()
    elif records:
        start_vllm_server()

    all_results: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        try:
            image_path = resolve_image_path(record)
            gt_instances, annotation_ocr_lines = record_instances(record, class_map)
        except Exception as exc:
            print(f"\nDATASET RECORD {idx + 1} LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        image_name = str(record.get("file_name") or os.path.basename(image_path))
        file_id = os.path.splitext(os.path.basename(image_path))[0]

        print("\n====================================================")
        print(f"[{idx + 1}/{len(records)}] PROCESSING: {image_name}")
        print("====================================================\n")

        # Evaluation sees semantic labels through gt_instances.  Grounding sees
        # either independent OCR predictions or annotation_ocr_lines, whose
        # dictionaries contain text+bbox but no semantic category.
        ocr_path: Optional[str] = None
        if ARGS.ocr_source == "annotations":
            ocr_lines = annotation_ocr_lines
            ocr_source_description = "WildReceipt annotations with labels stripped"
        else:
            candidates = [
                os.path.join(str(ARGS.ocr_dir), file_id + extension)
                for extension in (".json", ".txt")
            ]
            ocr_path = next((path for path in candidates if os.path.exists(path)), candidates[-1])
            ocr_lines = load_ocr_lines(ocr_path)
            ocr_source_description = ocr_path
        print(f"OCR lines loaded: {len(ocr_lines)} from {ocr_source_description}")

        gt_values: Dict[str, List[str]] = {
            field: [str(item.get("value") or "") for item in gt_instances.get(field, [])]
            for field in EVAL_FIELDS
        }

        raw_output = ""
        response_json: Dict[str, Any] = {}
        pred_json: Optional[Dict[str, Any]] = None
        parse_success = False
        cleaned_output = ""
        original_size: Optional[Tuple[int, int]] = None
        sent_size: Optional[Tuple[int, int]] = None
        image_target_max = ""
        ocr_native_text = ""
        ocr_native_draft: Optional[Dict[str, List[Dict[str, Any]]]] = None
        native_alignment: List[Dict[str, Any]] = []

        try:
            if use_ocr_native_adapter():
                (
                    raw_output,
                    response_json,
                    original_size,
                    sent_size,
                    image_target_max,
                    ocr_native_text,
                    ocr_native_draft,
                    native_alignment,
                ) = run_ocr_native_image(image_path, ocr_lines)
                print("\nOCR-NATIVE TEXT OUTPUT:\n")
                print(ocr_native_text)
                print("\nOCR-NATIVE SEMANTIC DRAFT JSON:\n")
                print(raw_output)
                matched = sum(1 for item in native_alignment if item.get("matched"))
                print(
                    f"\nOCR alignment: {matched}/{len(native_alignment)} "
                    f"native lines matched to label-free geometry"
                )
            else:
                raw_output, response_json, original_size, sent_size, image_target_max = run_vllm_image(image_path)
                print("\nRAW OUTPUT:\n")
                print(raw_output)
        except Exception as exc:
            print("MODEL INFERENCE FAILED")
            print(type(exc).__name__, exc)

        if raw_output.strip():
            pred_json, parse_success, cleaned_output = parse_model_json(raw_output)
        else:
            print("EMPTY MODEL OUTPUT")

        vlm_field_instances: Dict[str, List[Dict[str, Any]]] = {}
        vlm_field_values: Dict[str, List[str]] = {}
        vlm_field_bboxes: Dict[str, List[Optional[List[float]]]] = {}
        grounded_fields: Dict[str, List[Dict[str, Any]]] = {}
        final_values: Dict[str, List[str]] = {}
        # Native adapter bboxes come from original-image layout OCR rather than
        # from the resized image sent to the OCR model, so they must not be scaled.
        prediction_coordinate_size = original_size if use_ocr_native_adapter() else sent_size

        for field in EVAL_FIELDS:
            predicted = extract_field_predictions(
                pred_json,
                field,
                prediction_coordinate_size,
                original_size,
            )
            vlm_field_instances[field] = predicted
            vlm_field_values[field] = [item["value"] for item in predicted]
            vlm_field_bboxes[field] = [item.get("bbox") for item in predicted]
            grounded = ground_field_instances(image_path, field, predicted, ocr_lines)
            grounded_fields[field] = grounded
            final_values[field] = occurrence_values(grounded)

            print(
                f"GROUNDING {field}: base_instances={len(predicted)} "
                f"final_instances={len(final_values[field])} "
                f"values={final_values[field]!r}"
            )
            for occurrence in grounded:
                print(
                    f"  [{occurrence.get('occurrence_index')}] "
                    f"status={occurrence.get('status')} "
                    f"score={float(occurrence.get('evidence_score') or 0.0):.3f} "
                    f"bbox={occurrence.get('final_bbox')} "
                    f"evidence={occurrence.get('evidence_text')!r}"
                )

        field_scores: Dict[str, Dict[str, Any]] = {}
        doc_tp = 0
        doc_fp = 0
        doc_fn = 0
        exact_fields = 0

        for field in EVAL_FIELDS:
            gt_list = gt_values.get(field, [])
            pred_list = final_values.get(field, [])
            normalized_gt = normalize_instance_values(field, gt_list)
            normalized_pred = normalize_instance_values(field, pred_list)
            gt_metric_text = instances_as_metric_text(normalized_gt)
            pred_metric_text = instances_as_metric_text(normalized_pred)
            tp, fp, fn = token_multiset_counts(gt_metric_text, pred_metric_text)
            scores = prf(tp, fp, fn)
            exact = normalized_gt == normalized_pred
            if exact:
                exact_fields += 1
            doc_tp += tp
            doc_fp += fp
            doc_fn += fn
            field_scores[field] = {
                "gt": gt_list,
                "gt_instances": gt_instances.get(field, []),
                "vlm_pred": vlm_field_values.get(field, []),
                "pre_crop_pred": occurrence_values(
                    grounded_fields[field], key="pre_crop_final_value"
                ),
                "pred": pred_list,
                "normalized_gt": normalized_gt,
                "normalized_pred": normalized_pred,
                "exact_match": exact,
                "field_present": bool(normalized_gt),
                "gt_instance_count": len(normalized_gt),
                "pred_instance_count": len(normalized_pred),
                "token_counts": {"tp": tp, "fp": fp, "fn": fn},
                "token_metrics": scores,
                "grounding_status": [item.get("status") for item in grounded_fields[field]],
                "grounding_bbox": [item.get("final_bbox") for item in grounded_fields[field]],
                "evidence_text": [item.get("evidence_text") for item in grounded_fields[field]],
                "evidence_score": [item.get("evidence_score") for item in grounded_fields[field]],
                "crop_verification": [
                    item.get("crop_verification") for item in grounded_fields[field]
                ],
            }

        doc_scores = prf(doc_tp, doc_fp, doc_fn)
        exact_doc = exact_fields == len(EVAL_FIELDS)

        result = {
            "image": image_name,
            "image_path": image_path,
            "ocr_path": ocr_path,
            "ocr_source": ARGS.ocr_source,
            "annotation_file": record.get("_annotation_file"),
            "ground_truth": gt_values,
            "ground_truth_instances": gt_instances,
            "raw_output": raw_output,
            "cleaned_output": cleaned_output,
            "raw_prediction": pred_json,
            "ocr_native_text": ocr_native_text,
            "ocr_native_semantic_draft": ocr_native_draft,
            "native_ocr_alignment": native_alignment,
            "prediction_provenance": (
                "ocr_native_text_aligned_to_label_free_geometry_then_heuristically_classified"
                if use_ocr_native_adapter()
                else "structured_vlm_json"
            ),
            "vlm_field_instances": vlm_field_instances,
            "vlm_field_values": vlm_field_values,
            "vlm_field_bboxes_original_coords": vlm_field_bboxes,
            "grounded_prediction": final_values,
            "grounded_fields": grounded_fields,
            "json_parse_success": parse_success,
            "field_scores": field_scores,
            "doc_token_counts": {"tp": doc_tp, "fp": doc_fp, "fn": doc_fn},
            "doc_token_metrics": doc_scores,
            "exact_field_count": exact_fields,
            "exact_document_match": exact_doc,
            "image_sizes": {"original": original_size, "sent": sent_size, "image_target": image_target_max},
            "usage": response_json.get("usage", {}),
        }
        all_results.append(result)

        with open(ARGS.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if ARGS.sleep > 0:
            time.sleep(ARGS.sleep)

    write_csv_outputs(all_results)
    print_base_summary(all_results)
    print_summary(all_results)


def write_csv_outputs(all_results: List[Dict[str, Any]]) -> None:
    with open(ARGS.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "image",
                "field",
                "json_parse_success",
                "exact_match",
                "ground_truth",
                "vlm_prediction",
                "grounded_prediction",
                "token_tp",
                "token_fp",
                "token_fn",
                "token_precision",
                "token_recall",
                "token_f1",
                "grounding_status",
                "evidence_score",
                "evidence_text",
                "grounding_bbox",
                "gt_instance_count",
                "pred_instance_count",
                "field_present",
            ]
        )
        for result in all_results:
            for field, vals in result["field_scores"].items():
                writer.writerow(
                    [
                        result["image"],
                        field,
                        result["json_parse_success"],
                        vals["exact_match"],
                        vals["gt"],
                        vals["vlm_pred"],
                        vals["pred"],
                        vals["token_counts"]["tp"],
                        vals["token_counts"]["fp"],
                        vals["token_counts"]["fn"],
                        round(vals["token_metrics"]["precision"] * 100, 2),
                        round(vals["token_metrics"]["recall"] * 100, 2),
                        round(vals["token_metrics"]["f1"] * 100, 2),
                        json.dumps(vals.get("grounding_status"), ensure_ascii=False),
                        json.dumps(vals.get("evidence_score"), ensure_ascii=False),
                        json.dumps(vals.get("evidence_text"), ensure_ascii=False),
                        json.dumps(vals.get("grounding_bbox"), ensure_ascii=False),
                        vals.get("gt_instance_count"),
                        vals.get("pred_instance_count"),
                        vals.get("field_present"),
                    ]
                )

    with open(ARGS.grounding_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "image",
                "field",
                "vlm_value",
                "pre_crop_final_value",
                "final_value",
                "status",
                "source",
                "evidence_score",
                "evidence_text",
                "final_bbox",
                "vlm_bbox",
                "vlm_bbox_evidence_score",
                "crop_verify_attempted",
                "crop_supports_field",
                "crop_value",
            ]
        )
        for result in all_results:
            for field, occurrences in result.get("grounded_fields", {}).items():
                for vals in occurrences:
                    crop = vals.get("crop_verification") or {}
                    writer.writerow(
                        [
                            result["image"],
                            field,
                            vals.get("vlm_value"),
                            vals.get("pre_crop_final_value"),
                            vals.get("final_value"),
                            vals.get("status"),
                            vals.get("source"),
                            round(float(vals.get("evidence_score") or 0.0), 4),
                            vals.get("evidence_text"),
                            json.dumps(vals.get("final_bbox"), ensure_ascii=False),
                            json.dumps(vals.get("vlm_bbox"), ensure_ascii=False),
                            round(float(vals.get("vlm_bbox_evidence_score") or 0.0), 4),
                            crop.get("attempted"),
                            crop.get("supports_field"),
                            crop.get("value"),
                        ]
                    )


def print_base_summary(all_results: List[Dict[str, Any]]) -> None:
    docs_processed = len(all_results)
    parse_ok = sum(1 for result in all_results if result["json_parse_success"])
    total_fields = docs_processed * len(EVAL_FIELDS)

    doc_metrics: List[Dict[str, float]] = []
    doc_counts: List[Dict[str, int]] = []
    exact_docs = 0
    exact_fields = 0
    field_totals: Dict[str, Dict[str, int]] = {
        field: {
            "exact": 0,
            "total": 0,
            "present_exact": 0,
            "present_total": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for field in EVAL_FIELDS
    }

    for result in all_results:
        gt = result.get("ground_truth") or {}
        base_values = result.get("vlm_field_values") or {}
        doc_tp = 0
        doc_fp = 0
        doc_fn = 0
        doc_exact_fields = 0

        for field in EVAL_FIELDS:
            gt_values = gt.get(field, [])
            pred_values = base_values.get(field, [])
            if not isinstance(gt_values, list):
                gt_values = [gt_values]
            if not isinstance(pred_values, list):
                pred_values = [pred_values]
            normalized_gt = normalize_instance_values(field, gt_values)
            normalized_pred = normalize_instance_values(field, pred_values)
            tp, fp, fn = token_multiset_counts(
                instances_as_metric_text(normalized_gt),
                instances_as_metric_text(normalized_pred),
            )
            exact = normalized_gt == normalized_pred

            doc_tp += tp
            doc_fp += fp
            doc_fn += fn
            if exact:
                doc_exact_fields += 1
                exact_fields += 1

            field_totals[field]["total"] += 1
            field_totals[field]["exact"] += 1 if exact else 0
            if normalized_gt:
                field_totals[field]["present_total"] += 1
                field_totals[field]["present_exact"] += 1 if exact else 0
            field_totals[field]["tp"] += tp
            field_totals[field]["fp"] += fp
            field_totals[field]["fn"] += fn

        if doc_exact_fields == len(EVAL_FIELDS):
            exact_docs += 1
        doc_counts.append({"tp": doc_tp, "fp": doc_fp, "fn": doc_fn})
        doc_metrics.append(prf(doc_tp, doc_fp, doc_fn))

    macro_precision = (
        sum(metric["precision"] for metric in doc_metrics) / docs_processed if docs_processed else 0.0
    )
    macro_recall = sum(metric["recall"] for metric in doc_metrics) / docs_processed if docs_processed else 0.0
    macro_f1 = sum(metric["f1"] for metric in doc_metrics) / docs_processed if docs_processed else 0.0

    micro_tp = sum(counts["tp"] for counts in doc_counts)
    micro_fp = sum(counts["fp"] for counts in doc_counts)
    micro_fn = sum(counts["fn"] for counts in doc_counts)
    micro_scores = prf(micro_tp, micro_fp, micro_fn)

    print("\n====================================================")
    print(f"FINAL {ARGS.model_label} WILDRECEIPT BASE SUMMARY")
    print("====================================================\n")
    print(f"Documents processed : {docs_processed}")
    print(f"JSON parse OK       : {parse_ok}/{docs_processed}")
    print("\nMacro token metrics over documents:")
    print(f"  Precision: {round(macro_precision * 100, 2):.2f}")
    print(f"  Recall   : {round(macro_recall * 100, 2):.2f}")
    print(f"  F1       : {round(macro_f1 * 100, 2):.2f}")
    print("\nMicro token metrics over all evaluated field tokens:")
    print(f"  Precision: {round(micro_scores['precision'] * 100, 2):.2f}")
    print(f"  Recall   : {round(micro_scores['recall'] * 100, 2):.2f}")
    print(f"  F1       : {round(micro_scores['f1'] * 100, 2):.2f}")
    print(f"  TP/FP/FN : {micro_tp}/{micro_fp}/{micro_fn}")
    print("\nExact full-document evaluated-field match:")
    print(f"  Exact    : {exact_docs}/{docs_processed} ({safe_percent(exact_docs, docs_processed):.2f}%)")
    print("\nExact field match:")
    print(f"  Exact    : {exact_fields}/{total_fields} ({safe_percent(exact_fields, total_fields):.2f}%)")
    print("\nField-level token metrics:")
    for field, stats in field_totals.items():
        scores = prf(stats["tp"], stats["fp"], stats["fn"])
        print(
            f"  {field}: Exact={stats['exact']}/{stats['total']} "
            f"({safe_percent(stats['exact'], stats['total']):.2f}%) | "
            f"PresentExact={stats['present_exact']}/{stats['present_total']} "
            f"({safe_percent(stats['present_exact'], stats['present_total']):.2f}%) | "
            f"P={round(scores['precision'] * 100, 2):.2f} "
            f"R={round(scores['recall'] * 100, 2):.2f} "
            f"F1={round(scores['f1'] * 100, 2):.2f} | "
            f"TP/FP/FN={stats['tp']}/{stats['fp']}/{stats['fn']}"
        )
    print("====================================================\n")


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    docs_processed = len(all_results)
    parse_ok = sum(1 for result in all_results if result["json_parse_success"])
    exact_docs = sum(1 for result in all_results if result["exact_document_match"])
    total_fields = docs_processed * len(EVAL_FIELDS)
    exact_fields = sum(result["exact_field_count"] for result in all_results)

    macro_precision = (
        sum(result["doc_token_metrics"]["precision"] for result in all_results) / docs_processed if docs_processed else 0.0
    )
    macro_recall = (
        sum(result["doc_token_metrics"]["recall"] for result in all_results) / docs_processed if docs_processed else 0.0
    )
    macro_f1 = sum(result["doc_token_metrics"]["f1"] for result in all_results) / docs_processed if docs_processed else 0.0

    micro_tp = sum(result["doc_token_counts"]["tp"] for result in all_results)
    micro_fp = sum(result["doc_token_counts"]["fp"] for result in all_results)
    micro_fn = sum(result["doc_token_counts"]["fn"] for result in all_results)
    micro_scores = prf(micro_tp, micro_fp, micro_fn)

    field_totals: Dict[str, Dict[str, int]] = {
        field: {
            "exact": 0,
            "total": 0,
            "present_exact": 0,
            "present_total": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for field in EVAL_FIELDS
    }
    status_counts: Counter = Counter()
    for result in all_results:
        for field, vals in result["field_scores"].items():
            field_totals[field]["total"] += 1
            field_totals[field]["exact"] += 1 if vals["exact_match"] else 0
            if vals.get("field_present"):
                field_totals[field]["present_total"] += 1
                field_totals[field]["present_exact"] += 1 if vals["exact_match"] else 0
            field_totals[field]["tp"] += vals["token_counts"]["tp"]
            field_totals[field]["fp"] += vals["token_counts"]["fp"]
            field_totals[field]["fn"] += vals["token_counts"]["fn"]
            for status in vals.get("grounding_status") or []:
                status_counts[str(status)] += 1

    print("\n====================================================")
    print(f"FINAL {ARGS.model_label} WILDRECEIPT GROUNDED SUMMARY")
    print("====================================================\n")
    print(f"Documents processed : {docs_processed}")
    print(f"JSON parse OK       : {parse_ok}/{docs_processed}")
    print("\nMacro token metrics over documents:")
    print(f"  Precision: {round(macro_precision * 100, 2):.2f}")
    print(f"  Recall   : {round(macro_recall * 100, 2):.2f}")
    print(f"  F1       : {round(macro_f1 * 100, 2):.2f}")
    print("\nMicro token metrics over all evaluated field tokens:")
    print(f"  Precision: {round(micro_scores['precision'] * 100, 2):.2f}")
    print(f"  Recall   : {round(micro_scores['recall'] * 100, 2):.2f}")
    print(f"  F1       : {round(micro_scores['f1'] * 100, 2):.2f}")
    print(f"  TP/FP/FN : {micro_tp}/{micro_fp}/{micro_fn}")
    print("\nExact full-document evaluated-field match:")
    print(f"  Exact    : {exact_docs}/{docs_processed} ({safe_percent(exact_docs, docs_processed):.2f}%)")
    print("\nExact field match:")
    print(f"  Exact    : {exact_fields}/{total_fields} ({safe_percent(exact_fields, total_fields):.2f}%)")
    print("\nField-level token metrics:")
    for field, stats in field_totals.items():
        scores = prf(stats["tp"], stats["fp"], stats["fn"])
        print(
            f"  {field}: Exact={stats['exact']}/{stats['total']} "
            f"({safe_percent(stats['exact'], stats['total']):.2f}%) | "
            f"PresentExact={stats['present_exact']}/{stats['present_total']} "
            f"({safe_percent(stats['present_exact'], stats['present_total']):.2f}%) | "
            f"P={round(scores['precision'] * 100, 2):.2f} "
            f"R={round(scores['recall'] * 100, 2):.2f} "
            f"F1={round(scores['f1'] * 100, 2):.2f} | "
            f"TP/FP/FN={stats['tp']}/{stats['fp']}/{stats['fn']}"
        )
    print("\nGrounding status counts:")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")
    if input_token_lengths:
        print("\nToken statistics:")
        print(
            f"  Prompt tokens     : min={min(input_token_lengths)} "
            f"max={max(input_token_lengths)} avg={round(sum(input_token_lengths) / len(input_token_lengths), 2)}"
        )
        print(
            f"  Completion tokens : min={min(output_token_lengths)} "
            f"max={max(output_token_lengths)} avg={round(sum(output_token_lengths) / len(output_token_lengths), 2)}"
        )
    print("====================================================")
    print(f"JSON results saved to      : {ARGS.output_json}")
    print(f"Evaluation CSV saved to    : {ARGS.output_csv}")
    print(f"Grounding CSV saved to     : {ARGS.grounding_csv}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
