#!/usr/bin/env python3
"""
Train/evaluate a Qwen-VL LoRA model for Hindi SROIE-style extraction.

Goal
----
Input  : Hindi rendered receipt image created by layout_aware_invoice_localizer_*.py
Target : Hindi analogue of SROIE JSON, e.g.
         {"company":"...", "date":"...", "address":"...", "total":"..."}

The Hindi target JSON is constructed from:
  1) original SROIE entity JSON files, and
  2) per-document metadata JSON produced by the Hindi renderer, where each OCR span has
     original_text -> converted_text.

Subcommands
-----------
prepare : build train/val/test JSONL files from image dir + metadata dir + SROIE entity dir
train   : LoRA fine-tune Qwen2-VL/Qwen2.5-VL-style models using Transformers + PEFT
infer   : run base or LoRA model on a prepared JSONL split
score   : compute exact-match and token-level scores from prediction JSONL
run_all : base eval -> train -> LoRA eval -> score both

Expected packages for training/eval on GPU:
  pip install torch transformers peft accelerate qwen-vl-utils pillow tqdm
  Optional: pip install bitsandbytes   # for --load-4bit on Linux CUDA

Example
-------
python train_hindi_sroie_qwenvl_lora.py prepare \
  --images ./v7_hi_indicxlit/images \
  --metadata ./v7_hi_indicxlit/metadata \
  --entities './SROIE dataset/entities' \
  --out ./hindi_sroie_qwenvl_data \
  --train-ratio 0.8 --val-ratio 0.1 --seed 42

python train_hindi_sroie_qwenvl_lora.py train \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train-jsonl ./hindi_sroie_qwenvl_data/train.jsonl \
  --val-jsonl ./hindi_sroie_qwenvl_data/val.jsonl \
  --output-dir ./qwen25vl_hindi_sroie_lora \
  --load-4bit --epochs 3 --batch-size 1 --grad-accum 8

python train_hindi_sroie_qwenvl_lora.py infer \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter ./qwen25vl_hindi_sroie_lora \
  --data-jsonl ./hindi_sroie_qwenvl_data/test.jsonl \
  --pred-jsonl ./pred_lora.jsonl

python train_hindi_sroie_qwenvl_lora.py score \
  --pred-jsonl ./pred_lora.jsonl
"""

from __future__ import annotations


import os

os.environ["HF_HOME"] = "/workspace/"

os.environ["HF_HUB_CACHE"] = "/workspace/hub/"

os.environ["HF_XET_CACHE"] = "/workspace/xet/"

os.environ["TRANSFORMERS_CACHE"] = "/workspace/transformers/"

os.environ["HF_DATASETS_CACHE"] = "/workspace/datasets/"

import argparse
import dataclasses
import difflib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

FIELD_ORDER = ["company", "date", "address", "total"]
DEFAULT_PROMPT = (
    "You are given a Hindi receipt image. Extract the receipt information and return ONLY a valid JSON object "
    "with exactly these keys: company, date, address, total. "
    "Use the text as it appears in the Hindi image. Keep dates, numbers, currency amounts, and IDs unchanged. "
    "Do not add explanations."
)

# -----------------------------
# Small text utilities
# -----------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def norm_match(s: str) -> str:
    s = str(s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return norm_space(s)


def simple_tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(s or "").upper())


def looks_numeric_or_date(s: str) -> bool:
    s = str(s or "").strip()
    return bool(re.fullmatch(r"[\s\d.,:/\-RM$€£]+", s))


def clean_json_value(s: str) -> str:
    s = norm_space(s)
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    s = re.sub(r"([(])\s+", r"\1", s)
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(a: str, b: str) -> float:
    ta = re.findall(r"\w+", str(a).lower())
    tb = re.findall(r"\w+", str(b).lower())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(tb)
    r = overlap / len(ta)
    return 2 * p * r / (p + r)


def char_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()


def canonical_for_exact(s: str) -> str:
    s = norm_space(str(s or ""))
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

# -----------------------------
# Metadata -> Hindi target mapping
# -----------------------------

@dataclasses.dataclass
class Span:
    original: str
    converted: str
    x1: float
    y1: float
    x2: float
    y2: float
    action: str = ""
    reason: str = ""

    @property
    def nm(self) -> str:
        return norm_match(self.original)


def load_spans(metadata_path: Path) -> List[Span]:
    data = read_json(metadata_path)
    spans: List[Span] = []
    for r in data:
        rect = r.get("bbox_rect") or [0, 0, 0, 0]
        if len(rect) < 4:
            rect = [0, 0, 0, 0]
        spans.append(
            Span(
                original=str(r.get("original_text", "")),
                converted=str(r.get("converted_text", "")),
                x1=float(rect[0]), y1=float(rect[1]), x2=float(rect[2]), y2=float(rect[3]),
                action=str(r.get("action", "")), reason=str(r.get("reason", "")),
            )
        )
    spans.sort(key=lambda s: (s.y1, s.x1))
    return spans


def group_spans_into_lines(spans: List[Span], y_tol: float = 12.0) -> List[List[Span]]:
    lines: List[List[Span]] = []
    for sp in sorted(spans, key=lambda s: (s.y1, s.x1)):
        placed = False
        cy = (sp.y1 + sp.y2) / 2
        for line in lines:
            ly = sum((x.y1 + x.y2) / 2 for x in line) / max(1, len(line))
            if abs(cy - ly) <= y_tol:
                line.append(sp)
                placed = True
                break
        if not placed:
            lines.append([sp])
    for line in lines:
        line.sort(key=lambda s: s.x1)
    lines.sort(key=lambda line: (sum(s.y1 for s in line) / max(1, len(line)), line[0].x1 if line else 0))
    return lines


def line_text(line: List[Span], attr: str = "original") -> str:
    return clean_json_value(" ".join(getattr(s, attr) for s in line if getattr(s, attr)))


def best_line_matches(value: str, lines: List[List[Span]], min_ratio: float = 0.72) -> List[List[Span]]:
    """Return line(s) that likely correspond to an entity value.

    For company/date/total this is often a single line. For address this may be
    multiple lines; the caller can ask field-specific logic.
    """
    nv = norm_match(value)
    if not nv:
        return []
    scored: List[Tuple[float, List[Span]]] = []
    for line in lines:
        lo = line_text(line, "original")
        nl = norm_match(lo)
        if not nl:
            continue
        ratio = difflib.SequenceMatcher(None, nv, nl).ratio()
        # Token overlap bonus handles punctuation/OCR splits.
        vt, lt = set(simple_tokens(value)), set(simple_tokens(lo))
        overlap = len(vt & lt) / max(1, len(vt))
        score = max(ratio, overlap)
        if nv in nl or nl in nv:
            score = max(score, 0.95)
        if score >= min_ratio:
            scored.append((score, line))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]


def greedy_span_map(value: str, spans: List[Span]) -> str:
    """Map an English entity value to Hindi by selecting metadata spans.

    This is conservative: it first tries exact/substring line mapping, then token coverage.
    Dates/totals are usually copied unchanged.
    """
    value = norm_space(value)
    if not value:
        return ""
    if looks_numeric_or_date(value):
        return value

    nv = norm_match(value)
    if not nv:
        return value

    # Exact full-span match.
    for sp in spans:
        if sp.nm == nv and sp.converted:
            return clean_json_value(sp.converted)

    lines = group_spans_into_lines(spans)

    # Exact/near line match.
    matches = best_line_matches(value, lines, min_ratio=0.88)
    if matches:
        return clean_json_value(line_text(matches[0], "converted"))

    # Greedy token coverage at span level.
    target_toks = simple_tokens(value)
    remaining = set(target_toks)
    chosen: List[Span] = []
    for sp in spans:
        toks = simple_tokens(sp.original)
        if not toks:
            continue
        inter = set(toks) & remaining
        if inter:
            # avoid selecting huge unrelated product lines for small targets
            coverage = len(inter) / max(1, len(set(toks)))
            if coverage >= 0.35 or len(inter) >= 2:
                chosen.append(sp)
                remaining -= inter
        if not remaining:
            break
    if chosen and (1 - len(remaining) / max(1, len(set(target_toks)))) >= 0.55:
        chosen.sort(key=lambda s: (s.y1, s.x1))
        return clean_json_value(" ".join(s.converted for s in chosen if s.converted))

    # Last resort: return English value. This is better than hallucinating a wrong Hindi entity.
    return value


def map_address(value: str, spans: List[Span]) -> str:
    value = norm_space(value)
    if not value:
        return ""
    lines = group_spans_into_lines(spans)
    vtoks = set(simple_tokens(value))
    selected: List[List[Span]] = []
    covered: set[str] = set()
    for line in lines:
        lo = line_text(line, "original")
        ltoks = set(simple_tokens(lo))
        if not ltoks:
            continue
        overlap = vtoks & ltoks
        # Address lines are often exact substrings of the SROIE address value.
        if overlap and (len(overlap) >= 2 or len(overlap) / max(1, len(ltoks)) >= 0.45):
            selected.append(line)
            covered |= overlap
    if selected and len(covered) / max(1, len(vtoks)) >= 0.45:
        return clean_json_value(" ".join(line_text(line, "converted") for line in selected))
    return greedy_span_map(value, spans)


def hindi_target_from_english(eng: Dict[str, Any], spans: List[Span], fields: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in fields:
        v = str(eng.get(k, "") or "").strip()
        if k in {"date", "total"}:
            out[k] = clean_json_value(v)  # preserve exact SROIE date/amount target
        elif k == "address":
            out[k] = map_address(v, spans)
        else:
            out[k] = greedy_span_map(v, spans)
    return out


def find_entity_file(entity_dir: Path, doc_id: str) -> Optional[Path]:
    candidates = [
        entity_dir / f"{doc_id}.json",
        entity_dir / f"{doc_id}.txt",
        entity_dir / f"{doc_id}.key",
    ]
    for p in candidates:
        if p.exists() and not p.name.startswith("._"):
            return p
    # fallback glob
    for p in entity_dir.glob(f"{doc_id}.*"):
        if p.suffix.lower() in {".json", ".txt", ".key"} and not p.name.startswith("._"):
            return p
    return None


def read_entity_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        obj = json.loads(text)
        return {str(k).lower(): str(v) for k, v in obj.items()}
    except Exception:
        # SROIE entity files are normally JSON, but keep a line fallback.
        out: Dict[str, str] = {}
        for line in text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip().lower()] = v.strip()
        return out


def locate_image(image_dir: Path, doc_id: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = image_dir / f"{doc_id}{ext}"
        if p.exists() and not p.name.startswith("._"):
            return p
    matches = [p for p in image_dir.glob(f"{doc_id}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    return matches[0] if matches else None


def build_record(doc_id: str, image_path: Path, target: Dict[str, str], prompt: str) -> Dict[str, Any]:
    return {
        "id": doc_id,
        "image": str(image_path),
        "prompt": prompt,
        "target": {k: target.get(k, "") for k in FIELD_ORDER},
        "target_json": json.dumps({k: target.get(k, "") for k in FIELD_ORDER}, ensure_ascii=False, separators=(",", ":")),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps({k: target.get(k, "") for k in FIELD_ORDER}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }

# -----------------------------
# Prepare command
# -----------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    image_dir = Path(args.images)
    metadata_dir = Path(args.metadata)
    entity_dir = Path(args.entities)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted([p for p in metadata_dir.glob("*.json") if not p.name.startswith("._")])
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for mp in tqdm(meta_files, desc="building Hindi SROIE records"):
        doc_id = mp.stem
        img = locate_image(image_dir, doc_id)
        ent = find_entity_file(entity_dir, doc_id)
        if img is None:
            skipped.append({"id": doc_id, "reason": "missing_image"})
            continue
        if ent is None:
            skipped.append({"id": doc_id, "reason": "missing_entity_json"})
            continue
        try:
            eng = read_entity_file(ent)
            spans = load_spans(mp)
            target = hindi_target_from_english(eng, spans, FIELD_ORDER)
            records.append(build_record(doc_id, img.resolve(), target, args.prompt))
        except Exception as e:
            skipped.append({"id": doc_id, "reason": f"error:{type(e).__name__}:{e}"})

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n = len(records)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    def dump_jsonl(path: Path, rows: List[Dict[str, Any]]):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(out_dir / "train.jsonl", train)
    dump_jsonl(out_dir / "val.jsonl", val)
    dump_jsonl(out_dir / "test.jsonl", test)
    write_json(out_dir / "skipped.json", skipped)
    write_json(out_dir / "summary.json", {
        "num_records": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "skipped": len(skipped),
        "fields": FIELD_ORDER,
        "prompt": args.prompt,
    })

    print(f"Wrote {n} records to {out_dir}")
    print(f"train={len(train)} val={len(val)} test={len(test)} skipped={len(skipped)}")
    if skipped:
        print(f"Skipped details: {out_dir / 'skipped.json'}")

# -----------------------------
# Model loading / Qwen-VL helpers
# -----------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_qwen_tools():
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception as e:
        raise RuntimeError("Please install qwen-vl-utils: pip install qwen-vl-utils") from e
    return process_vision_info


def get_device_dtype(args: argparse.Namespace):
    import torch
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    if args.dtype == "fp32":
        return torch.float32
    # auto
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model_and_processor(model_name: str, args: argparse.Namespace, for_train: bool = False):
    import torch
    from transformers import AutoProcessor
    dtype = get_device_dtype(args)

    quantization_config = None
    if getattr(args, "load_4bit", False):
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            raise RuntimeError("--load-4bit requires bitsandbytes and a CUDA/Linux-compatible setup") from e

    # AutoModel is usually enough for Qwen2-VL/Qwen2.5-VL with recent transformers.
    from transformers import AutoModelForVision2Seq
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=args.device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            cache_dir="/workspace/"
        )
    except Exception:
        # Fallback for older transformers/model classes.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=args.device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                cache_dir="/workspace/"
            )
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=args.device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                cache_dir="/workspace/"
            )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir="/workspace/")
    if for_train:
        model.config.use_cache = False
    return model, processor


def record_to_messages(record: Dict[str, Any], include_answer: bool) -> List[Dict[str, Any]]:
    user = {
        "role": "user",
        "content": [
            # {"type": "image", "image": record["image"]},
            {"type": "image", "image": os.path.join("/workspace/ocr_benchmark/v7_hi_indicxlit/images/",record["image"].split("/")[-1])},
            {"type": "text", "text": record.get("prompt", DEFAULT_PROMPT)},
        ],
    }
    if not include_answer:
        return [user]
    return [user, {"role": "assistant", "content": record["target_json"]}]


def safe_json_parse(text: str) -> Dict[str, str]:
    text = str(text or "").strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except Exception:
        # Try minimal repairs: smart quotes/trailing commas are common.
        repaired = text.replace("“", '"').replace("”", '"').replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            obj = json.loads(repaired)
        except Exception:
            return {k: "" for k in FIELD_ORDER}
    return {k: clean_json_value(str(obj.get(k, "") or "")) for k in FIELD_ORDER}

# -----------------------------
# Train command
# -----------------------------

class SROIEDataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx]


def make_collator(processor, max_length: int):
    process_vision_info = import_qwen_tools()

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_messages = [record_to_messages(r, include_answer=True) for r in batch]
        prompt_messages = [record_to_messages(r, include_answer=False) for r in batch]

        full_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_messages]
        prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_messages]

        image_inputs, video_inputs = process_vision_info(full_messages)
        inputs = processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100

        # Mask prompt tokens. This is approximate but works well for Qwen-VL chat SFT.
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)
        prompt_inputs = processor(
            text=prompt_texts,
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        for i in range(labels.shape[0]):
            plen = int(prompt_inputs["attention_mask"][i].sum().item())
            labels[i, :min(plen, labels.shape[1])] = -100

        inputs["labels"] = labels
        return inputs

    return collate


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl) if args.val_jsonl else []
    model, processor = load_model_and_processor(args.model, args, for_train=True)

    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.freeze_vision:
        for n, p in model.named_parameters():
            if any(key in n.lower() for key in ["vision", "visual", "vit"]):
                p.requires_grad = False

    targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SROIEDataset(train_rows)
    val_ds = SROIEDataset(val_rows) if val_rows else None
    collator = make_collator(processor, args.max_length)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_rows else None,
        eval_strategy="steps" if val_rows else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=(args.dtype == "bf16" or (args.dtype == "auto" and torch.cuda.is_available())),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and processor to {args.output_dir}")

# -----------------------------
# Inference command
# -----------------------------

def cmd_infer(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel

    rows = load_jsonl(args.data_jsonl)
    model, processor = load_model_and_processor(args.model, args, for_train=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    process_vision_info = import_qwen_tools()
    out_path = Path(args.pred_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(rows, desc="inference"):
            messages = record_to_messages(r, include_answer=False)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # Move tensors to model device when device_map is not doing it automatically.
            first_device = next(model.parameters()).device
            inputs = {k: (v.to(first_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            input_len = inputs["input_ids"].shape[1]
            new_tokens = gen[:, input_len:]
            pred_text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            pred_json = safe_json_parse(pred_text)
            rec = {
                "id": r.get("id"),
                "image": r.get("image"),
                "gold": r.get("target", {}),
                "gold_json": r.get("target_json", ""),
                "prediction_text": pred_text,
                "prediction": pred_json,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote predictions to {out_path}")

# -----------------------------
# Score command
# -----------------------------

def compute_scores(pred_rows: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    n = len(pred_rows)
    field_exact = {k: 0 for k in fields}
    field_f1 = {k: 0.0 for k in fields}
    field_char = {k: 0.0 for k in fields}
    doc_exact = 0

    examples = []
    for r in pred_rows:
        gold = r.get("gold", {}) or {}
        pred = r.get("prediction", {}) or safe_json_parse(r.get("prediction_text", ""))
        one_doc_exact = True
        for k in fields:
            g = clean_json_value(str(gold.get(k, "") or ""))
            p = clean_json_value(str(pred.get(k, "") or ""))
            ex = canonical_for_exact(g) == canonical_for_exact(p)
            field_exact[k] += int(ex)
            field_f1[k] += token_f1(g, p)
            field_char[k] += char_similarity(g, p)
            one_doc_exact = one_doc_exact and ex
        doc_exact += int(one_doc_exact)
        if len(examples) < 20 and not one_doc_exact:
            examples.append({"id": r.get("id"), "gold": gold, "prediction": pred, "raw": r.get("prediction_text", "")})

    denom = max(1, n)
    per_field = {
        k: {
            "exact": field_exact[k] / denom,
            "token_f1": field_f1[k] / denom,
            "char_similarity": field_char[k] / denom,
        }
        for k in fields
    }
    return {
        "num_examples": n,
        "document_exact_match": doc_exact / denom,
        "macro_field_exact_match": sum(per_field[k]["exact"] for k in fields) / max(1, len(fields)),
        "macro_token_f1": sum(per_field[k]["token_f1"] for k in fields) / max(1, len(fields)),
        "macro_char_similarity": sum(per_field[k]["char_similarity"] for k in fields) / max(1, len(fields)),
        "per_field": per_field,
        "bad_examples_sample": examples,
    }


def cmd_score(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.pred_jsonl)
    scores = compute_scores(rows, FIELD_ORDER)
    if args.out_json:
        write_json(Path(args.out_json), scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))

# -----------------------------
# Run-all command
# -----------------------------

def cmd_run_all(args: argparse.Namespace) -> None:
    base_pred = Path(args.work_dir) / "pred_base.jsonl"
    lora_pred = Path(args.work_dir) / "pred_lora.jsonl"
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    infer_base_args = argparse.Namespace(**vars(args))
    infer_base_args.adapter = None
    infer_base_args.data_jsonl = args.test_jsonl
    infer_base_args.pred_jsonl = str(base_pred)
    cmd_infer(infer_base_args)
    base_scores = compute_scores(load_jsonl(base_pred), FIELD_ORDER)
    write_json(Path(args.work_dir) / "scores_base.json", base_scores)

    train_args = argparse.Namespace(**vars(args))
    cmd_train(train_args)

    infer_lora_args = argparse.Namespace(**vars(args))
    infer_lora_args.adapter = args.output_dir
    infer_lora_args.data_jsonl = args.test_jsonl
    infer_lora_args.pred_jsonl = str(lora_pred)
    cmd_infer(infer_lora_args)
    lora_scores = compute_scores(load_jsonl(lora_pred), FIELD_ORDER)
    write_json(Path(args.work_dir) / "scores_lora.json", lora_scores)

    print("BASE:")
    print(json.dumps(base_scores, ensure_ascii=False, indent=2))
    print("LORA:")
    print(json.dumps(lora_scores, ensure_ascii=False, indent=2))

# -----------------------------
# CLI
# -----------------------------

def add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--load-4bit", action="store_true")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hindi SROIE Qwen-VL LoRA training/evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--images", required=True, help="Hindi rendered image directory")
    p.add_argument("--metadata", required=True, help="Metadata JSON directory from Hindi renderer")
    p.add_argument("--entities", required=True, help="Original SROIE entity/key JSON directory")
    p.add_argument("--out", required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train")
    add_common_model_args(p)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--freeze-vision", action="store_true", default=True)
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--report-to", default="none")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer")
    add_common_model_args(p)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--pred-jsonl", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("score")
    p.add_argument("--pred-jsonl", required=True)
    p.add_argument("--out-json", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("run_all")
    add_common_model_args(p)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--freeze-vision", action="store_true", default=True)
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--report-to", default="none")
    p.set_defaults(func=cmd_run_all)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
