"""L3 — Semantic similarity layer.

Compares prompt embeddings against a database of known attack embeddings
using cosine similarity via FAISS. Detects paraphrased or novel variations
of known attacks that regex cannot catch.

Requires: onnxruntime, tokenizers, faiss-cpu
Falls back to sentence-transformers if ONNX model not available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from prompt_armor.config import ShieldConfig
from prompt_armor.layers.base import BaseLayer
from prompt_armor.models import CATEGORY_MAP, Category, Evidence, LayerResult

logger = logging.getLogger("prompt_armor")

_CATEGORY_MAP: dict[str, Category | None] = {**CATEGORY_MAP, "benign": None}

_DEFAULT_ATTACKS_PATH = Path(__file__).parent.parent / "data" / "attacks" / "known_attacks.jsonl"
_ONNX_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "l3-contrastive-onnx"
_CONTRASTIVE_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "l3-contrastive"
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# On-disk FAISS + metadata cache (skip re-encoding ~25k attacks on every process start)
_CACHE_VERSION = 1
_CACHE_DISABLE_ENV = "PROMPT_ARMOR_DISABLE_L3_INDEX_CACHE"
_CACHE_DIR_ENV = "PROMPT_ARMOR_L3_CACHE_DIR"
_CACHE_DIR_FALLBACK = Path(__file__).resolve().parent.parent / "data" / "models" / "l3-faiss-cache"

# Similarity thresholds
_HIGH_SIMILARITY = 0.75
_MEDIUM_SIMILARITY = 0.55


def _l3_cache_base_dir() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _CACHE_DIR_FALLBACK


def _file_sig(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    st = path.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _attacks_file_sig(path: Path) -> dict[str, int | str]:
    p = path.resolve()
    st = p.stat()
    return {"path": str(p), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _build_model_fingerprint(use_onnx: bool) -> dict[str, Any]:
    if use_onnx:
        m = _ONNX_MODEL_PATH / "model_quant.onnx"
        t = _ONNX_MODEL_PATH / "tokenizer.json"
        ms, ts = _file_sig(m), _file_sig(t)
        return {"kind": "onnx", "model": ms, "tokenizer": ts}
    p = _CONTRASTIVE_MODEL_PATH
    if p.exists():
        pr = p.resolve()
        if pr.is_file():
            sig = _file_sig(pr)
            return {"kind": "st_local_file", "path": str(pr), "file": sig}
        st = pr.stat()
        return {"kind": "st_local_dir", "path": str(pr), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
    return {"kind": "st_hf", "name": _DEFAULT_MODEL_NAME}


class L3SimilarityLayer(BaseLayer):
    """Cosine similarity against known attack embeddings."""

    name = "l3_similarity"

    def __init__(self, config: ShieldConfig | None = None) -> None:
        self._config = config or ShieldConfig()
        self._onnx_session: Any = None
        self._tokenizer: Any = None
        self._st_model: Any = None  # SentenceTransformer fallback
        self._index: Any = None
        self._attack_metadata: list[dict[str, str]] = []
        self._use_onnx = False

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling over non-padding tokens, then L2 normalize."""
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)

    @staticmethod
    def _download_onnx_model() -> None:
        """Auto-download L3 ONNX model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            _ONNX_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            logger.info("L3: downloading ONNX model from prompt-armor/l3-contrastive-onnx...")
            hf_hub_download(
                repo_id="prompt-armor/l3-contrastive-onnx",
                filename="model_quant.onnx",
                local_dir=str(_ONNX_MODEL_PATH),
            )
            hf_hub_download(
                repo_id="prompt-armor/l3-contrastive-onnx",
                filename="tokenizer.json",
                local_dir=str(_ONNX_MODEL_PATH),
            )
            logger.info("L3: ONNX model downloaded")
        except Exception as e:
            logger.warning("L3: auto-download failed: %s", e)

    def _encode_onnx(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts using ONNX model + tokenizers. Batched for efficiency."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self._tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            outputs = self._onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            pooled = self._mean_pool(outputs[0], attention_mask)
            all_embeddings.append(pooled)
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 384), dtype=np.float32)

    def _encode_single_onnx(self, text: str) -> np.ndarray:
        """Encode a single text using ONNX. Returns (1, 384)."""
        encoding = self._tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        outputs = self._onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        return self._mean_pool(outputs[0], attention_mask)

    def _cache_disabled(self) -> bool:
        return os.environ.get(_CACHE_DISABLE_ENV, "").strip().lower() in ("1", "true", "yes")

    def _try_load_cached_index(self, faiss_mod: Any, attacks_sig: dict[str, Any], model_fp: dict[str, Any]) -> bool:
        if self._cache_disabled():
            return False
        base = _l3_cache_base_dir()
        man_path = base / "manifest.json"
        idx_path = base / "index.faiss"
        meta_path = base / "metadata.json"
        if not (man_path.is_file() and idx_path.is_file() and meta_path.is_file()):
            return False
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("cache_version") != _CACHE_VERSION:
            return False
        if manifest.get("faiss_version") != faiss_mod.__version__:
            return False
        if manifest.get("use_onnx") != self._use_onnx:
            return False
        if manifest.get("attacks") != attacks_sig:
            return False
        if manifest.get("model_fingerprint") != model_fp:
            return False
        n_exp = manifest.get("n_attacks")
        if not isinstance(n_exp, int) or n_exp < 0:
            return False
        try:
            index = faiss_mod.read_index(str(idx_path))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("L3: cache read failed (%s), rebuilding index", e)
            return False
        if getattr(index, "d", None) != 384:
            return False
        if not isinstance(metadata, list) or len(metadata) != n_exp or index.ntotal != n_exp:
            return False
        self._attack_metadata = [
            {"category": str(m.get("category", "")), "source": str(m.get("source", "unknown"))} for m in metadata
        ]
        if len(self._attack_metadata) != n_exp:
            return False
        self._index = index
        logger.info("L3: loaded FAISS index from cache (%d vectors, %s)", n_exp, base)
        return True

    def _save_cached_index(
        self,
        faiss_mod: Any,
        attacks_sig: dict[str, Any],
        model_fp: dict[str, Any],
        n_attacks: int,
    ) -> None:
        if self._cache_disabled() or self._index is None or n_attacks <= 0:
            return
        base = _l3_cache_base_dir()
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("L3: could not create cache directory %s: %s", base, e)
            return
        manifest: dict[str, Any] = {
            "cache_version": _CACHE_VERSION,
            "faiss_version": faiss_mod.__version__,
            "use_onnx": self._use_onnx,
            "attacks": attacks_sig,
            "model_fingerprint": model_fp,
            "n_attacks": n_attacks,
            "dim": 384,
        }
        idx_path = base / "index.faiss"
        meta_path = base / "metadata.json"
        man_path = base / "manifest.json"
        pid = os.getpid()
        tmp_idx = base / f"index.faiss.tmp.{pid}"
        tmp_meta = base / f"metadata.json.tmp.{pid}"
        tmp_man = base / f"manifest.json.tmp.{pid}"
        try:
            faiss_mod.write_index(self._index, str(tmp_idx))
            os.replace(tmp_idx, idx_path)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(self._attack_metadata, f, ensure_ascii=False)
            os.replace(tmp_meta, meta_path)
            with open(tmp_man, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_man, man_path)
            logger.info("L3: wrote FAISS index cache (%d vectors, %s)", n_attacks, base)
        except Exception as e:
            logger.warning("L3: could not write index cache: %s", e)
            for p in (tmp_idx, tmp_meta, tmp_man):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def setup(self) -> None:
        """Load embedding model, build or load cached FAISS index from attack database."""
        import faiss

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        onnx_model = _ONNX_MODEL_PATH / "model_quant.onnx"
        onnx_tokenizer = _ONNX_MODEL_PATH / "tokenizer.json"

        if not onnx_model.exists() or not onnx_tokenizer.exists():
            self._download_onnx_model()

        if onnx_model.exists() and onnx_tokenizer.exists():
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._onnx_session = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(str(onnx_tokenizer))
            self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            self._tokenizer.enable_truncation(max_length=128)
            self._use_onnx = True
            logger.info("L3: using ONNX model")
        else:
            import io
            import sys

            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

            from sentence_transformers import SentenceTransformer

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                if _CONTRASTIVE_MODEL_PATH.exists():
                    self._st_model = SentenceTransformer(str(_CONTRASTIVE_MODEL_PATH))
                else:
                    self._st_model = SentenceTransformer(_DEFAULT_MODEL_NAME)
            finally:
                sys.stdout = old_stdout
            self._use_onnx = False
            logger.info("L3: using SentenceTransformer fallback")

        attacks_path = (self._config.attacks_path or _DEFAULT_ATTACKS_PATH).resolve()
        if not attacks_path.is_file():
            logger.warning("L3: attacks file not found: %s", attacks_path)
            self._index = faiss.IndexFlatIP(384)
            return

        attacks_sig = _attacks_file_sig(attacks_path)
        model_fp = _build_model_fingerprint(self._use_onnx)
        if self._try_load_cached_index(faiss, attacks_sig, model_fp):
            return

        texts: list[str] = []
        self._attack_metadata = []

        with open(attacks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cat = entry.get("category", "")
                if cat == "benign":
                    continue
                texts.append(entry["text"])
                self._attack_metadata.append({"category": cat, "source": entry.get("source", "unknown")})

        if not texts:
            self._index = faiss.IndexFlatIP(384)
            return

        if self._use_onnx:
            embeddings = self._encode_onnx(texts)
        else:
            embeddings = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        if n_vectors >= 10_000:
            n_clusters = min(int(np.sqrt(n_vectors)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
            self._index.add(embeddings)
            self._index.nprobe = min(16, n_clusters)
        else:
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)

        self._save_cached_index(faiss, attacks_sig, model_fp, n_vectors)

    def analyze(self, text: str) -> LayerResult:
        """Compare prompt against known attacks via cosine similarity."""
        start = time.perf_counter()

        if self._index is None or self._index.ntotal == 0:
            latency = (time.perf_counter() - start) * 1000
            return LayerResult(layer=self.name, score=0.0, confidence=0.5, latency_ms=latency)

        if not self._use_onnx and self._st_model is None:
            latency = (time.perf_counter() - start) * 1000
            return LayerResult(layer=self.name, score=0.0, confidence=0.5, latency_ms=latency)

        # Encode the input
        if self._use_onnx:
            embedding = self._encode_single_onnx(text)
        else:
            embedding = self._st_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        embedding = np.asarray(embedding, dtype=np.float32)

        # Search top-k similar attacks
        k = min(5, self._index.ntotal)
        scores, indices = self._index.search(embedding, k)

        top_similarity = float(scores[0][0])
        evidence: list[Evidence] = []
        categories_seen: set[Category] = set()

        for i in range(k):
            sim = float(scores[0][i])
            idx = int(indices[0][i])
            if sim < _MEDIUM_SIMILARITY:
                break

            meta = self._attack_metadata[idx]
            cat = _CATEGORY_MAP.get(meta["category"])
            if cat is None:
                continue

            evidence.append(
                Evidence(
                    layer=self.name,
                    category=cat,
                    description=f"Similarity {sim:.2f} to known {meta['category']} attack (source: {meta['source']})",
                    score=sim,
                )
            )
            categories_seen.add(cat)

        # Map similarity to risk score
        if top_similarity >= _HIGH_SIMILARITY:
            risk_score = 0.5 + (top_similarity - _HIGH_SIMILARITY) * 2.78
        elif top_similarity >= _MEDIUM_SIMILARITY:
            risk_score = (top_similarity - _MEDIUM_SIMILARITY) / (_HIGH_SIMILARITY - _MEDIUM_SIMILARITY) * 0.5
        else:
            risk_score = 0.0

        risk_score = min(1.0, max(0.0, risk_score))

        if top_similarity > 0.9 or top_similarity < 0.4:
            confidence = 0.95
        else:
            confidence = 0.7

        latency = (time.perf_counter() - start) * 1000
        return LayerResult(
            layer=self.name,
            score=round(risk_score, 4),
            confidence=round(confidence, 4),
            categories=tuple(sorted(categories_seen, key=lambda c: c.value)),
            evidence=tuple(evidence),
            latency_ms=round(latency, 2),
        )
