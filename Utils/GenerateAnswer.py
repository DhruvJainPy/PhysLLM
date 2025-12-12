# physllm_app.py  (updated full file)
import os
import re
import json
import torch
import pickle
import warnings
import zipfile
import tarfile
import shutil
from pathlib import Path
import functools
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# Load .env for local dev fallback
load_dotenv()

# ====== Force single-threaded BLAS/OpenMP for safety (helps avoid segfaults) ======
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

# Minimal import that is safe
import numpy as np

try:
    import streamlit as st

    def _cache_resource(fn):
        return st.cache_resource(fn)

    def _cache_data(fn):
        return st.cache_data(fn)
except Exception:
    def _cache_resource(fn):
        return functools.lru_cache(maxsize=1)(fn)

    def _cache_data(fn):
        return functools.lru_cache(maxsize=1)(fn)

# ----------------- Render / Streamlit secrets helper -----------------
def get_secret(key: str) -> str:
    """
    Robust secret loader:
    1) Try streamlit secrets (st.secrets) if available
    2) Then OS environment variables (Render exposes secrets as env vars)
    3) Then dotenv / os.getenv (local development)
    Raises RuntimeError if secret not found.
    """
    # 1) Try streamlit secrets if streamlit is imported and has secrets
    try:
        import streamlit as _st
        try:
            val = _st.secrets.get(key) if hasattr(_st, "secrets") else None
        except Exception:
            val = None
        if val:
            return val
    except Exception:
        # streamlit not available or import failed; continue to next source
        pass

    # 2) Check environment variables (Render uses env vars)
    val = os.environ.get(key)
    if val:
        return val

    # 3) Check dotenv / os.getenv (already loaded by load_dotenv earlier)
    val = os.getenv(key)
    if val:
        return val

    # Not found — raise to make the failure visible in logs
    raise RuntimeError(
        f"Missing secret: {key}. Set it in Render environment variables, "
        "or in Streamlit secrets (.streamlit/secrets.toml), or in your .env file for local dev."
    )

# Use the helper to obtain GOOGLE_API_KEY (works in Render)
try:
    google_key = get_secret("GOOGLE_API_KEY")
except Exception as e:
    # If you prefer non-fatal behavior, change this to set google_key = None
    raise

# ================================================================
# CONFIGURATION & PATHS (now dynamic: models downloaded at runtime)
# ================================================================
GEMINI_MODEL = "gemma-3-12b-it"
MINILM_TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Google Drive file IDs you provided (classifier, embedder)
# These IDs were taken from your provided links:
# https://drive.google.com/file/d/<ID>/view?usp=...
DRIVE_ID_CLASSIFIER = "1RbbMdpUKBhwRyv5s0ICJiBQGNAUBWSsX"
DRIVE_ID_EMBEDDER = "1rxn7AhE4VFi6xEIcXIvTHFpsPAOSfR3Z"
# If you later have FAISS / metadata as Drive files, add them here similarly:
DRIVE_ID_FAISS = None
DRIVE_ID_METADATA = None

# Local model storage (relative to repo root)
BASE_MODELS_DIR = Path("models")
VECTOR_STORE_DIR = Path("VectorStore")

CLASSIFIER_DIR = BASE_MODELS_DIR / "classifier"
EMBEDDER_PATH = BASE_MODELS_DIR / "embedder"

# FAISS / metadata paths (can be downloaded separately if you add Drive IDs)
INDEX_PATH = VECTOR_STORE_DIR / "faiss_minilm_final.index"
META_PATH = VECTOR_STORE_DIR / "faiss_metadata.pkl"

# ================================================================
# Utilities: download from Google Drive and extract
# ================================================================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _download_drive_file(file_id: str, out_path: Path, quiet: bool = False):
    """
    Download a file from Google Drive using gdown.
    If gdown not installed, raise informative error.
    """
    try:
        import gdown
    except Exception as e:
        raise RuntimeError("gdown is required to download files from Google Drive. Install it via `pip install gdown`.") from e

    url = f"https://drive.google.com/uc?id={file_id}"
    # gdown will write to out_path; we ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[download] already exists: {out_path}, skipping download.")
        return out_path
    print(f"[download] fetching {url} -> {out_path} ...")
    # we use gdown.download with output
    gdown.download(url, str(out_path), quiet=quiet)
    if not out_path.exists():
        raise RuntimeError(f"Download failed, file not found at {out_path}")
    print(f"[download] completed: {out_path} (size: {out_path.stat().st_size} bytes)")
    return out_path

def _extract_archive_if_needed(archive_path: Path, target_dir: Path):
    """
    If archive_path is a zip or tar, extract into target_dir.
    If not archive, move the file into target_dir preserving filename.
    """
    _ensure_dir(target_dir)
    # ZIP
    if zipfile.is_zipfile(archive_path):
        print(f"[extract] {archive_path} is a zip, extracting to {target_dir}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return True
    # TAR
    try:
        if tarfile.is_tarfile(archive_path):
            print(f"[extract] {archive_path} is a tar archive, extracting to {target_dir}")
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_dir)
            return True
    except Exception:
        pass

    # Not an archive: move the single file into target_dir
    dest = target_dir / archive_path.name
    if dest.exists():
        print(f"[extract] destination exists {dest}, skipping move")
    else:
        print(f"[extract] moving {archive_path} -> {dest}")
        shutil.move(str(archive_path), str(dest))
    return False

def ensure_models_downloaded(force: bool = False):
    """
    Ensure classifier and embedder models are present on disk.
    Downloads from Google Drive if not present.
    """
    _ensure_dir(BASE_MODELS_DIR)

    # Classifier
    if DRIVE_ID_CLASSIFIER is None:
        print("[ensure_models] DRIVE_ID_CLASSIFIER not provided; skipping classifier download")
    else:
        if force or not CLASSIFIER_DIR.exists() or not any(CLASSIFIER_DIR.iterdir()):
            print("[ensure_models] Downloading classifier model from Google Drive...")
            classifier_archive = BASE_MODELS_DIR / "classifier_download"
            # prefer zip extension for downloaded file name to allow extraction
            classifier_archive_zip = classifier_archive.with_suffix(".zip")
            _download_drive_file(DRIVE_ID_CLASSIFIER, classifier_archive_zip)
            _extract_archive_if_needed(classifier_archive_zip, CLASSIFIER_DIR)
            # remove the archive file if it still exists
            if classifier_archive_zip.exists():
                try:
                    classifier_archive_zip.unlink()
                except Exception:
                    pass
            print(f"[ensure_models] Classifier available at: {CLASSIFIER_DIR}")
        else:
            print(f"[ensure_models] Classifier already present at {CLASSIFIER_DIR}, skipping download")

    # Embedder
    if DRIVE_ID_EMBEDDER is None:
        print("[ensure_models] DRIVE_ID_EMBEDDER not provided; skipping embedder download")
    else:
        if force or not EMBEDDER_PATH.exists() or not any(EMBEDDER_PATH.iterdir()):
            print("[ensure_models] Downloading embedder model from Google Drive...")
            embedder_archive = BASE_MODELS_DIR / "embedder_download"
            embedder_archive_zip = embedder_archive.with_suffix(".zip")
            _download_drive_file(DRIVE_ID_EMBEDDER, embedder_archive_zip)
            _extract_archive_if_needed(embedder_archive_zip, EMBEDDER_PATH)
            if embedder_archive_zip.exists():
                try:
                    embedder_archive_zip.unlink()
                except Exception:
                    pass
            print(f"[ensure_models] Embedder available at: {EMBEDDER_PATH}")
        else:
            print(f"[ensure_models] Embedder already present at {EMBEDDER_PATH}, skipping download")

    # If FAISS/META drive ids are provided, you can add similar logic here.

# Attempt to download models now (this runs at import/cold-start time)
# It's useful for deployment logs to see this happen.
try:
    ensure_models_downloaded()
except Exception as e:
    # Fail fast so deployment logs show the issue - avoids obscure downstream errors.
    print("[ERROR] Failed to ensure models downloaded:", e)
    raise

# ================================================================
# CACHED LOADERS (tokenizer/model, label_map, faiss, embeddings/LLM)
# ================================================================
@_cache_resource
def load_classifier(classifier_dir=CLASSIFIER_DIR):
    """Load tokenizer and classifier model once (cached)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # classifier_dir may be a folder path or contain a single file; AutoTokenizer expects a folder or HF repo.
    if not Path(classifier_dir).exists():
        raise FileNotFoundError(f"Classifier directory not found: {classifier_dir}")
    tok = AutoTokenizer.from_pretrained(str(classifier_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(classifier_dir))
    model.eval()
    return tok, model


@_cache_data
def load_label_map(classifier_dir=CLASSIFIER_DIR):
    """Load label map (cached)."""
    p = Path(classifier_dir) / "label_map.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if "id2label" in label_map:
        id2label = {int(k): v for k, v in label_map["id2label"].items()}
    elif "label2id" in label_map:
        id2label = {int(v): k for k, v in label_map["label2id"].items()}
    else:
        id2label = {}
    return id2label


@_cache_resource
def load_faiss_index_and_metadata(index_path=INDEX_PATH, meta_path=META_PATH):
    """Load FAISS index and metadata (cached)."""
    import faiss
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"FAISS metadata file not found: {meta_path}")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    
    texts = []
    topics = []
    if isinstance(metadata, dict):
        if "texts" in metadata and isinstance(metadata["texts"], list):
            texts = [str(t) for t in metadata["texts"]]
        elif "docs" in metadata and isinstance(metadata["docs"], list):
            texts = [str(t) for t in metadata["docs"]]
        elif "contents" in metadata and isinstance(metadata["contents"], list):
            texts = [str(t) for t in metadata["contents"]]
        elif "page_contents" in metadata and isinstance(metadata["page_contents"], list):
            texts = [str(t) for t in metadata["page_contents"]]
        if "topics" in metadata and isinstance(metadata["topics"], list):
            topics = [str(t) for t in metadata["topics"]]
        elif "titles" in metadata and isinstance(metadata["titles"], list):
            topics = [str(t) for t in metadata["titles"]]
        if not texts:
            list_values = [v for v in metadata.values() if isinstance(v, list)]
            if len(list_values) == 1:
                texts = [str(t) for t in list_values[0]]

    elif isinstance(metadata, list):
        if all(isinstance(x, str) for x in metadata):
            texts = [str(x) for x in metadata]
            topics = [""] * len(texts)
        elif all(isinstance(x, dict) for x in metadata):
            for item in metadata:
                text = None
                for key in ("content", "page_content", "text", "body", "doc", "chunk"):
                    if key in item and isinstance(item[key], str) and item[key].strip():
                        text = item[key].strip()
                        break
                if text is None:
                    text = str(item)
                texts.append(text)
                topic = None
                for tkey in ("topic", "title", "section", "meta_title"):
                    if tkey in item and isinstance(item[tkey], str) and item[tkey].strip():
                        topic = item[tkey].strip()
                        break
                topics.append(topic if topic is not None else "")
        else:
            texts = [str(x) for x in metadata]
    else:
        texts = [str(metadata)]

    if not topics:
        topics = [""] * len(texts)
    if len(topics) != len(texts):
        if len(topics) > len(texts):
            topics = topics[: len(texts)]
        else:
            topics = topics + [""] * (len(texts) - len(topics))

    print(f"Loaded metadata: {len(texts)} texts, {len(topics)} topics")
    for i in range(min(3, len(texts))):
        print(f" sample[{i}] len={len(texts[i])} topic={topics[i]!r}")

    if index.ntotal != len(texts):
        print(f"[Warning] FAISS index.ntotal={index.ntotal} but metadata texts length={len(texts)}. They should match!")
    else:
        print("FAISS index and metadata appear aligned (ntotal == len(texts)).")
        
    return index, texts, topics

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (B, L, H)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

@_cache_resource
def load_embedding_and_llm_clients(gemini_model=GEMINI_MODEL):
    """Create embeddings and LLM clients (cached)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    minilm_tokenizer = AutoTokenizer.from_pretrained(MINILM_TOKENIZER_NAME)
    if not Path(EMBEDDER_PATH).exists():
        raise FileNotFoundError(f"Embedder model directory not found: {EMBEDDER_PATH}")
    embeddings_client = AutoModel.from_pretrained(str(EMBEDDER_PATH))
    embeddings_client.eval()
    
    llm_client = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.0,
        top_p=1.0
    )

    # Separate normalizer instance; same model/params but separate object for clarity
    normalizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, top_p=1.0)

    return minilm_tokenizer,embeddings_client, llm_client, normalizer_llm


# Try to restrict torch threads to reduce native threading conflicts
try:
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass

# ================================================================
# Populate module-level objects using cached loaders
# ================================================================
# These calls will use the downloaded model folders (ensured earlier)
clf_tokenizer, clf_model = load_classifier()
id2label = load_label_map()
index, texts, topics = load_faiss_index_and_metadata()
minilm_tokenizer, embeddings_client, llm, normalizer_llm = load_embedding_and_llm_clients()

# ================================================================
# CLASSIFICATION UTIL
# ================================================================
def classify_query(query: str) -> str:
    """Classify query into predefined labels."""
    encoded = clf_tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    clf_model.eval()
    with torch.no_grad():
        outputs = clf_model(**encoded)
        pred = int(torch.argmax(outputs.logits, dim=-1).item())
    return id2label.get(pred, str(pred))


# ================================================================
# QUESTION NORMALIZER
# ================================================================
QUESTION_NORMALIZER_PROMPT = """
You rewrite student questions into a single NCERT-style canonical question.
Meaning of “keep meaning IDENTICAL”:
- Preserve the SAME physics concept being asked.
- Do NOT add formulas, steps, values, or assumptions not in the user’s original question.
- Only rewrite for clarity, correctness, and NCERT tone.

Also classify whether the question is NUMERICAL or CONCEPTUAL.
Output format:
canonical_question: <rewritten NCERT-style question>
type: <numerical or conceptual>

Examples:
User: "tell me newton first law in simple words?"
canonical_question: What is Newton’s first law of motion?
type: conceptual

User: "a force of 10N acts on a mass 5kg find acceleration"
canonical_question: What is the acceleration of a 5 kg mass when a force of 10 N acts on it?
type: numerical

User: "definition of work?"
canonical_question: What is the definition of work in physics?
type: conceptual

User: "find electric field at 2cm from 5mC charge"
canonical_question: What is the electric field at a distance of 2 cm from a 5 mC point charge?
type: numerical

Now rewrite the new user question.
User question: {query}
canonical_question:
"""

def _parse_normalizer_output(text: str):
    """Parse normalizer LLM output to [canonical_question, TYPE]."""
    # Normalize line endings and lowercase markers for robust parsing
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    canonical = None
    qtype = None
    for l in lines:
        low = l.lower()
        if low.startswith("canonical_question:") or low.startswith("canonical question:"):
            # take everything after colon
            canonical = l.split(":", 1)[1].strip()
        elif low.startswith("type:"):
            qtype = l.split(":", 1)[1].strip().upper()
        else:
            # fallback: maybe the first non-empty line is the canonical question
            if canonical is None:
                canonical = l
    # sanitize qtype
    if qtype is None:
        # attempt to detect keywords
        if canonical and re.search(r"\b(find|calculate|compute|determine|what is the|how many|how much)\b", canonical, re.I):
            qtype = "NUMERICAL"
        else:
            qtype = "CONCEPTUAL"
    if canonical is None:
        canonical = ""
    return canonical, qtype

def normalize_question(query: str) -> list:
    """Normalize user question into canonical NCERT-style question. Returns [canonical_question, TYPE]."""
    try:
        out = normalizer_llm.invoke(QUESTION_NORMALIZER_PROMPT.format(query=query))
        normalized_text = out.content.strip() if hasattr(out, "content") else str(out).strip()
        canonical, qtype = _parse_normalizer_output(normalized_text)
        # ensure canonical is not empty — fallback to cleaned query
        if not canonical:
            q = query.strip()
            q = re.sub(r'\s+', ' ', q)
            return [q, "CONCEPTUAL"]
        return [canonical, qtype]
    except Exception:
        # Fallback: simple normalization
        q = query.strip()
        q = re.sub(r'\s+', ' ', q)
        return [q, "CONCEPTUAL"]

# ================================================================
# RETRIEVAL (FAISS) & RESCORING
# ================================================================
def retrieve_candidates_faiss(query, fetch_k=10):
    """Retrieve top candidates from FAISS."""
    toks = minilm_tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        out = embeddings_client(**toks)
        emb = mean_pooling(out, toks["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p = 2, dim = 1)
        qv = emb.numpy().astype("float32")[0]
        
    qv = np.array(qv, dtype="float32").reshape(1, -1)
    import faiss as _faiss
    _faiss.normalize_L2(qv)
    distances, indices = index.search(qv, fetch_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx < 0 or idx >= len(texts):
            continue
        results.append({
            "rank": rank,
            "chunk_id": int(idx),
            "topic": topics[idx] if idx < len(topics) else "unknown",
            "text": texts[idx],
            "faiss_score": float(score)
        })

    print(f"[retrieve] fetched {len(results)} candidates (fetch_k={fetch_k}) for query='{query[:80]}...'")
    for r in results[:5]:
        print(f" rank={r['rank']} chunk_id={r['chunk_id']} faiss_score={r['faiss_score']:.4f}")

    return results

def retrieve_similar_chunks(query, k=10, fetch_k=10):
    """Retrieve top-k most relevant chunks for a query."""
    candidates = retrieve_candidates_faiss(query, fetch_k=fetch_k)
    if not candidates:
        return []

    topk = candidates[:k]
    return [(c["text"],c["faiss_score"], c["chunk_id"]) for c in topk]

# ================================================================
# HALLUCINATION / SUPPORT CHECK
# ================================================================
def sentence_word_overlap_support(sentence: str, chunk_texts: list) -> float:
    """Compute support score for a sentence based on overlapping words with chunks."""
    sent = re.sub(r'[^a-z0-9\s]', ' ', sentence.lower())
    words = [w for w in sent.split() if len(w) >= 3]

    if not words:
        return 0.0

    matched = 0
    for w in words:
        for txt in chunk_texts:
            if w in txt:
                matched += 1
                break

    return matched / len(words)

def simple_verifier(answer_text: str, chunks: list) -> tuple[bool, list, list]:
    """Verify if answer is supported by chunks."""
    chunk_texts = [c["text"].lower() for c in chunks]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer_text) if s.strip()]

    if not sentences:
        return False, [], []

    per_support = [(s, sentence_word_overlap_support(s, chunk_texts)) for s in sentences]
    unsupported = [s for s, sc in per_support if sc < 0.35]

    supported_count = sum(1 for s, sc in per_support if sc >= 0.35)
    context_coverage = supported_count / len(per_support)
    verified = context_coverage >= 0.9

    return verified, unsupported, per_support

# ================================================================
# PROMPTS
# ================================================================
GEN_PROMPT_TEMPLATE = """
You are an NCERT physics tutor. Use ONLY the provided Context to answer. Never use outside knowledge. Never add new assumptions.
The question type is: {qtype} (question type is either CONCEPTUAL or NUMERICAL)
Follow the corresponding output format strictly.
CONVERT ANY LATEX EXPRESSIONS TO PLAIN TEXT.


--------------------
If qtype = CONCEPTUAL:
ANSWER (Conceptual): <Multi-paragraph explanation using ONLY the Context. Use **bold** for key terms. Cite like [Chunk X].>
SUMMARY:
- 3–5 crisp revision bullets.

--------------------
If qtype = NUMERICAL, follow the detailed NUMERICAL structure:
1. GIVEN (with units)
2. FORMULAE & PLAN (cite chunks)
3. DETAILED CALCULATION (symbolic → numeric with intermediate steps)
4. SANITY CHECK
5. FINAL ANSWER

--------------------
If the Context lacks the required information, output EXACTLY:
ANSWER: The context does not provide enough information.
SUMMARY: None

--------------------
CONTEXT: {context}
QUESTION: {query}
"""

REPAIR_PROMPT_TEMPLATE = """
You are an NCERT physics tutor. The previous answer contained unsupported statements.
Rewrite the answer using ONLY these snippet IDs: {supported_ids}. Do NOT add any extra knowledge.
Use the same required structure based on the question type: {qtype}

CONTEXT: {context}
PRIOR ANSWER: {prior}
Rewritten answer:
"""

def is_explicit_not_enough_message(text: str) -> bool:
    cleaned = re.sub(r'[\s\.\-_,:;]+', ' ', text.strip().lower())
    return "the context does not provide enough information" in cleaned

# ================================================================
# MAIN PIPELINE
# ================================================================
def generate_answer(user_query, k=10, fetch_k=10):
    # STEP 1: Classify
    try:
        label = classify_query(user_query)
    except Exception as e:
        # non-fatal: log and continue
        print("[pipeline] classifier error:", e)
        label = "unknown"
    print(f"[pipeline] classified question as: {label}")

    # STEP 2: Normalize
    normalized = normalize_question(user_query)
    print(f"[pipeline] normalized question: {normalized}")

    # normalized is expected to be [canonical_question, TYPE]
    if not isinstance(normalized, (list, tuple)) or len(normalized) < 2:
        normalized = [str(normalized), "CONCEPTUAL"]

    # STEP 3: Retrieve
    retrieved = retrieve_similar_chunks(normalized[0], k=k, fetch_k=fetch_k)
    if not retrieved:
        return (
            "ANSWER: The context does not provide enough information.\n"
            "RETRIEVAL: No chunks were retrieved. Check your FAISS index or embedding generation."
        )

    # Build context
    top_chunks = []
    ctx_blocks = []
    for text,faiss_score, cid in retrieved:
        top_chunks.append({"chunk_id": cid, "text": text})
        excerpt = text.replace("\n", " ")
        ctx_blocks.append(f"[Chunk {cid}]\n{excerpt}")
    context_final = "\n\n".join(ctx_blocks)

    # STEP 4: Generate answer
    prompt = GEN_PROMPT_TEMPLATE.format(context=context_final, query=normalized[0], qtype=normalized[1])
    raw_resp = llm.invoke(prompt)
    raw = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
    raw = raw.strip()
    print("[llm] generated answer length:", len(raw))

    if is_explicit_not_enough_message(raw):
        return "ANSWER: The context does not provide enough information."

    # STEP 5: Verify
    verified, unsupported, per_supports = simple_verifier(raw, top_chunks)
    if verified:
        return raw  # Return verified answer directly

    # STEP 6: Repair
    allowed_ids = ",".join([str(c["chunk_id"]) for c in top_chunks])
    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
        supported_ids=allowed_ids, context=context_final, prior=raw, qtype=normalized[1]
    )
    repaired_resp = llm.invoke(repair_prompt)
    repaired = repaired_resp.content if hasattr(repaired_resp, "content") else str(repaired_resp)
    repaired = repaired.strip()
    print("[llm] repaired answer length:", len(repaired))

    if is_explicit_not_enough_message(repaired):
        top_ids = [str(c["chunk_id"]) for c in top_chunks]
        debug_ctx_preview = "\n".join([f"[Chunk {c['chunk_id']}] {c['text'][:140]}" for c in top_chunks])
        return (
            "ANSWER: The context does not provide enough information.\n"
            f"TOP_RETRIEVED_CHUNKS: {','.join(top_ids)}\n"
            f"CONTEXT_PREVIEW:\n{debug_ctx_preview}"
        )

    return repaired
