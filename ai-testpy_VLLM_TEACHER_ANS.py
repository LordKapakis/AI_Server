# ai-testpy_VLLM.py
# Stable server with:
# - legacy flat domains supported
# - nested Courses/<Course>/YEAR_X/SEM_Y/<Lesson> supported
# - year-scope escalation via scope="year"
# - ✅ accepts year alias: Courses__<Course>__YEAR_X
# - safe file serving for nested domain folders

from flask import Flask, request, jsonify, abort, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import safe_join
import os, io, re, time, json, hashlib, shutil, unicodedata
import numpy as np
import requests
import PyPDF2

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import docx
except Exception:
    docx = None

try:
    import pptx
except Exception:
    pptx = None

try:
    import torch
except Exception:
    torch = None

app = Flask(__name__)
CORS(app)

# ============
# Config
# ============
API_KEY = os.getenv("API_KEY", "").strip()
if not API_KEY:
    print("WARNING: API_KEY is empty. Set it via env for production use.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FOLDER = os.getenv("BASE_FOLDER", os.path.join(SCRIPT_DIR, "domains"))

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))

MODEL_NAME = os.getenv("MODEL_NAME", "vLLM-chat")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "140"))

# vLLM token budgeting (prevents 400s when prompt is large)
VLLM_MAX_CONTEXT = int(os.getenv("VLLM_MAX_CONTEXT", "2048"))
VLLM_SAFETY_MARGIN = int(os.getenv("VLLM_SAFETY_MARGIN", "256"))
VLLM_MIN_COMPLETION = int(os.getenv("VLLM_MIN_COMPLETION", "96"))
VLLM_MAX_COMPLETION_CAP = int(os.getenv("VLLM_MAX_COMPLETION_CAP", str(MAX_TOKENS)))

# Retrieval tuning
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "280"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
R_TOP_K = int(os.getenv("R_TOP_K", "4"))
NUM_ALTERNATE_QUERIES = int(os.getenv("NUM_ALTERNATE_QUERIES", "1"))
MIN_SIM_THRESHOLD = float(os.getenv("MIN_SIM_THRESHOLD", "0.30"))

STRICT_MODE = os.getenv("STRICT_MODE", "true").lower() in ("1", "true", "yes")
EXTRACTIVE_ONLY = os.getenv("EXTRACTIVE_ONLY", "false").lower() in ("1", "true", "yes")

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
MIN_WORDS_TO_INDEX = int(os.getenv("MIN_WORDS_TO_INDEX", "20"))

# Teacher answers (per-domain PDF)
TEACHER_MIN_SIM = float(os.getenv("TEACHER_MIN_SIM", "0.55"))

# { domain_id: { "questions": [...], "answers": [...],
#               "embeddings": np.ndarray, "pages": [...], "file": "..." } }
teacher_answers_cache = {}

# Index cache
INDEX_CACHE_DIR = os.getenv("INDEX_CACHE_DIR", os.path.join(SCRIPT_DIR, ".index_cache"))
os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

CACHE_ADMIN_KEY = os.getenv("CACHE_ADMIN_KEY", "").strip()

SUPPORTED_EXTS = (".pdf", ".txt", ".docx", ".pptx")

# Embedding model
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
if not EMBEDDING_DEVICE:
    if torch is not None and torch.cuda.is_available():
        EMBEDDING_DEVICE = "cuda"
    else:
        EMBEDDING_DEVICE = "cpu"

if SentenceTransformer is None:
    raise RuntimeError("sentence-transformers is required")
if faiss is None:
    raise RuntimeError("faiss is required")

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=EMBEDDING_DEVICE)

PROMPT_LANG = (
    "Απάντησε πάντα στα Ελληνικά, με φυσική και κατανοητή γλώσσα. "
    "Αν η ερώτηση είναι σε άλλη γλώσσα, μετέφρασέ την συνοπτικά και απάντησε στα Ελληνικά."
)

# =========================
# In-memory indexes
# =========================
# domain_data:
# { domainId: { "chunks": [...], "index": faiss.IndexFlatIP, "meta": [...],
#              "embeddings": np.ndarray, "folder": "...", "kind": "lesson|year",
#              "children": [...]} }
domain_data = {}

# domainId -> folder path
domain_folders = {}

# mapping for nested Courses/<Course>/YEAR_X/SEM_Y/<Lesson>
lesson_to_year = {}
year_alias_to_year = {}


# =========================
# Helpers
# =========================
def require_api_key():
    key = request.headers.get("x-api-key") or ""
    if not API_KEY:
        # For local dev, we allow missing api-key but log it
        if key:
            print("⚠️ API_KEY not configured on server but client sent a key.")
        return
    if key != API_KEY:
        abort(401, description="Unauthorized")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def rough_tokenize(text: str) -> int:
    # crude: count words as tokens
    return len((text or "").split())


def sliding_word_chunks(words, max_tokens, overlap_tokens):
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += 1
        if cur_len >= max_tokens:
            chunks.append(" ".join(cur))
            if overlap_tokens > 0:
                cur = cur[-overlap_tokens:]
                cur_len = len(cur)
            else:
                cur = []
                cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def safe_read_pdf_text(path):
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            texts = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    texts.append(t)
            return "\n".join(texts)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""


def safe_read_docx_text(path):
    if docx is None:
        return ""
    try:
        d = docx.Document(path)
        return "\n".join([p.text for p in d.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")
        return ""


def safe_read_pptx_text(path):
    if pptx is None:
        return ""
    try:
        pres = pptx.Presentation(path)
        texts = []
        for slide in pres.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception as e:
        print(f"Error reading PPTX {path}: {e}")
        return ""


def read_file_text(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return safe_read_pdf_text(path)
    elif lower.endswith(".docx"):
        return safe_read_docx_text(path)
    elif lower.endswith(".pptx"):
        return safe_read_pptx_text(path)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {path}: {e}")
            return ""


def embed_texts(texts):
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        emb = embedding_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_vecs.append(emb.astype(np.float32))
    return np.vstack(all_vecs)


def load_cached_file(domain_id: str, fpath: str, chunk_tokens: int, overlap_tokens: int):
    """
    Load cached chunks+embeddings for a single file, if present.
    """
    rel = os.path.relpath(fpath, BASE_FOLDER)
    key_str = f"{domain_id}::{rel}::{chunk_tokens}::{overlap_tokens}"
    h = hashlib.sha256(key_str.encode("utf-8", errors="ignore")).hexdigest()
    folder = os.path.join(INDEX_CACHE_DIR, h)
    meta_path = os.path.join(folder, "meta.json")
    emb_path = os.path.join(folder, "embeddings.npy")
    if not (os.path.isfile(meta_path) and os.path.isfile(emb_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        embeddings = np.load(emb_path)
    except Exception as e:
        print(f"Failed to load cached file index for {fpath}: {e}")
        return None
    return data, embeddings


def save_cached_file(domain_id: str, fpath: str, chunk_tokens: int, overlap_tokens: int, data, embeddings):
    """
    Save chunks+embeddings for a single file.
    """
    rel = os.path.relpath(fpath, BASE_FOLDER)
    key_str = f"{domain_id}::{rel}::{chunk_tokens}::{overlap_tokens}"
    h = hashlib.sha256(key_str.encode("utf-8", errors="ignore")).hexdigest()
    folder = os.path.join(INDEX_CACHE_DIR, h)
    os.makedirs(folder, exist_ok=True)
    meta_path = os.path.join(folder, "meta.json")
    emb_path = os.path.join(folder, "embeddings.npy")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        np.save(emb_path, embeddings)
    except Exception as e:
        print(f"Failed to save cached file index for {fpath}: {e}")


def build_domain_index(domain_id: str, folder: str, kind: str = "lesson", children=None):
    """
    Build a FAISS index for a domain (folder).
    Reuses per-file caches stored under INDEX_CACHE_DIR.
    """
    t0 = time.time()
    children = children or []
    print(f"Indexing domain '{domain_id}' (kind={kind}) from folder: {folder}")
    all_chunks = []
    all_meta = []
    all_embs = []

    for root, dirs, files in os.walk(folder):
        for fname in files:
            lower = fname.lower()
            if not lower.endswith(SUPPORTED_EXTS):
                continue
            fpath = os.path.join(root, fname)
            text = read_file_text(fpath)
            text = normalize_text(text)
            if not text:
                continue
            words = text.split()
            if len(words) < MIN_WORDS_TO_INDEX:
                continue

            cached = load_cached_file(domain_id, fpath, CHUNK_TOKENS, CHUNK_OVERLAP)
            if cached is not None:
                data, emb = cached
                chunks = data["chunks"]
                meta_list = data["meta"]
                print(f"  ↪ reused cache for {fname}: {len(chunks)} chunks")
            else:
                # fresh chunking
                w_chunks = sliding_word_chunks(words, CHUNK_TOKENS, CHUNK_OVERLAP)
                chunks = []
                meta_list = []
                for ch in w_chunks:
                    chunks.append(ch)
                    meta_list.append(
                        {
                            "file": fname,
                            "rel": os.path.relpath(fpath, folder),
                            "path": fpath,
                            "page": None,
                        }
                    )
                emb = embed_texts(chunks)
                save_cached_file(
                    domain_id,
                    fpath,
                    CHUNK_TOKENS,
                    CHUNK_OVERLAP,
                    {"chunks": chunks, "meta": meta_list},
                    emb,
                )
                print(f"  ↪ indexed {fname}: {len(chunks)} chunks")

            all_chunks.extend(chunks)
            all_meta.extend(meta_list)
            all_embs.append(emb)

    if all_embs:
        embeddings = np.vstack(all_embs)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        embeddings = np.zeros((0, 384), dtype=np.float32)
        index = faiss.IndexFlatIP(384)

    domain_data[domain_id] = {
        "chunks": all_chunks,
        "embeddings": embeddings,
        "index": index,
        "meta": all_meta,
        "folder": folder,
        "kind": kind,
        "children": children,
    }
    domain_folders[domain_id] = folder

    dt = time.time() - t0
    print(
        f"✅ Indexed {len(all_chunks)} chunks for '{domain_id}' "
        f"(kind={kind}, children={len(children)}) in {dt:.2f}s."
    )
    return domain_data[domain_id]


def ensure_domain_index(domain_id: str, folder: str, kind: str = "lesson", children=None):
    if domain_id in domain_data and domain_data[domain_id].get("index") is not None:
        return domain_data[domain_id]
    children = children or []
    return build_domain_index(domain_id, folder, kind, children)


def discover_flat_domains():
    """
    Legacy flat domains: BASE_FOLDER/<domainId> with PDFs, etc.
    """
    if not os.path.isdir(BASE_FOLDER):
        return
    for name in os.listdir(BASE_FOLDER):
        if name == "Courses":
            continue
        path = os.path.join(BASE_FOLDER, name)
        if os.path.isdir(path):
            domain_folders[name] = path
            print(f"Discovered flat domain: {name} -> {path}")


def slugify_course_name(name):
    # used for YEAR__<CourseName>__YEAR_X
    return name.replace(" ", "_")


def discover_courses_hierarchy():
    """
    Discover nested Courses/<Course>/YEAR_X/SEM_Y/<Lesson>.
    For each leaf lesson, register a domain id:
      Courses__<Course>__YEAR_X__SEM_Y__<Lesson>
    Also build a year aggregate domain:
      YEAR__<Course>__YEAR_X
    """
    courses_root = os.path.join(BASE_FOLDER, "Courses")
    if not os.path.isdir(courses_root):
        print(f"No Courses root at {courses_root}")
        return

    for course_name in os.listdir(courses_root):
        course_path = os.path.join(courses_root, course_name)
        if not os.path.isdir(course_path):
            continue

        for year_name in os.listdir(course_path):
            if not re.match(r"YEAR_\d+", year_name, re.IGNORECASE):
                continue
            year_path = os.path.join(course_path, year_name)
            if not os.path.isdir(year_path):
                continue

            course_slug = slugify_course_name(course_name)
            year_dom = f"YEAR__{course_slug}__{year_name}"

            lesson_domains = []

            for sem_name in os.listdir(year_path):
                sem_path = os.path.join(year_path, sem_name)
                if not os.path.isdir(sem_path):
                    continue

                for lesson_name in os.listdir(sem_path):
                    lesson_path = os.path.join(sem_path, lesson_name)
                    if not os.path.isdir(lesson_path):
                        continue

                    lesson_dom = f"Courses__{course_name}__{year_name}__{sem_name}__{lesson_name}"
                    domain_folders[lesson_dom] = lesson_path
                    lesson_domains.append(lesson_dom)
                    lesson_to_year[lesson_dom] = year_dom

                    print(f"Discovered lesson domain: {lesson_dom} -> {lesson_path}")

            if lesson_domains:
                domain_folders[year_dom] = year_path
                domain_data[year_dom] = {
                    "folder": year_path,
                    "kind": "year",
                    "children": lesson_domains,
                    "chunks": [],
                    "embeddings": np.zeros((0, 384), dtype=np.float32),
                    "index": faiss.IndexFlatIP(384),
                    "meta": [],
                }
                print(
                    f"Registered year aggregate domain: {year_dom} -> {year_path}, "
                    f"children={len(lesson_domains)}"
                )

                alias_key = f"Courses__{course_name}__{year_name}"
                alias_key = alias_key.replace(" ", "_")
                year_alias_to_year[alias_key] = year_dom
                print(f"Alias registered: {alias_key} -> {year_dom}")


def init_domains():
    discover_flat_domains()
    discover_courses_hierarchy()


init_domains()


def retrieve_chunks(domain_id: str, query: str, top_k: int = R_TOP_K, alt_queries=None):
    """
    Retrieve top_k chunks for a query (and optional alternate queries) from domain.
    Returns: list of (chunk_text, score, meta)
    """
    if domain_id not in domain_folders:
        raise ValueError(f"Unknown domain: {domain_id}")
    info = domain_data.get(domain_id)
    if not info or "index" not in info:
        folder = domain_folders[domain_id]
        kind = info.get("kind", "lesson") if info else "lesson"
        children = info.get("children") if info else None
        info = ensure_domain_index(domain_id, folder, kind, children)

    index = info["index"]
    embeddings = info["embeddings"]
    chunks = info["chunks"]
    meta = info["meta"]

    if embeddings.shape[0] == 0:
        return []

    queries = [query] + (alt_queries or [])
    q_emb = embed_texts(queries)
    D, I = index.search(q_emb, top_k)

    results = []
    used = set()
    for row in range(D.shape[0]):
        for col in range(D.shape[1]):
            idx = int(I[row, col])
            if idx < 0 or idx >= len(chunks):
                continue
            if idx in used:
                continue
            score = float(D[row, col])
            if score < MIN_SIM_THRESHOLD:
                continue
            used.add(idx)
            results.append((chunks[idx], score, meta[idx]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def auto_extract_quotes(chunks):
    """
    Extract short bullet-like segments from chunks for quote-based answers
    when vLLM is not available.
    """
    lines = []
    for ch, _, _ in chunks:
        for ln in ch.split(". "):
            ln = ln.strip()
            if not ln:
                continue
            if len(ln.split()) < 200:
                lines.append("- " + ln.strip())
    return "\n".join(lines)


def build_prompt_from_chunks(question, retrieved, chat_messages=None):
    """
    Build the final prompt given retrieved chunks and a question.
    """
    context_parts = []
    sources = []

    for i, (chunk, score, m) in enumerate(retrieved, start=1):
        file = m.get("file")
        page = m.get("page")
        snippet = chunk
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"

        sources.append(
            {
                "label": i,
                "file": file,
                "page_start": page,
                "page_end": page,
                "snippet": snippet,
            }
        )

        context_parts.append(f"[{i}] ({file}, σελίδα {page}) {chunk}")

    history_str = ""
    if chat_messages:
        history_items = []
        for msg in chat_messages[-4:]:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "user":
                history_items.append(f"Χρήστης: {content}")
            elif role == "assistant":
                history_items.append(f"Βοηθός: {content}")
        if history_items:
            history_str = "\n\nΙστορικό συνομιλίας:\n" + "\n".join(history_items)

    context_text = "\n\n".join(context_parts) if context_parts else ""
    instructions = (
        f"{PROMPT_LANG}\n"
        "Χρησιμοποίησε ΜΟΝΟ τις παρακάτω πληροφορίες από τα έγγραφα για να απαντήσεις. "
        "Αν οι πληροφορίες δεν επαρκούν, πες ξεκάθαρα ότι δεν είσαι βέβαιος.\n"
    )

    full_prompt = (
        instructions
        + history_str
        + ("\n\nΠληροφορίες από έγγραφα:\n" + context_text if context_text else "")
        + f"\n\nΕρώτηση: {question}\n\nΑπάντηση:"
    )

    return full_prompt, sources


def query_model(prompt: str) -> str:
    """
    Call vLLM chat endpoint with token budgeting.
    """
    est_prompt_tokens = max(1, int(len(prompt) / 4))
    max_context = VLLM_MAX_CONTEXT
    safety = VLLM_SAFETY_MARGIN

    avail_for_completion = max_context - est_prompt_tokens - safety
    avail_for_completion = max(avail_for_completion, VLLM_MIN_COMPLETION)
    avail_for_completion = min(avail_for_completion, VLLM_MAX_COMPLETION_CAP)

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI tutor that answers in Greek."},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": avail_for_completion,
    }

    resp = requests.post(VLLM_URL, json=payload, timeout=VLLM_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"vLLM HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content.strip() or "Δεν μπόρεσα να παραγάγω απάντηση."


def answer_question(domain_id: str, question: str, chat_messages=None):
    """
    Main QA pipeline for a given domain.
    Keeps original behavior, including quote-based fallback when vLLM fails.
    """
    alt_queries = []
    if NUM_ALTERNATE_QUERIES > 0:
        alt_queries = [question.upper()] if NUM_ALTERNATE_QUERIES >= 1 else []

    retrieved = retrieve_chunks(domain_id, question, top_k=R_TOP_K, alt_queries=alt_queries)
    if not retrieved:
        if STRICT_MODE:
            return (
                "Δεν βρήκα σχετικές πληροφορίες στα έγγραφα αυτού του μαθήματος / έτους για να απαντήσω με σιγουριά.",
                [],
            )
        else:
            prompt = (
                f"{PROMPT_LANG}\n"
                "Δυστυχώς δεν βρήκα σχετικές πληροφορίες στα έγγραφα, αλλά προσπάθησε να απαντήσεις "
                "με βάση τις γενικές σου γνώσεις.\n\n"
                f"Ερώτηση: {question}\n\nΑπάντηση:"
            )
            try:
                ans = query_model(prompt)
            except Exception as e:
                print("⚠️ vLLM call failed in no-context path:", e)
                ans = "Δεν μπόρεσα να απαντήσω."
            return ans, []

    prompt, sources = build_prompt_from_chunks(question, retrieved, chat_messages=chat_messages)

    if EXTRACTIVE_ONLY:
        joined = "\n\n".join([c for (c, _, _) in retrieved])
        return joined, sources

    # Use quote-based fallback if vLLM fails
    quotes = auto_extract_quotes(retrieved)
    full_prompt = prompt
    try:
        final_answer = query_model(full_prompt)
    except Exception as e:
        print("⚠️ vLLM call failed; returning extractive answer:", e)
        lines_ = [ln.strip() for ln in quotes.splitlines() if ln.strip().startswith("- ")]
        final_answer = "Συνοπτικά:\n" + "\n".join(lines_[:4]) if lines_ else "Δεν ξέρω."

    return final_answer, sources


def _require_cache_admin():
    if not CACHE_ADMIN_KEY:
        abort(403, description="Cache admin key not configured")
    key = request.headers.get("x-cache-admin") or ""
    if key != CACHE_ADMIN_KEY:
        abort(403, description="Invalid cache admin key")


def _resolve_domain_folder(domain: str):
    if domain in domain_folders:
        return domain_folders[domain]
    d2 = next((k for k in domain_folders.keys() if k.lower() == (domain or "").lower()), None)
    if d2:
        return domain_folders[d2]
    return None


def load_teacher_answers_for_domain(domain: str):
    """
    Load and cache teacher answers for a given domain.

    Expected file name inside the domain folder:
      {domain}_TEACHER_ANSWERS.pdf

    Example:
      domain = "Courses__Psychology__YEAR_2__SEM_3__ΨΥΧΓΕ3"
      file   = "Courses__Psychology__YEAR_2__SEM_3__ΨΥΧΓΕ3_TEACHER_ANSWERS.pdf"
    """
    if not domain:
        return None

    # Cache hit
    if domain in teacher_answers_cache:
        return teacher_answers_cache[domain]

    folder = _resolve_domain_folder(domain)
    if not folder:
        teacher_answers_cache[domain] = None
        return None

    teacher_fname = f"{domain}_TEACHER_ANSWERS.pdf"
    teacher_path = os.path.join(folder, teacher_fname)
    if not os.path.isfile(teacher_path):
        teacher_answers_cache[domain] = None
        return None

    questions = []
    answers = []
    pages = []

    try:
        reader = PyPDF2.PdfReader(teacher_path)
    except Exception as e:
        print(f"⚠️ Could not read teacher PDF for domain {domain}: {e}")
        teacher_answers_cache[domain] = None
        return None

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if not text.strip():
            continue

        # Adjust these keys if your real PDF uses slightly different labels
        q_key = "Ερώτηση:"
        a_key = "Απάντηση καθηγητή:"
        d_key = "DocId:"

        q_idx = text.find(q_key)
        a_idx = text.find(a_key)

        if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
            continue

        q_text = text[q_idx + len(q_key):a_idx].strip()

        d_idx = text.find(d_key, a_idx)
        if d_idx != -1:
            a_text = text[a_idx + len(a_key):d_idx].strip()
        else:
            a_text = text[a_idx + len(a_key):].strip()

        if q_text and a_text:
            questions.append(q_text)
            answers.append(a_text)
            pages.append(i + 1)  # 1-based page index

    if not questions:
        teacher_answers_cache[domain] = None
        return None

    try:
        emb = embedding_model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Failed to embed teacher questions for domain {domain}: {e}")
        teacher_answers_cache[domain] = None
        return None

    teacher_answers_cache[domain] = {
        "questions": questions,
        "answers": answers,
        "embeddings": emb,
        "pages": pages,
        "file": teacher_fname,
    }
    print(f"📘 Loaded {len(questions)} teacher Q&A pairs for domain '{domain}'.")
    return teacher_answers_cache[domain]


def maybe_answer_from_teacher(domain: str, question: str):
    """
    Try to answer a question directly from the teacher answers PDF for this domain.
    Returns (answer_text, sources) or (None, None) if no good match.
    """
    if not question or not question.strip():
        return None, None

    info = load_teacher_answers_for_domain(domain)
    if not info:
        return None, None

    q_emb = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)[0]

    emb_mat = info.get("embeddings")
    if emb_mat is None or not len(emb_mat):
        return None, None

    # cosine similarity (embeddings are normalized)
    sims = emb_mat @ q_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < TEACHER_MIN_SIM:
        return None, None

    pdf_question = info["questions"][best_idx]
    pdf_answer = info["answers"][best_idx]

    # Include the teacher's question + answer in the response
    answer_text = (
        "Ερώτηση (καθηγητή): "
        + pdf_question.strip()
        + "\n\nΑπάντηση (καθηγητή): "
        + pdf_answer.strip()
    )

    page = info["pages"][best_idx] if info.get("pages") else None
    teacher_file = info.get("file")

    snippet = pdf_answer
    if len(snippet) > 240:
        snippet = snippet[:240] + "…"

    sources = [{
        "label": 1,
        "file": teacher_file,
        "page_start": page,
        "page_end": page,
        "snippet": snippet,
    }]

    return answer_text, sources


@app.route("/files/<domain>/<path:filename>", methods=["GET"])
def serve_file(domain, filename):
    if not filename.lower().endswith(".pdf"):
        abort(403)
    folder = _resolve_domain_folder(domain)
    if not folder or not os.path.isdir(folder):
        abort(404)
    safe_path = safe_join(folder, filename)
    if not safe_path or not os.path.isfile(safe_path):
        abort(404)
    return send_file(safe_path, as_attachment=False)


@app.route("/domains", methods=["GET"])
def list_domains():
    items = []
    for dom, folder in domain_folders.items():
        info = domain_data.get(dom) or {}
        items.append(
            {
                "domain": dom,
                "folder": folder,
                "kind": info.get("kind", "lesson"),
                "children": info.get("children", []),
            }
        )
    return jsonify(items)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/ask", methods=["POST"])
def ask():
    require_api_key()
    data = request.get_json(force=True) or {}

    question = (data.get("question") or "").strip()
    domain_in = (data.get("domain") or "").strip()
    scope = (data.get("scope") or "lesson").strip().lower()  # lesson | year
    chat_messages = data.get("messages") or []

    if not question:
        return jsonify({"answer": "Δεν δόθηκε ερώτηση.", "sources": []})

    # match domain case-insensitively
    matched_domain = next((d for d in domain_data.keys() if d.lower() == domain_in.lower()), domain_in)

    # scope escalation (supports BOTH lesson id and year-alias id)
    if scope == "year":
        # 1) if frontend sends "Courses__Course__YEAR_X", map to year aggregate
        alias_key = domain_in.replace(" ", "_")
        if alias_key in year_alias_to_year and year_alias_to_year[alias_key] in domain_data:
            matched_domain = year_alias_to_year[alias_key]
        else:
            # 2) if it's a lesson id, map to its year aggregate
            yd = lesson_to_year.get(matched_domain)
            if yd and yd in domain_data:
                matched_domain = yd
            else:
                # 3) if frontend already sent YEAR__..., keep it
                yd2 = next((d for d in domain_data.keys() if d.lower() == domain_in.lower()), None)
                if yd2 and domain_data.get(yd2, {}).get("kind") == "year":
                    matched_domain = yd2

    print(f"Question for domain [{matched_domain}] (scope={scope}): {question}")

    # 1) Try teacher answers first (if {domain}_TEACHER_ANSWERS.pdf exists)
    teacher_answer, teacher_sources = maybe_answer_from_teacher(matched_domain, question)
    if teacher_answer is not None:
        return jsonify({
            "answer": teacher_answer,
            "sources": teacher_sources or [],
            "resolvedDomain": matched_domain,
            "teacherAnswer": True,  # identifier for frontend styling
        })

    # 2) Fallback to existing retrieval + vLLM + quotes logic (unchanged)
    answer, sources = answer_question(matched_domain, question, chat_messages=chat_messages)

    return jsonify({"answer": answer, "sources": sources, "resolvedDomain": matched_domain})


@app.route("/cache/clear", methods=["POST"])
def cache_clear():
    _require_cache_admin()
    data = request.get_json(force=True, silent=True) or {}
    domain = data.get("domain")
    reload_after = bool(data.get("reload", False))

    summary = clear_index_cache(domain=domain, reload_after=reload_after)
    return jsonify(summary)


def clear_index_cache(domain=None, reload_after=False):
    """
    Clear index caches from disk and/or memory.
    If domain is None, clear all.
    """
    removed_files = 0

    if domain:
        pattern = f"{domain}::"
        for name in os.listdir(INDEX_CACHE_DIR):
            if not name:
                continue
            folder = os.path.join(INDEX_CACHE_DIR, name)
            meta_path = os.path.join(folder, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks = data.get("chunks") or []
                if not chunks:
                    # domain info not stored here; we stored only file-level keys
                    # but we used a hash based on domain+relpath, so we cannot
                    # easily filter by domain now. So for domain-specific clear,
                    # we simply clear all and re-index.
                    pass
            except Exception:
                pass

        # For domain-specific clear, easiest is to clear all caches and re-index as needed.
        for name in os.listdir(INDEX_CACHE_DIR):
            folder = os.path.join(INDEX_CACHE_DIR, name)
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                    removed_files += 1
                except Exception:
                    pass
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

        if domain in domain_data:
            domain_data.pop(domain, None)
        if domain in domain_folders:
            # we keep domain_folders so that domains are still discoverable
            pass

    else:
        # clear all
        for name in os.listdir(INDEX_CACHE_DIR):
            folder = os.path.join(INDEX_CACHE_DIR, name)
            if os.path.isdir(folder):
                try:
                    shutil.rmtree(folder)
                    removed_files += 1
                except Exception:
                    pass
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        domain_data.clear()
        # keep domain_folders mapping

    return {
        "domain": domain or "*",
        "removed_files": removed_files,
        "reloaded": bool(reload_after),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
