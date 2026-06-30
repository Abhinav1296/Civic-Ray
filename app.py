import os
import re
import time
import traceback
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# ---------------------------------------------------------------------------
# 0.  Environment & Global Configuration
# ---------------------------------------------------------------------------
# Load all secrets from the project-root .env file before reading any
# os.getenv() calls.  This must happen before the constants below.
load_dotenv()

# ---------------------------------------------------------------------------
# Application-Level Constants
# ---------------------------------------------------------------------------

DB_PATH: str = "vector_db"
"""Filesystem path to the pre-built FAISS index directory."""

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
"""
HuggingFace model ID for the dense embedding encoder.

'all-mpnet-base-v2' is trained with a contrastive objective (multiple
negatives ranking loss) on 1-billion sentence pairs, producing 768-
dimensional unit-normalised vectors. Cosine similarity in this space
correlates strongly with semantic relatedness.
"""

TOP_K_DOCUMENTS: int = 3
"""
Number of document chunks to retrieve from FAISS per query.
Increasing k raises recall but also increases noise in the context window.
"""

GENERATION_TEMPERATURE: float = 0.1
"""
Sampling temperature for the generation call.
A low value (→ 0) makes outputs near-deterministic and factual,
which is desirable for legal Q&A where precision outweighs creativity.
"""

# ---------------------------------------------------------------------------
# LLM Provider Configuration
# ---------------------------------------------------------------------------
# Both OpenRouter and Groq expose OpenAI-compatible endpoints.
# We configure each as a named tuple of (base_url, api_key, model_id).
# This makes the fallback logic completely explicit and easy to read.

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = "google/gemini-2.0-flash-exp:free"
"""
Google Gemini 2.0 Flash (Experimental) via OpenRouter free tier.

Why this model?
- 1M token context window handles large retrieved legal documents.
- Genuinely free with no per-query cost on OpenRouter free tier.
- Strong instruction-following for structured legal prompts.
- Fast inference latency (~2-4s for typical legal queries).
"""

GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
"""
Meta LLaMA 3.3 70B Versatile via Groq free tier.

Why this model?
- 70 billion parameters: excellent legal reasoning capability.
- 32k token context window: fits all retrieved bye-law chunks.
- Groq's LPU hardware delivers ~300 tokens/sec — fastest free option.
- 'versatile' variant is tuned for instruction-following and Q&A.
"""

GREETING_TOKENS: set[str] = {
    "hi", "hello", "hey", "greetings", "sup", "yo",
    "good morning", "good evening", "hey buddy",
}
"""Set of normalised greeting strings that short-circuit the RAG pipeline."""


# ---------------------------------------------------------------------------
# 1.  LLM Client Factory
# ---------------------------------------------------------------------------

def build_openrouter_client() -> Optional[OpenAI]:
    """Construct an OpenAI-compatible client pointed at OpenRouter.

    OpenRouter aggregates hundreds of LLM models behind a single
    OpenAI-compatible API. By setting ``base_url`` to OpenRouter's
    endpoint and providing an OpenRouter API key, the standard
    ``openai.OpenAI`` client works without any modification.

    The ``HTTP-Referer`` and ``X-Title`` default headers are optional
    but recommended by OpenRouter for usage tracking in their dashboard.
    They do not affect billing or rate limits.

    Returns:
        A configured :class:`openai.OpenAI` client for OpenRouter,
        or ``None`` if the API key is missing from the environment.
    """
    if not OPENROUTER_API_KEY:
        print("[INIT] OPENROUTER_API_KEY not found in environment. "
              "OpenRouter will not be available.")
        return None

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            # Recommended by OpenRouter for dashboard attribution.
            # Change to your actual project URL / name.
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Civic Ray — Municipal Law Assistant",
        },
    )
    print(f"[INIT] OpenRouter client ready  → model: {OPENROUTER_MODEL}")
    return client


def build_groq_client() -> Optional[OpenAI]:
    """Construct an OpenAI-compatible client pointed at Groq.

    Groq's API is fully OpenAI-compatible. Pointing ``base_url`` at
    Groq's endpoint converts the standard client into a Groq client.
    No Groq-specific SDK is required.

    Returns:
        A configured :class:`openai.OpenAI` client for Groq,
        or ``None`` if the API key is missing from the environment.
    """
    if not GROQ_API_KEY:
        print("[INIT] GROQ_API_KEY not found in environment. "
              "Groq fallback will not be available.")
        return None

    client = OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
    )
    print(f"[INIT] Groq client ready        → model: {GROQ_MODEL}")
    return client


# ---------------------------------------------------------------------------
# 2.  Infrastructure Initialisation
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and return a HuggingFace sentence-embedding model.

    The embedding model converts raw text into a fixed-length dense vector
    so that semantic similarity can be measured via cosine distance.
    This function is intentionally separated from the FAISS loader so that
    each component can be tested or swapped independently.

    Args:
        model_name: A valid HuggingFace Hub model identifier string,
                    e.g. ``"sentence-transformers/all-mpnet-base-v2"``.

    Returns:
        A :class:`HuggingFaceEmbeddings` instance ready for encoding.

    Raises:
        RuntimeError: If the model cannot be downloaded or initialised.
                      The full traceback is printed so the developer can
                      diagnose network or library issues immediately.
    """
    print(f"[INIT] Loading embedding model: '{model_name}' ...")
    try:
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        print("[INIT] Embedding model loaded successfully.")
        return embedder
    except Exception:
        print("[ERROR] Failed to load the embedding model.")
        traceback.print_exc()
        raise RuntimeError(
            f"Could not initialise HuggingFaceEmbeddings "
            f"with model '{model_name}'."
        )


def load_vector_database(
    db_path: str,
    embedder: HuggingFaceEmbeddings,
) -> Optional[FAISS]:
    """Load a persisted FAISS vector index from disk.

    FAISS stores document chunk embeddings in an optimised binary index
    (typically an IVF or Flat index). At query time, the query vector is
    compared against all stored vectors using inner-product or L2 distance,
    returning the top-k nearest neighbours in sub-linear time for large
    corpora.

    Args:
        db_path: Path to the directory containing ``index.faiss``
                 and ``index.pkl``.
        embedder: The same embedding model used when the index was built.
                  Mismatched models produce meaningless similarity scores.

    Returns:
        A loaded :class:`FAISS` vector store, or ``None`` if loading
        fails. Returning ``None`` lets the Flask server start and return
        a graceful degraded response instead of crashing.

    Note:
        ``allow_dangerous_deserialization=True`` is required by LangChain
        because ``index.pkl`` is loaded with ``pickle``. Only set this
        flag for indexes you have built yourself from trusted sources.
    """
    print(f"[INIT] Loading FAISS vector database from '{db_path}' ...")
    try:
        vector_db = FAISS.load_local(
            db_path,
            embedder,
            allow_dangerous_deserialization=True,
        )
        print("[INIT] FAISS database loaded successfully.")
        return vector_db
    except Exception:
        print("[ERROR] Failed to load the FAISS database. "
              "Retrieval will be skipped for all queries.")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 3.  LLM Call — Primary + Fallback
# ---------------------------------------------------------------------------

def call_llm_with_fallback(
    messages: list[dict[str, str]],
    temperature: float,
    primary_client: Optional[OpenAI],
    fallback_client: Optional[OpenAI],
) -> tuple[str, str]:
    """Attempt the primary LLM provider; fall back to secondary on failure.

    Provider Fallback Design:
        This function implements a simple linear fallback chain:

            OpenRouter (Gemini 2.0 Flash) → Groq (LLaMA 3.3 70B)

        Each provider is tried sequentially. The first successful
        response is returned immediately. If both fail, a descriptive
        error string is returned so the Flask route can surface it
        to the user without crashing.

        This approach is preferred over silent retry loops because:
        (a) it is completely transparent in the console logs,
        (b) it avoids burning free-tier quota on repeated retries,
        (c) the exact failure point is always visible in the traceback.

    Args:
        messages:         OpenAI-format message list (system + user turns).
        temperature:      Sampling temperature passed to both providers.
        primary_client:   Configured OpenRouter client (may be ``None``
                          if the key was absent at startup).
        fallback_client:  Configured Groq client (may be ``None``
                          if the key was absent at startup).

    Returns:
        A two-tuple ``(response_text, provider_label)`` where:

        - ``response_text``: The model's reply as a plain string.
        - ``provider_label``: Human-readable label for the UI,
          e.g. ``"OpenRouter — gemini-2.0-flash-exp:free"``.

        Returns an error string + ``"None"`` label if both providers fail.
    """
    # ── Attempt 1: OpenRouter (Primary) ─────────────────────────────────
    if primary_client is not None:
        try:
            print(f"[LLM] Trying primary: OpenRouter ({OPENROUTER_MODEL}) ...")
            completion = primary_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
            content: str = completion.choices[0].message.content or ""
            print("[LLM] OpenRouter responded successfully.")
            return content, f"OpenRouter — {OPENROUTER_MODEL}"

        except Exception:
            # Print the full traceback so the developer can see exactly
            # why OpenRouter failed (auth error, rate limit, timeout, etc.)
            print("[WARNING] OpenRouter call failed. Attempting Groq fallback ...")
            traceback.print_exc()

    # ── Attempt 2: Groq (Fallback) ───────────────────────────────────────
    if fallback_client is not None:
        try:
            print(f"[LLM] Trying fallback: Groq ({GROQ_MODEL}) ...")
            completion = fallback_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
            content = completion.choices[0].message.content or ""
            print("[LLM] Groq responded successfully.")
            return content, f"Groq — {GROQ_MODEL}"

        except Exception:
            print("[ERROR] Groq fallback also failed.")
            traceback.print_exc()

    # ── Both Providers Failed ────────────────────────────────────────────
    error_message: str = (
        "⚠️ Both LLM providers are currently unavailable.\n\n"
        "Please check:\n"
        "  1. OPENROUTER_API_KEY is set correctly in your .env file.\n"
        "  2. GROQ_API_KEY is set correctly in your .env file.\n"
        "  3. Your network connection is active.\n\n"
        "Full error details have been printed to the server console."
    )
    print("[ERROR] All LLM providers exhausted. Returning error to client.")
    return error_message, "None"


# ---------------------------------------------------------------------------
# 4.  RAG Core Logic
# ---------------------------------------------------------------------------

def optimise_query(
    raw_query: str,
    primary_client: Optional[OpenAI],
    fallback_client: Optional[OpenAI],
) -> str:
    """Expand a raw user query with domain-specific legal keywords.

    Motivation (Query Expansion):
        A user may write "my neighbour built without permission", but the
        municipal bye-law corpus uses terminology like "unauthorised
        construction", "building permit violation", or "Section 52 notice".
        A vanilla embedding of colloquial words may not surface the most
        relevant chunks because the embedding space was shaped by formal
        legal text during indexing.

        By asking the LLM to translate the query into expert vocabulary
        *before* embedding, we improve retrieval recall without any
        changes to the FAISS index itself. This technique is known as
        HyDE (Hypothetical Document Embeddings) in its stronger form;
        here we use a lighter keyword-expansion variant.

    Args:
        raw_query:       The unmodified query string entered by the user.
        primary_client:  OpenRouter client (may be ``None``).
        fallback_client: Groq client (may be ``None``).

    Returns:
        A single enriched string: ``"<raw_query> <kw1> <kw2> ..."``.
        Falls back to ``raw_query`` unchanged if both LLM providers fail,
        ensuring the pipeline continues with reduced retrieval quality
        rather than aborting entirely.
    """
    optimisation_prompt: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Legal Search Optimiser specialised in Indian "
                "municipal law. Given the user's natural-language query, "
                "return ONLY 3–5 space-separated technical keywords drawn "
                "from municipal bye-laws. "
                "Do not include punctuation, explanations, or full sentences."
            ),
        },
        {
            "role": "user",
            "content": raw_query,
        },
    ]

    try:
        keywords, _ = call_llm_with_fallback(
            messages=optimisation_prompt,
            temperature=0.1,
            primary_client=primary_client,
            fallback_client=fallback_client,
        )

        # Guard: if both providers returned an error string, the keywords
        # will start with "⚠️" — in that case skip the expansion.
        if keywords.startswith("⚠️"):
            print("[WARNING] Query optimisation skipped (provider error).")
            return raw_query

        expanded_query: str = f"{raw_query} {keywords.strip()}"
        print(f"[RAG] Expanded query: '{expanded_query}'")
        return expanded_query

    except Exception:
        # Defensive catch-all: optimisation failure is non-fatal.
        print("[WARNING] Query optimisation raised an unexpected error. "
              "Falling back to raw query.")
        traceback.print_exc()
        return raw_query


def retrieve_context(
    query: str,
    vector_db: FAISS,
    top_k: int = TOP_K_DOCUMENTS,
) -> tuple[str, list[str]]:
    """Retrieve the most semantically relevant document chunks from FAISS.

    How Similarity Search Works:
        1. The ``query`` string is encoded by the same embedding model
           used to build the index, producing a 768-dim query vector.
        2. FAISS computes cosine similarity (or L2 distance) between the
           query vector and every stored chunk vector.
        3. The ``top_k`` chunks with the highest similarity scores are
           returned as :class:`langchain.schema.Document` objects.

    Grounding Principle:
        The concatenated chunks form the *context window* injected into
        the generation prompt. By restricting the LLM to only this
        context, we prevent it from drawing on parametric (trained)
        knowledge that may be outdated, jurisdiction-specific, or
        hallucinated. This is the defining property of RAG.

    Args:
        query:     The expanded query string from :func:`optimise_query`.
        vector_db: A loaded FAISS vector store instance.
        top_k:     Maximum number of document chunks to retrieve.

    Returns:
        A two-tuple:

        - **context_text** (``str``): Retrieved chunks concatenated with
          page-number annotations for prompt injection.
        - **sources** (``list[str]``): Deduplicated, sorted source labels
          for UI citation (e.g. ``["Page 17", "Page 23"]``).
    """
    raw_results = vector_db.similarity_search(query, k=top_k)

    context_parts: list[str] = []
    source_labels: list[str] = []

    for document in raw_results:
        page_number: str = str(document.metadata.get("page", "unknown"))
        source_label: str = f"Page {page_number}"

        context_parts.append(
            f"\n--- [Source: {source_label}] ---\n"
            f"{document.page_content}\n"
        )
        source_labels.append(source_label)

    context_text: str = "".join(context_parts)
    deduplicated_sources: list[str] = sorted(set(source_labels))

    print(f"[RAG] Retrieved {len(raw_results)} chunks | "
          f"sources: {deduplicated_sources}")

    return context_text, deduplicated_sources


def build_generation_prompt(
    user_query: str,
    retrieved_context: str,
    mode: str,
) -> list[dict[str, str]]:
    """Construct the structured prompt that bounds LLM generation to retrieved facts.

    Prompt Engineering Rationale:
        The system message acts as a *constraint boundary* for the LLM.
        Injecting the retrieved context and instructing the model to answer
        *only* from that context is the foundational RAG principle: the
        language model provides reasoning and fluent generation; the
        retriever provides factual grounding.

        Two prompt variants handle different user personas:

        - **citizen mode**: Plain-language explanations accessible to
          members of the public. The model is explicitly told not to
          invent fine amounts absent from the context.
        - **lawyer mode**: Formal legal drafting. The model produces a
          structured risk assessment and a formal municipal notice with
          mandatory section references.

    Args:
        user_query:        The original, un-expanded user query.
        retrieved_context: Concatenated FAISS chunks with page annotations.
        mode:              ``"citizen"`` or ``"lawyer"``.

    Returns:
        A list of OpenAI-format message dicts ready for the API call.
    """
    if mode == "lawyer":
        system_instruction: str = (
            "You are an expert Municipal Legal Drafting Assistant.\n\n"
            "Structure your response into exactly two sections. You MUST use these "
            "exact headers verbatim, with no variations:\n\n"
            "⚠️ RISK ASSESSMENT\n"
            "Determine risk severity (Low / Medium / High) based strictly "
            "on the provided context. Cite the relevant section numbers.\n\n"
            "📝 DRAFT NOTICE\n"
            "Write a formal legal notification addressed to the Municipal "
            "Commissioner. Use professional legal language and cite the "
            "relevant bye-law sections verbatim.\n\n"
            "STRICT RULE: Do NOT include any information not present in "
            "the CONTEXT below."
        )
    else:
        system_instruction = (
            "You are Civic Ray, a precise and helpful Municipal Law Assistant. "
            "Explain the relevant bye-laws in clear, simple language using ONLY "
            "the provided CONTEXT. "
            "If an exact fine amount is not stated in the context, describe "
            "the applicable enforcement action instead. "
            "Do NOT invent or estimate fine amounts."
        )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"{system_instruction}\n\n"
                f"CONTEXT FROM BYE-LAWS:\n"
                f"{retrieved_context}"
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]
    return messages


def parse_lawyer_response(raw_response: str) -> tuple[str, str]:
    """Split a lawyer-mode LLM response into risk section and draft notice.

    The generation prompt instructs the model to use emoji markers
    ``"⚠️ RISK ASSESSMENT"`` and ``"📝 DRAFT NOTICE"`` as structural
    delimiters. This function extracts each section so the front-end
    can render them in separate UI panels.

    Args:
        raw_response: The complete LLM reply string for a lawyer-mode request.

    Returns:
        A two-tuple ``(risk_section, draft_section)``. If markers are
        absent (model deviated from prompt), the entire response is
        returned in ``draft_section`` so no content is silently lost.
    """
    if "📝 DRAFT NOTICE" not in raw_response:
        print("[WARNING] Lawyer-mode response missing expected section markers. "
              "Returning full response in draft_section.")
        return "", raw_response

    parts: list[str] = raw_response.split("📝 DRAFT NOTICE", maxsplit=1)
    risk_section: str = (
        parts[0].replace("⚠️ RISK ASSESSMENT", "").strip()
    )
    draft_section: str = (
        "📝 DRAFT NOTICE" + parts[1] if len(parts) > 1 else ""
    )
    return risk_section, draft_section


def extract_metadata_patterns(text: str, pattern: str) -> list[str]:
    """Find all unique regex matches in *text* and return them sorted.

    Used to surface fine amounts and section references from the
    free-text LLM response as discrete structured fields for the UI.
    Presenting these separately improves scannability for both citizens
    (who want to know the fine) and lawyers (who need section numbers).

    Args:
        text:    The string to search (typically the full LLM response).
        pattern: A compiled-compatible regex pattern string.

    Returns:
        A sorted list of unique matched strings, or ``[]`` if none found.
    """
    matches: list[str] = re.findall(pattern, text, re.IGNORECASE)
    return sorted(set(matches))


# ---------------------------------------------------------------------------
# 5.  Flask Application & Routing
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Construct, configure, and return the Flask application instance.

    Using an application factory pattern (rather than a module-level
    ``app = Flask(__name__)``) is a Flask best practice for testability:
    each test suite call to ``create_app()`` gets a fresh, isolated instance
    with no shared state between tests.

    Startup sequence:
        1. Load the HuggingFace embedding model.
        2. Load the FAISS vector index from disk.
        3. Construct the OpenRouter (primary) LLM client.
        4. Construct the Groq (fallback) LLM client.
        5. Register all Flask routes.

    Returns:
        A fully configured :class:`flask.Flask` application instance.
    """
    flask_app = Flask(__name__)

    # ------------------------------------------------------------------
    # Initialise all shared resources once at server startup.
    # Storing them in flask_app.config avoids module-level globals and
    # makes the dependency chain explicit and testable.
    # ------------------------------------------------------------------
    flask_app.config["EMBEDDER"] = load_embedding_model(EMBEDDING_MODEL_NAME)

    flask_app.config["VECTOR_DB"] = load_vector_database(
        DB_PATH,
        flask_app.config["EMBEDDER"],
    )

    # Build both LLM clients. Either may be None if its key is missing.
    flask_app.config["PRIMARY_CLIENT"] = build_openrouter_client()
    flask_app.config["FALLBACK_CLIENT"] = build_groq_client()

    # Safety check: warn loudly if BOTH providers are unconfigured.
    if (flask_app.config["PRIMARY_CLIENT"] is None
            and flask_app.config["FALLBACK_CLIENT"] is None):
        print(
            "\n" + "=" * 60 + "\n"
            "[CRITICAL WARNING] No LLM provider is configured!\n"
            "Set at least one of the following in your .env file:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  GROQ_API_KEY=gsk_...\n"
            "The server will start but ALL chat queries will fail.\n"
            + "=" * 60 + "\n"
        )

    # ──────────────────────────────────────────────────────────────────
    # Route: Home
    # ──────────────────────────────────────────────────────────────────
    @flask_app.route("/")
    def home() -> str:
        """Serve the main chat interface HTML page.

        Returns:
            Rendered HTML string from ``templates/index.html``.
        """
        return render_template("index.html")

    # ──────────────────────────────────────────────────────────────────
    # Route: Chat — Full RAG Pipeline Entry Point
    # ──────────────────────────────────────────────────────────────────
    @flask_app.route("/chat", methods=["POST"])
    def chat() -> tuple[Response, int] | Response:
        """Handle a user query through the full three-stage RAG pipeline.

        Request Body (JSON):
            message (str): The user's natural-language question.
            mode    (str): ``"citizen"`` (default) or ``"lawyer"``.

        Pipeline Execution Order:
            1. Input validation — reject empty/missing messages early.
            2. Greeting detection — skip the full pipeline for greetings.
            3. Query optimisation — enrich the query with legal keywords.
            4. Dense retrieval — FAISS ANN search for top-k chunks.
            5. Grounded generation — LLM call bounded to retrieved context.
            6. Response parsing — split risk/draft sections (lawyer mode).
            7. Metadata extraction — surface fines and section references.

        Returns:
            JSON response (HTTP 200) with keys:
                answer, risk_section, draft_section, sources,
                fines, sections, mode, engine_used, time.

            JSON error (HTTP 400) if message is empty.
            JSON error (HTTP 500) if both LLM providers fail.
        """
        request_start_time: float = time.time()

        # ── Stage 1: Input Validation ────────────────────────────────
        request_data: dict = request.get_json() or {}
        user_query: str = request_data.get("message", "").strip()
        mode: str = request_data.get("mode", "citizen").strip().lower()

        if not user_query:
            return jsonify({"error": "Message cannot be empty."}), 400

        # ── Stage 2: Greeting Short-Circuit ─────────────────────────
        # Normalise to a bare token and check against the known greeting
        # set. This avoids wasting free-tier LLM quota on non-queries.
        normalised_query: str = (
            user_query.lower().strip().replace(".", "").replace("!", "")
        )
        if normalised_query in GREETING_TOKENS:
            return jsonify({
                "answer": (
                    "Hello! I am Civic Ray, your Municipal Law Assistant. "
                    "Please describe a specific municipal issue or violation "
                    "and I will retrieve the relevant bye-law for you."
                ),
                "risk_section":  "",
                "draft_section": "",
                "sources":       [],
                "fines":         [],
                "sections":      [],
                "mode":          mode,
                "engine_used":   "N/A — greeting shortcut",
                "time":          round(time.time() - request_start_time, 3),
            })

        # Pull shared resources from app config (set at startup).
        vector_db: Optional[FAISS] = flask_app.config["VECTOR_DB"]
        primary_client: Optional[OpenAI] = flask_app.config["PRIMARY_CLIENT"]
        fallback_client: Optional[OpenAI] = flask_app.config["FALLBACK_CLIENT"]

        # ── Stage 3: Query Optimisation ──────────────────────────────
        expanded_query: str = optimise_query(
            raw_query=user_query,
            primary_client=primary_client,
            fallback_client=fallback_client,
        )

        # ── Stage 4: Dense Retrieval ─────────────────────────────────
        retrieved_context: str = ""
        sources: list[str] = []

        if vector_db is not None:
            try:
                retrieved_context, sources = retrieve_context(
                    query=expanded_query,
                    vector_db=vector_db,
                    top_k=TOP_K_DOCUMENTS,
                )
            except Exception:
                # Retrieval failure is logged but non-fatal: the LLM
                # will still generate a (less grounded) response.
                print("[ERROR] FAISS retrieval failed. "
                      "Proceeding with empty context.")
                traceback.print_exc()
        else:
            print("[WARNING] Vector database unavailable. "
                  "Skipping retrieval stage.")

        # ── Stage 5: Grounded Generation ────────────────────────────
        generation_messages: list[dict[str, str]] = build_generation_prompt(
            user_query=user_query,
            retrieved_context=retrieved_context,
            mode=mode,
        )

        raw_answer, engine_used = call_llm_with_fallback(
            messages=generation_messages,
            temperature=GENERATION_TEMPERATURE,
            primary_client=primary_client,
            fallback_client=fallback_client,
        )

        # Detect the all-providers-failed sentinel string and return
        # a proper HTTP 500 instead of a fake "answer".
        if engine_used == "None":
            return jsonify({
                "error": raw_answer  # contains the user-friendly error msg
            }), 500

        # ── Stage 6: Response Parsing ────────────────────────────────
        risk_section: str = ""
        draft_section: str = ""

        if mode == "lawyer":
            risk_section, draft_section = parse_lawyer_response(raw_answer)

        # ── Stage 7: Metadata Extraction ─────────────────────────────
        fines: list[str] = extract_metadata_patterns(
            raw_answer,
            r"(?:Rs\.?|₹|INR)\s?[\d,]+",
        )
        sections: list[str] = extract_metadata_patterns(
            raw_answer,
            r"(?:section|sec\.?|clause|article)\s*[\d\w\-\.]+",
        )

        total_time: float = round(time.time() - request_start_time, 3)
        print(
            f"[RAG] ✓ Pipeline complete | {total_time}s | "
            f"mode={mode} | engine={engine_used} | sources={sources}"
        )

        return jsonify({
            "answer":        raw_answer,
            "risk_section":  risk_section,
            "draft_section": draft_section,
            "sources":       sources,
            "fines":         fines,
            "sections":      sections,
            "mode":          mode,
            "engine_used":   engine_used,
            "time":          total_time,
        })

    return flask_app


# ---------------------------------------------------------------------------
# 6.  Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False)