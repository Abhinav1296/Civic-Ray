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

# 0.  Environment & Global Configuration
load_dotenv()

# Application-Level Constants
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

# LLM Provider Configuration
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = "google/gemini-2.0-flash-exp:free"

GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

GREETING_TOKENS: set[str] = {
    "hi", "hello", "hey", "greetings", "sup", "yo",
    "good morning", "good evening", "hey buddy",
}

# 1.  LLM Client Factory

def build_openrouter_client() -> Optional[OpenAI]:
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


# 2.  Infrastructure Initialisation
def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
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

# 3.  LLM Call — Primary + Fallback
def call_llm_with_fallback(
    messages: list[dict[str, str]],
    temperature: float,
    primary_client: Optional[OpenAI],
    fallback_client: Optional[OpenAI],
) -> tuple[str, str]:
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


# 4.  RAG Core Logic
def optimise_query(
    raw_query: str,
    primary_client: Optional[OpenAI],
    fallback_client: Optional[OpenAI],
) -> str:
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
    matches: list[str] = re.findall(pattern, text, re.IGNORECASE)
    return sorted(set(matches))

# 5.  Flask Application & Routing
def create_app() -> Flask:
    flask_app = Flask(__name__)

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

    # Route: Home
    @flask_app.route("/")
    def home() -> str:
        """Serve the main chat interface HTML page.

        Returns:
            Rendered HTML string from ``templates/index.html``.
        """
        return render_template("index.html")

    # Route: Chat — Full RAG Pipeline Entry Point
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


# 6.  Entry Point
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False)