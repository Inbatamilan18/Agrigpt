"""
rag.py — Retrieval-Augmented Generation
========================================

1. Loads all .txt files from data/agriculture_docs
2. Splits them into ~800-character chunks
3. Embeds each chunk with a sentence-transformer model
4. Stores embeddings in Chroma
5. Retrieves the 4 most relevant chunks for each question
6. Sends the retrieved context to the Groq LLM
7. Returns ONLY the farmer-facing answer

Also contains optional NLLB-200 translation for Indian languages.
"""

import glob
import logging
import os
import re

from fastapi import HTTPException
from groq import Groq

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import Chroma

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


logger = logging.getLogger("agrigpt.rag")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen/qwen3.6-27b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
)

DOCS_DIR = os.getenv(
    "DOCS_DIR",
    os.path.join(
        _BASE_DIR,
        "data",
        "agriculture_docs",
    ),
)

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(
        _BASE_DIR,
        "chroma_db",
    ),
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
K_RETRIEVE = 4


# ============================================================================
# GLOBAL OBJECTS
# ============================================================================

_vectorstore = None
_client = None
_translator = None


# ============================================================================
# GROQ CLIENT
# ============================================================================

def _get_client() -> Groq:
    """Create and cache the Groq client."""

    global _client

    if _client is None:

        if (
            not GROQ_API_KEY
            or GROQ_API_KEY.startswith("paste-your")
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "GROQ_API_KEY is not set. "
                    "Copy .env.example to .env and add your "
                    "Groq API key."
                ),
            )

        _client = Groq(
            api_key=GROQ_API_KEY
        )

    return _client


# ============================================================================
# VECTOR DATABASE / RAG
# ============================================================================

def build_or_load_vectorstore() -> Chroma:
    """
    Build the Chroma vector database on first run.
    Load the existing database on subsequent runs.
    """

    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    files = glob.glob(
        os.path.join(
            DOCS_DIR,
            "**",
            "*.txt",
        ),
        recursive=True,
    )

    if not files:

        raise HTTPException(
            status_code=500,
            detail=(
                f"No knowledge documents found in "
                f"{DOCS_DIR}. Add .txt files there."
            ),
        )

    # ------------------------------------------------------------------------
    # Create vector database if it does not exist
    # ------------------------------------------------------------------------

    if not glob.glob(
        os.path.join(
            CHROMA_DIR,
            "*",
        )
    ):

        logger.info(
            "Embedding %d knowledge document(s)...",
            len(files),
        )

        loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        texts = splitter.split_documents(
            docs
        )

        _vectorstore = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=CHROMA_DIR,
        )

        logger.info(
            "Vector store built with %d chunks.",
            len(texts),
        )

    # ------------------------------------------------------------------------
    # Otherwise load existing vector database
    # ------------------------------------------------------------------------

    else:

        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    return _vectorstore


def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant agriculture knowledge chunks.
    """

    vs = build_or_load_vectorstore()

    docs = vs.similarity_search(
        query,
        k=K_RETRIEVE,
    )

    return "\n\n".join(
        d.page_content
        for d in docs
    )


# ============================================================================
# NLLB-200 TRANSLATION
# ============================================================================

NLLB_MODEL = (
    "facebook/nllb-200-distilled-600M"
)

NLLB_TGT_LANG = "eng_Latn"


LANGUAGE_PROMPTS = {
    "en": "English",
    "hi": "Hindi (Hinglish is fine)",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "bn": "Bengali",
    "pa": "Punjabi",
}


# Unicode script ranges -> NLLB source language
_SCRIPT_MAP = [
    (0x0900, 0x097F, "hin_Deva"),
    (0x0980, 0x09FF, "ben_Beng"),
    (0x0A00, 0x0A7F, "pan_Guru"),
    (0x0B80, 0x0BFF, "tam_Taml"),
    (0x0C00, 0x0C7F, "tel_Telu"),
    (0x0C80, 0x0CFF, "kan_Knda"),
    (0x0D00, 0x0D7F, "mal_Mlym"),
]


def _get_translator():
    """
    Load NLLB-200 lazily.

    It is only loaded when an Indian-language query
    needs translation.
    """

    global _translator

    if _translator is None:

        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        logger.info(
            "Loading NLLB-200 translator..."
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                NLLB_MODEL
            )
        )

        model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL
            )
        )

        model.eval()

        _translator = {
            "model": model,
            "tokenizer": tokenizer,
            "tgt_id": tokenizer.convert_tokens_to_ids(
                NLLB_TGT_LANG
            ),
        }

    return _translator


def _detect_source_code(text: str):
    """
    Detect the Indian script used by the query.
    """

    for lo, hi, code in _SCRIPT_MAP:

        if any(
            lo <= ord(ch) <= hi
            for ch in text
        ):
            return code

    return None


def translate_to_english(text: str) -> str:
    """
    Translate supported Indian-language queries to English.

    English passes through unchanged.
    If translation fails, the original text is returned.
    """

    src_code = _detect_source_code(
        text
    )

    if not src_code:
        return text

    try:

        import torch

        translator = _get_translator()

        model = translator["model"]
        tokenizer = translator["tokenizer"]

        src_id = (
            tokenizer.convert_tokens_to_ids(
                src_code
            )
        )

        enc = tokenizer(
            text,
            add_special_tokens=False,
        )

        input_ids = torch.tensor(
            [[src_id] + enc.input_ids],
            dtype=torch.long,
        )

        with torch.no_grad():

            out = model.generate(
                input_ids=input_ids,
                forced_bos_token_id=translator[
                    "tgt_id"
                ],
                max_new_tokens=512,
            )

        return tokenizer.decode(
            out[0],
            skip_special_tokens=True,
        ).strip()

    except Exception as exc:

        logger.warning(
            "Translation unavailable (%s). "
            "Using original text.",
            exc,
        )

        return text


# ============================================================================
# CLEAN LLM OUTPUT
# ============================================================================

def _clean_llm_output(content: str) -> str:
    """
    Clean the LLM response.

    The API is configured to hide reasoning, but this
    function is an additional safety layer in case the
    model ever returns <think> tags.
    """

    if not content:
        return ""

    content = content.strip()

    # ------------------------------------------------------------------------
    # Remove complete <think>...</think> blocks
    # ------------------------------------------------------------------------

    content = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        content,
    )

    # ------------------------------------------------------------------------
    # Remove complete <thinking>...</thinking> blocks
    # ------------------------------------------------------------------------

    content = re.sub(
        r"(?is)<thinking>.*?</thinking>",
        "",
        content,
    )

    # ------------------------------------------------------------------------
    # Handle an unclosed <think> block.
    #
    # Example:
    #
    # <think>
    # internal reasoning...
    #
    # ------------------------------------------------------------------------

    content = re.sub(
        r"(?is)<think>.*$",
        "",
        content,
    )

    # ------------------------------------------------------------------------
    # Handle an unclosed <thinking> block.
    # ------------------------------------------------------------------------

    content = re.sub(
        r"(?is)<thinking>.*$",
        "",
        content,
    )

    # ------------------------------------------------------------------------
    # Remove leftover tags.
    # ------------------------------------------------------------------------

    content = re.sub(
        r"(?is)</?think>",
        "",
        content,
    )

    content = re.sub(
        r"(?is)</?thinking>",
        "",
        content,
    )

    return content.strip()


# ============================================================================
# LLM ANSWER GENERATION
# ============================================================================

def generate_advice(
    user_query: str,
    disease: str,
    language: str,
    history: str,
    profile: str = "",
) -> str:
    """
    Complete RAG pipeline:

    User question
        ↓
    Retrieve relevant agriculture documents
        ↓
    Build grounded prompt
        ↓
    Groq Qwen 3.6 27B
        ↓
    Non-thinking mode
        ↓
    Clean final answer
        ↓
    Return answer to frontend
    """

    # ------------------------------------------------------------------------
    # Retrieve relevant agriculture context
    # ------------------------------------------------------------------------

    context = retrieve_context(
        f"{user_query} "
        f"{disease} "
        f"treatment "
        f"control "
        f"management"
    )

    # ------------------------------------------------------------------------
    # Language instruction
    # ------------------------------------------------------------------------

    lang_instruction = (
        "Answer in clear, simple "
        f"{LANGUAGE_PROMPTS.get(language, 'English')}."
    )

    # ------------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------------

    system_prompt = f"""
You are AgriGPT, a trusted Indian agriculture advisor.

Your job is to give the farmer a direct, useful answer.

RULES:

1. Base your answer primarily on the CONTEXT provided below.

2. If the CONTEXT does not contain the exact answer,
   honestly say that the provided reference material does
   not specifically cover the issue.

3. When the context does not cover the issue, you may give
   safe general agricultural guidance, but do not invent
   exact pesticide doses, fertilizer doses, chemical rates,
   or product recommendations.

4. Answer the farmer's actual question directly.

5. Use short sections when useful:
   - What is happening
   - Immediate action
   - Prevention

6. Keep the answer below 250 words.

7. Use simple language suitable for a farmer.

8. Do NOT explain your reasoning.

9. Do NOT describe your analysis.

10. Do NOT say things like:
    "I analyzed your question"
    "Let's analyze"
    "My reasoning is"
    "Step 1: Analyze"
    "I need to check the context"

11. Do NOT output <think> tags.

12. Do NOT output <thinking> tags.

13. Output ONLY the final farmer-facing answer.

{lang_instruction}

{profile}

CONTEXT:
{context}
"""

    # ------------------------------------------------------------------------
    # Add conversation history
    # ------------------------------------------------------------------------

    if history:

        system_prompt += (
            "\n\nCONVERSATION SO FAR:\n"
            f"{history}"
        )

    # ------------------------------------------------------------------------
    # User message
    # ------------------------------------------------------------------------

    user_content = (
        "Detected disease from crop image: "
        f"{disease or 'none (no image provided)'}\n\n"
        f"Farmer's question: {user_query}"
    )

    # ------------------------------------------------------------------------
    # Groq client
    # ------------------------------------------------------------------------

    client = _get_client()

    try:

        # ====================================================================
        # IMPORTANT:
        #
        # reasoning_effort="none"
        #
        # tells Qwen not to spend tokens on reasoning.
        #
        # reasoning_format="hidden"
        #
        # makes sure reasoning is not returned to the frontend.
        # ====================================================================

        response = client.chat.completions.create(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],

            # Disable Qwen thinking/reasoning.
            reasoning_effort="none",

            # Return only the final answer.
            reasoning_format="hidden",

            # Recommended non-thinking temperature.
            temperature=0.7,

            # Give the final answer enough room.
            max_completion_tokens=800,

            stream=False,
        )

    except Exception as exc:

        logger.error(
            "Groq API error: %s",
            exc,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM (Groq) error: {exc} "
                "— check the LLM_MODEL name in .env"
            ),
        )

    # ------------------------------------------------------------------------
    # Get model content
    # ------------------------------------------------------------------------

    content = (
        response.choices[0].message.content
        or ""
    )

    logger.info(
        "Received LLM response: %d characters.",
        len(content),
    )

    # ------------------------------------------------------------------------
    # Remove any accidental reasoning tags
    # ------------------------------------------------------------------------

    cleaned_answer = _clean_llm_output(
        content
    )

    # ------------------------------------------------------------------------
    # Empty response protection
    # ------------------------------------------------------------------------

    if not cleaned_answer:

        logger.warning(
            "LLM returned an empty final answer."
        )

        return (
            "I could not generate a clear answer "
            "right now. Please try asking the question again."
        )

    return cleaned_answer