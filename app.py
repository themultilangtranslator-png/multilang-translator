from flask import Flask, request, jsonify
import os
import time
import json
import hashlib
from openai import OpenAI

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
DEFAULT_LANGS = ["en", "fr", "es", "it"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))  # 24h
CACHE_MAX_ITEMS = int(os.environ.get("CACHE_MAX_ITEMS", "2000"))

# Cache in-memory : {cache_key: (expires_at, payload_dict)}
CACHE = {}


def _now() -> float:
    return time.time()


def _make_cache_key(author: str, text: str, languages: list[str]) -> str:
    raw = json.dumps(
        {"author": author, "text": text, "languages": languages},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    item = CACHE.get(key)
    if not item:
        return None
    expires_at, payload = item
    if _now() >= expires_at:
        CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: dict):
    if CACHE_TTL_SECONDS <= 0:
        return
    if len(CACHE) >= CACHE_MAX_ITEMS:
        cutoff = int(CACHE_MAX_ITEMS * 0.1) or 1
        for k in list(CACHE.keys())[:cutoff]:
            CACHE.pop(k, None)
    CACHE[key] = (_now() + CACHE_TTL_SECONDS, payload)


def _build_line_text(author: str, original_text: str, detected_language: str, translations: dict, ordered_langs: list[str]) -> str:
    lines = []
    lines.append(f"Author: {author}")
    lines.append(f"Detected: {detected_language}")
    lines.append("")
    lines.append("Original:")
    lines.append(original_text)
    lines.append("")
    for lang in ordered_langs:
        lines.append(f"[{lang}]")
        lines.append(translations.get(lang, ""))
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_detected_language(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "english": "en",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "persian": "fa",
        "farsi": "fa",
    }
    if v in mapping:
        return mapping[v]
    if len(v) in (2, 3):
        return v
    return "unknown"


# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


# -------------------------
# TRANSLATE (POST)
# -------------------------
@app.route("/translate", methods=["POST"])
def translate():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    client = OpenAI(api_key=api_key)

    data = request.json or {}
    author = data.get("author", "Unknown")
    text = (data.get("text") or "").strip()
    languages = data.get("languages") or DEFAULT_LANGS
    include_line_text = bool(data.get("include_line_text", True))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not isinstance(languages, list) or not all(isinstance(x, str) for x in languages):
        return jsonify({"error": "languages must be a list of strings"}), 400

    ordered_langs = [x.strip().lower() for x in languages if x and x.strip()]
    if not ordered_langs:
        ordered_langs = DEFAULT_LANGS

    # -------------------------
    # CACHE
    # -------------------------
    cache_key = _make_cache_key(author, text, ordered_langs)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    # -------------------------
    # OPENAI PROMPT (GARDE-FOUS GÉNÉRIQUES & SCALABLES)
    # -------------------------
    system_prompt = {
        "role": "system",
        "content": (
            "You are a professional multilingual interpreter, not a literal translator. "
            "Your task is to convey meaning, intent, and tone accurately.\n\n"

            "Core principles:\n"
            "- Do NOT translate word-for-word.\n"
            "- Correct grammatical mistakes, typos, and awkward phrasing.\n"
            "- Adapt idioms, slang, and regional expressions naturally.\n"
            "- Use natural, idiomatic language in the target language.\n"
            "- Use proper capitalization and punctuation.\n\n"

            "Quality & meaning safeguards:\n"
            "- Always prioritize meaning and real-world plausibility over literal wording.\n"
            "- Infer the most likely intent and context before translating (chat, informal, operational, etc.).\n"
            "- If a word or phrase is ambiguous, choose the interpretation that best fits the inferred context.\n"
            "- Do not upgrade or downgrade formality unless explicitly indicated.\n"
            "- Do not change the scenario type (e.g., do not turn a casual action into a formal event).\n"
            "- Preserve tense and aspect (ongoing vs future; entering vs going).\n"
            "- Translate the meaning of idioms and slang, not the words.\n"
            "- If the source is poorly written, improve it minimally for clarity while preserving intent.\n"
            "- If ambiguity remains, resolve it using the most conservative, context-consistent interpretation.\n\n"

            "Fidelity control:\n"
            "- Never introduce new facts.\n"
            "- Never remove meaningful information.\n"
            "- If rephrasing is needed, keep the same intent and approximate length.\n\n"

            "Output rules:\n"
            "- Detect the source language automatically.\n"
            "- Return ONLY valid JSON.\n"
            "- No markdown, no quotes, no explanations.\n"
            "- Do not mention corrections or improvements.\n"
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            f"Target languages (in order): {ordered_langs}\n"
            f"Text:\n{text}\n\n"
            "Return this exact JSON schema:\n"
            "{\n"
            '  "detected_language": "<iso_code>",\n'
            '  "translations": {\n'
            '    "en": "...",\n'
            '    "fr": "...",\n'
            '    "es": "...",\n'
            '    "it": "..."\n'
            "  }\n"
            "}\n"
        )
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[system_prompt, user_prompt],
        )

        raw = (resp.choices[0].message.content or "").strip()
        result = json.loads(raw)

        detected_language = _normalize_detected_language(result.get("detected_language"))
        translations = result.get("translations", {}) or {}

        ordered_translations = {}
        for lang in ordered_langs:
            t = str(translations.get(lang, "")).strip()
            t = t.replace("```", "").replace("**", "").strip()
            ordered_translations[lang] = t

        payload = {
            "author": author,
            "original_text": text,
            "detected_language": detected_language,
            "translations": ordered_translations
        }

        if include_line_text:
            payload["line_text"] = _build_line_text(
                author, text, detected_language, ordered_translations, ordered_langs
            )

        _cache_set(cache_key, payload)
        return jsonify(payload), 200

    except json.JSONDecodeError:
        return jsonify({
            "error": "internal_error",
            "details": "OpenAI response was not valid JSON",
            "raw": raw[:500] if 'raw' in locals() else ""
        }), 500

    except Exception as e:
        return jsonify({
            "error": "internal_error",
            "details": str(e)
        }), 500


# -------------------------
# APP ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
