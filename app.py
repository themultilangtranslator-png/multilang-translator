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

# Simple cache in-memory : {cache_key: (expires_at, payload_dict)}
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

    # Éviction simple si le cache grossit trop
    if len(CACHE) >= CACHE_MAX_ITEMS:
        cutoff = int(CACHE_MAX_ITEMS * 0.1) or 1
        for k in list(CACHE.keys())[:cutoff]:
            CACHE.pop(k, None)

    CACHE[key] = (_now() + CACHE_TTL_SECONDS, payload)


def _build_line_text(author: str, original_text: str, detected_language: str, translations: dict, ordered_langs: list[str]) -> str:
    # Format prêt à coller dans LINE / WhatsApp
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
    # Tolérance: "spanish" -> "es", "french" -> "fr", etc.
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
    # Si ça ressemble déjà à un code ISO
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

    # Normalisation languages (conserve l’ordre)
    if not isinstance(languages, list) or not all(isinstance(x, str) for x in languages):
        return jsonify({"error": "languages must be a list of strings"}), 400

    ordered_langs = [x.strip().lower() for x in languages if x and x.strip()]
    if not ordered_langs:
        ordered_langs = DEFAULT_LANGS

    # --------------- CACHE ---------------
    cache_key = _make_cache_key(author, text, ordered_langs)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    # --------------- OPENAI ---------------
    # Objectif: 1 seul appel, JSON strict, traduction naturelle (capitalisation/ponctuation),
    # sans markdown ni texte additionnel.
    prompt = {
        "role": "system",
        "content": (
            "You are a professional translator. "
            "Detect the source language automatically. "
            "Return a natural, idiomatic translation in each target language. "
            "Keep the same tone, do not add explanations. "
            "Use proper capitalization and punctuation. "
            "Return ONLY valid JSON (no markdown, no code fences, no extra text)."
        )
    }

    user_msg = {
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
            '    "it": "..." \n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- detected_language MUST be a 2-letter ISO code when possible (en/fr/es/it/fa).\n"
            "- translations keys MUST match the provided target languages list.\n"
            "- Output MUST be valid JSON only."
        )
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[prompt, user_msg],
        )

        raw = (resp.choices[0].message.content or "").strip()

        # Parse JSON
        result = json.loads(raw)

        detected_language = _normalize_detected_language(str(result.get("detected_language", "unknown")))
        translations = result.get("translations", {}) or {}

        # Respecter l’ordre et forcer la présence des clés demandées
        ordered_translations = {}
        for lang in ordered_langs:
            t = str(translations.get(lang, "")).strip()
            # Nettoyage minimal au cas où
            t = t.replace("```", "").replace("**", "").strip()
            ordered_translations[lang] = t

        payload = {
            "author": author,
            "original_text": text,
            "detected_language": detected_language,
            "translations": ordered_translations
        }

        if include_line_text:
            payload["line_text"] = _build_line_text(author, text, detected_language, ordered_translations, ordered_langs)

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
