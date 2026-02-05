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
        # Supprime ~10% des plus anciens/expirés
        cutoff = int(CACHE_MAX_ITEMS * 0.1) or 1
        for k in list(CACHE.keys())[:cutoff]:
            CACHE.pop(k, None)

    CACHE[key] = (_now() + CACHE_TTL_SECONDS, payload)


def _build_line_text(author: str, original_text: str, detected_language: str, translations: dict, ordered_langs: list[str]) -> str:
    # Format prêt à coller dans LINE / WhatsApp
    # (tu peux ajuster la mise en page plus tard, mais ça reste stable et lisible)
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

    # Option : inclure un flag pour récupérer un bloc prêt à coller
    include_line_text = bool(data.get("include_line_text", True))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Normalisation languages (conserve l’ordre)
    if not isinstance(languages, list) or not all(isinstance(x, str) for x in languages):
        return jsonify({"error": "languages must be a list of strings"}), 400
    ordered_langs = [x.strip() for x in languages if x and x.strip()]

    if not ordered_langs:
        ordered_langs = DEFAULT_LANGS

    # --------------- CACHE ---------------
    cache_key = _make_cache_key(author, text, ordered_langs)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached), 200

    # --------------- OPENAI ---------------
    # Un seul appel : détecter la langue + traduire vers N langues.
    # Le modèle doit retourner un JSON strict.
    prompt = {
        "role": "user",
        "content": (
            "You are a professional translation engine.\n"
            "Task:\n"
            "1) Detect the language of the input text.\n"
            "2) Translate the text into each target language code provided.\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- No markdown, no code fences, no extra text.\n"
            "- Keep meaning, tone, emojis.\n"
            "- Do not add commentary.\n\n"
            f"Target languages (in order): {ordered_langs}\n"
            f"Text: {text}\n\n"
            "Return this JSON schema:\n"
            "{\n"
            '  "detected_language": "<language_code_or_name>",\n'
            '  "translations": {\n'
            '     "<lang>": "<translated_text>",\n'
            '     "...": "..." \n'
            "  }\n"
            "}\n"
        )
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[prompt],
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Parse JSON robuste
        result = json.loads(raw)
        detected_language = str(result.get("detected_language", "unknown")).strip()
        translations = result.get("translations", {}) or {}

        # Assure que l’ordre est respecté et que toutes les clés existent
        ordered_translations = {lang: str(translations.get(lang, "")).strip() for lang in ordered_langs}

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
