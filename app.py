from flask import Flask, request, jsonify
import os
import time
import json
import hashlib
import base64
import hmac
import requests
from openai import OpenAI

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
DEFAULT_LANGS = ["en", "fr", "es", "it"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))  # 24h
CACHE_MAX_ITEMS = int(os.environ.get("CACHE_MAX_ITEMS", "2000"))

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

# Cache in-memory : {cache_key: (expires_at, payload_dict)}
CACHE = {}


# -------------------------
# HELPERS: CACHE
# -------------------------
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


# -------------------------
# HELPERS: FORMAT OUTPUT
# -------------------------
def _build_line_text(author: str, original_text: str, detected_language: str,
                     translations: dict, ordered_langs: list[str]) -> str:
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
# HELPERS: LINE SECURITY + REPLY
# -------------------------
def _verify_line_signature(raw_body: bytes, signature: str) -> bool:
    """
    Vérification HMAC-SHA256 (recommandé).
    LINE envoie la signature dans l'entête: X-Line-Signature
    """
    if not LINE_CHANNEL_SECRET:
        # En dev, on peut tolérer. En prod, mets toujours le secret.
        return True

    mac = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256
    ).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


def _line_reply(reply_token: str, message_text: str):
    """
    Envoi d'une réponse au chat LINE.
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return False, "LINE_CHANNEL_ACCESS_TOKEN missing"

    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [
            {"type": "text", "text": message_text}
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=15)
    if r.status_code >= 300:
        return False, f"LINE reply failed: {r.status_code} - {r.text}"
    return True, "OK"


# -------------------------
# TRANSLATION CORE
# -------------------------
def _translate(author: str, text: str, ordered_langs: list[str], include_line_text: bool = True) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    cache_key = _make_cache_key(author, text, ordered_langs)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # PROMPT "GÉNÉRALISÉ" POUR ÉVITER LES ERREURS RÉCURRENTES
    # Objectif: traductions naturelles, correction erreurs, sens, registre, contexte culturel.
    prompt = {
        "role": "user",
        "content": (
            "You are a multilingual communication translator for casual chat messages.\n"
            "Goal: produce natural, idiomatic translations that sound like a real native speaker.\n\n"
            "Key requirements:\n"
            "- Do NOT translate word-for-word.\n"
            "- Correct typos, missing punctuation, and informal grammar so the message is clear.\n"
            "- Preserve the intent, tone, and social register (casual stays casual; formal stays formal).\n"
            "- Interpret common slang, regional expressions, and context; avoid unnatural calques.\n"
            "- Preserve meaning and timing/aspect (e.g., ongoing action vs future intention).\n"
            "- Keep emojis and emphasis if present.\n"
            "- If the original is very informal, keep it natural (avoid stiff or overly formal rewrites).\n\n"
            "Output rules:\n"
            "- Return ONLY valid JSON. No markdown. No extra text.\n\n"
            f"Target languages (in order): {ordered_langs}\n"
            f"Text: {text}\n\n"
            "Return this JSON schema:\n"
            "{\n"
            '  "detected_language": "<language_code>",\n'
            '  "translations": {\n'
            '    "<lang>": "<natural_translation>"\n'
            "  }\n"
            "}\n"
        )
    }

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt],
    )

    raw = (resp.choices[0].message.content or "").strip()
    result = json.loads(raw)

    detected_language = str(result.get("detected_language", "unknown")).strip()
    translations = result.get("translations", {}) or {}

    ordered_translations = {
        lang: str(translations.get(lang, "")).strip()
        for lang in ordered_langs
    }

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
    return payload


# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


# -------------------------
# REST API: /translate
# -------------------------
@app.route("/translate", methods=["POST"])
def translate():
    data = request.json or {}
    author = data.get("author", "Unknown")
    text = (data.get("text") or "").strip()
    languages = data.get("languages") or DEFAULT_LANGS
    include_line_text = bool(data.get("include_line_text", True))

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not isinstance(languages, list) or not all(isinstance(x, str) for x in languages):
        return jsonify({"error": "languages must be a list of strings"}), 400

    ordered_langs = [x.strip() for x in languages if x and x.strip()] or DEFAULT_LANGS

    try:
        payload = _translate(author, text, ordered_langs, include_line_text=include_line_text)
        if payload.get("error"):
            return jsonify(payload), 500
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"error": "internal_error", "details": str(e)}), 500


# -------------------------
# LINE WEBHOOK: /webhook
# -------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    raw_body = request.get_data()  # bytes
    signature = request.headers.get("X-Line-Signature", "")

    # 1) Sécurité: signature
    if not _verify_line_signature(raw_body, signature):
        return "Invalid signature", 403

    # 2) Parsing JSON LINE
    try:
        body = request.json or {}
        events = body.get("events", [])
    except Exception:
        return "Bad request", 400

    # 3) Pour chaque événement message, on traduit et on répond
    for ev in events:
        ev_type = ev.get("type")
        if ev_type != "message":
            continue

        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        if not reply_token:
            continue

        text = (msg.get("text") or "").strip()
        if not text:
            continue

        # Auteur (si disponible)
        source = ev.get("source", {}) or {}
        author = source.get("userId", "Unknown")

        # Langues cibles (config standard)
        ordered_langs = DEFAULT_LANGS

        try:
            payload = _translate(author=author, text=text, ordered_langs=ordered_langs, include_line_text=True)
            if payload.get("error"):
                _line_reply(reply_token, f"Error: {payload.get('error')}")
                continue

            # Message final à envoyer sur LINE: format bloc
            out_text = payload.get("line_text") or "OK"
            _line_reply(reply_token, out_text)

        except Exception as e:
            _line_reply(reply_token, f"internal_error: {str(e)}")

    # LINE attend un 200
    return "OK", 200


# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
