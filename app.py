# app.py
from flask import Flask, request, jsonify
import os
import time
import json
import hashlib
import hmac
import base64
import requests
import threading
import traceback
from openai import OpenAI

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
DEFAULT_LANGS = ["en", "fr", "es", "it", "fa", "de"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))  # 24h
CACHE_MAX_ITEMS = int(os.environ.get("CACHE_MAX_ITEMS", "2000"))

PROFILE_CACHE_TTL_SECONDS = int(os.environ.get("PROFILE_CACHE_TTL_SECONDS", "86400"))  # 24h

CACHE = {}          # {cache_key: (expires_at, payload_dict)}
PROFILE_CACHE = {}  # {user_id: (expires_at, profile_dict)}


# -------------------------
# TIME / CACHE
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


def _profile_cache_get(user_id: str):
    item = PROFILE_CACHE.get(user_id)
    if not item:
        return None
    expires_at, profile = item
    if _now() >= expires_at:
        PROFILE_CACHE.pop(user_id, None)
        return None
    return profile


def _profile_cache_set(user_id: str, profile: dict):
    if PROFILE_CACHE_TTL_SECONDS <= 0:
        return
    PROFILE_CACHE[user_id] = (_now() + PROFILE_CACHE_TTL_SECONDS, profile)


# -------------------------
# OUTPUT FORMAT (no empty lines)
# -------------------------
def _build_line_text(author: str, original_text: str, detected_language: str, translations: dict, ordered_langs: list[str]) -> str:
    flag_map = {
        "en": "🇺🇸",
        "fr": "🇫🇷",
        "es": "🇪🇸",
        "it": "🇮🇹",
        "fa": "🇮🇷",
        "de": "🇩🇪",
        "pt": "🇵🇹",
        "nl": "🇳🇱",
        "ar": "🇸🇦",
        "ja": "🇯🇵",
        "ko": "🇰🇷",
        "zh": "🇨🇳",
        "ru": "🇷🇺",
    }

    def clean(s: str) -> str:
        return (s or "").replace("\n", " ").replace("\r", " ").strip()

    lines = []
    lines.append(f"👤 {clean(author)}")
    lines.append(f"🌐 {clean(detected_language)}")
    lines.append(f"📝 {clean(original_text)}")

    for lang in ordered_langs:
        flag = flag_map.get(lang.lower(), f"🏳️({lang})")
        text = clean(translations.get(lang, ""))
        lines.append(f"{flag} {text}")

    return "\n".join(lines)


# -------------------------
# LINE HELPERS
# -------------------------
def verify_line_signature(raw_body: bytes, signature: str, channel_secret: str) -> bool:
    if not signature or not channel_secret:
        return False
    mac = hmac.new(channel_secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def get_line_profile(user_id: str) -> dict:
    if not user_id:
        return {}

    cached = _profile_cache_get(user_id)
    if cached:
        return cached

    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        return {}

    url = f"https://api.line.me/v2/bot/profile/{user_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            profile = r.json() or {}
            _profile_cache_set(user_id, profile)
            return profile
    except Exception:
        pass

    return {}


def reply_to_line(reply_token: str, text: str) -> bool:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token or not reply_token:
        return False

    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:4900]}],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code != 200:
            print("REPLY FAILED:", r.status_code, r.text)
        return r.status_code == 200
    except Exception as e:
        print("REPLY EXCEPTION:", str(e))
        return False


def push_to_line(to_id: str, text: str) -> bool:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token or not to_id:
        return False

    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "to": to_id,
        "messages": [{"type": "text", "text": text[:4900]}],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code != 200:
            print("PUSH FAILED:", r.status_code, r.text)
        return r.status_code == 200
    except Exception as e:
        print("PUSH EXCEPTION:", str(e))
        return False


# -------------------------
# OPENAI TRANSLATION CORE
# -------------------------
def translate_core(author: str, text: str, ordered_langs: list[str], include_line_text: bool = True) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    cache_key = _make_cache_key(author, text, ordered_langs)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prompt = {
        "role": "user",
        "content": (
            "You are a professional chat translator.\n"
            "Goal: produce natural, idiomatic translations suitable for real chat.\n\n"
            "Do this:\n"
            "1) Detect the language of the input text.\n"
            "2) For each target language, rewrite the message so it sounds natural to a native speaker.\n"
            "   - Fix typos, missing punctuation, and obvious grammar issues.\n"
            "   - Interpret idioms, slang, and regional expressions correctly.\n"
            "   - If the original is unclear, choose the most plausible meaning and make it readable.\n"
            "   - Keep the same intent, tone, and level of formality (do NOT over-formalize).\n"
            "   - Preserve emojis and emphasis.\n\n"
            "Output rules:\n"
            "- Return ONLY valid JSON. No markdown, no code fences, no extra text.\n"
            "- Keep each translation to a single message (no explanations).\n"
            "- Use proper capitalization and punctuation in each language.\n\n"
            f"Target languages (in order): {ordered_langs}\n"
            f"Text: {text}\n\n"
            "Return this JSON schema:\n"
            "{\n"
            '  "detected_language": "<language_code_or_name>",\n'
            '  "translations": {\n'
            '     "<lang>": "<translated_text>",\n'
            '     "...": "..."\n'
            "  }\n"
            "}\n"
        )
    }

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt],
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "").strip()

    result = json.loads(raw)
    detected_language = str(result.get("detected_language", "unknown")).strip()
    translations = result.get("translations", {}) or {}

    ordered_translations = {lang: str(translations.get(lang, "")).strip() for lang in ordered_langs}

    payload = {
        "author": author,
        "original_text": text,
        "detected_language": detected_language,
        "translations": ordered_translations,
    }

    if include_line_text:
        payload["line_text"] = _build_line_text(author, text, detected_language, ordered_translations, ordered_langs)

    _cache_set(cache_key, payload)
    return payload


def translate_text(author: str, text: str) -> str:
    payload = translate_core(author, text, DEFAULT_LANGS, include_line_text=True)
    return payload.get("line_text") or "Translation unavailable."


# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


# -------------------------
# API TEST
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
        payload = translate_core(author, text, ordered_langs, include_line_text=include_line_text)
        return jsonify(payload), 200
    except json.JSONDecodeError:
        return jsonify({"error": "internal_error", "details": "OpenAI response was not valid JSON"}), 500
    except Exception as e:
        return jsonify({"error": "internal_error", "details": str(e)}), 500


# -------------------------
# ASYNC WORKER
# -------------------------
def _process_event_async(reply_token: str, to_id: str, author: str, text: str):
    try:
        # Traduire
        line_text = translate_text(author, text)

        # 1) Essayer de répondre via replyToken (dans le même chat)
        ok = reply_to_line(reply_token, line_text)

        # 2) Fallback push si replyToken expiré / reply fail
        if not ok:
            push_to_line(to_id, line_text)

    except Exception:
        print("ASYNC ERROR:", traceback.format_exc())


# -------------------------
# LINE WEBHOOK
# -------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    raw_body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")
    channel_secret = os.environ.get("LINE_CHANNEL_SECRET", "")

    if not verify_line_signature(raw_body, signature, channel_secret):
        return "Invalid signature", 400

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return "Bad request", 400

    events = body.get("events", []) or []
    for event in events:
        try:
            if event.get("type") != "message":
                continue

            message = event.get("message", {}) or {}
            if message.get("type") != "text":
                continue

            source = event.get("source", {}) or {}
            source_type = source.get("type", "")

            # Destination (group/room/user)
            if source_type == "group":
                to_id = source.get("groupId", "")
            elif source_type == "room":
                to_id = source.get("roomId", "")
            else:
                to_id = source.get("userId", "")

            reply_token = event.get("replyToken", "")
            if not reply_token or not to_id:
                continue

            user_id = source.get("userId", "")
            profile = get_line_profile(user_id)
            author = profile.get("displayName") or (f"User-{user_id[-4:]}" if user_id else "Unknown")

            text = (message.get("text") or "").strip()
            if not text:
                continue

            # Thread (200 OK immédiat)
            t = threading.Thread(
                target=_process_event_async,
                args=(reply_token, to_id, author, text),
                daemon=True
            )
            t.start()

        except Exception:
            print("WEBHOOK EVENT ERROR:", traceback.format_exc())
            continue

    return "OK", 200


# -------------------------
# ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
