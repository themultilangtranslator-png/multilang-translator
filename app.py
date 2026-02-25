from flask import Flask, request, jsonify
import os
import json
import hashlib
import hmac
import base64
import traceback
import requests
from openai import OpenAI

app = Flask(__name__)

DEFAULT_LANGS = ["en", "fr", "es", "it", "fa", "de"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ACK_TEXT = "⏳ Traduction en cours…"

FLAG_MAP = {
    "en": "🇺🇸",
    "fr": "🇫🇷",
    "es": "🇪🇸",
    "it": "🇮🇹",
    "fa": "🇮🇷",
    "de": "🇩🇪",
}


# -------------------------
# SIGNATURE CHECK
# -------------------------
def verify_line_signature(raw_body: bytes, signature: str, channel_secret: str) -> bool:
    # Guards (évite 500)
    if not raw_body:
        return False
    if not signature or not isinstance(signature, str):
        return False
    if not channel_secret or not isinstance(channel_secret, str):
        return False

    mac = hmac.new(channel_secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# -------------------------
# LINE PUSH
# -------------------------
def push_to_line(to_id: str, text: str) -> bool:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN not set")
        return False
    if not to_id:
        print("ERROR: Missing to_id for push")
        return False
    if text is None:
        text = ""

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": to_id,
        "messages": [{"type": "text", "text": str(text)[:4900]}],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        ok = (r.status_code == 200)
        if not ok:
            print("LINE PUSH FAILED:", r.status_code, r.text)
        return ok
    except Exception:
        print("LINE PUSH EXCEPTION:", traceback.format_exc())
        return False


# -------------------------
# OPENAI TRANSLATION
# -------------------------
def translate_text(author: str, text: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

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
            "   - Keep the same intent, tone, and level of formality.\n\n"
            "Output rules:\n"
            "- Return ONLY valid JSON. No markdown, no code fences, no extra text.\n"
            "- Keep each translation to a single message.\n\n"
            f"Target languages (in order): {DEFAULT_LANGS}\n"
            f"Text: {text}\n\n"
            "Return this JSON schema:\n"
            "{\n"
            '  "detected_language": "<language_code_or_name>",\n'
            '  "translations": {\n'
            '     "<lang>": "<translated_text>"\n'
            "  }\n"
            "}\n"
        ),
    }

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt],
        temperature=0.2,
    )

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("OpenAI returned empty response")

    # Parse JSON (avec erreur explicite)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"OpenAI response was not valid JSON: {raw[:400]}")

    detected = str(data.get("detected_language", "unknown")).strip()
    translations = data.get("translations", {}) or {}
    if not isinstance(translations, dict):
        translations = {}

    def clean(s: str) -> str:
        return (s or "").replace("\n", " ").replace("\r", " ").strip()

    lines = [
        f"👤 {clean(author)}",
        f"🌐 {clean(detected)}",
        f"📝 {clean(text)}",
    ]

    for lang in DEFAULT_LANGS:
        flag = FLAG_MAP.get(lang, f"🏳️({lang})")
        t = clean(str(translations.get(lang, "")))
        lines.append(f"{flag} {t}")

    return "\n".join(lines)


# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return "Translator running", 200


@app.route("/translate", methods=["POST"])
def translate_api():
    try:
        data = request.json or {}
        author = str(data.get("author", "Unknown"))
        text = str((data.get("text") or "")).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        out = translate_text(author, text)
        return jsonify({"line_text": out}), 200

    except Exception as e:
        return jsonify({"error": "internal_error", "details": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        raw_body = request.get_data()  # bytes
        signature = request.headers.get("X-Line-Signature", "")
        channel_secret = os.environ.get("LINE_CHANNEL_SECRET", "")

        if not verify_line_signature(raw_body, signature, channel_secret):
            return "Invalid signature", 400

        body = json.loads(raw_body.decode("utf-8"))
        events = body.get("events", []) or []

        # Log minimal (utile pour Railway)
        print("INCOMING EVENTS:", len(events))

        for event in events:
            try:
                if event.get("type") != "message":
                    continue

                message = event.get("message", {}) or {}
                if message.get("type") != "text":
                    continue

                source = event.get("source", {}) or {}
                source_type = source.get("type", "")
                text = str((message.get("text") or "")).strip()
                if not text:
                    continue

                # IMPORTANT: répondre dans le même contexte (groupe/room/privé)
                if source_type == "group":
                    to_id = source.get("groupId", "")
                elif source_type == "room":
                    to_id = source.get("roomId", "")
                else:
                    to_id = source.get("userId", "")

                user_id = source.get("userId", "")
                author = f"User-{user_id[-4:]}" if user_id else "Unknown"

                # ACK immédiat (push)
                push_to_line(to_id, ACK_TEXT)

                # Traduction
                translated = translate_text(author, text)

                # Envoi final
                push_to_line(to_id, translated)

            except Exception:
                print("EVENT ERROR:", traceback.format_exc())
                continue

        return "OK", 200

    except Exception:
        print("WEBHOOK ERROR:", traceback.format_exc())
        return "Internal error", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
