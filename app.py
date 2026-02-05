import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TARGET_LANGS = ["fr", "en", "es", "it"]

@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}

    member = data.get("member")
    text = data.get("text")

    if not member or not text:
        return jsonify({
            "error": "member and text are required"
        }), 400

    prompt = f"""
You are a professional translator.

Return ONLY valid JSON in the following format:

{{
  "member": "{member}",
  "original_text": "{text}",
  "detected_language": "<language>",
  "translations": {{
    "fr": "<french>",
    "en": "<english>",
    "es": "<spanish>",
    "it": "<italian>"
  }}
}}

Rules:
- Keep the original text unchanged.
- Detect the language automatically.
- Do NOT translate into the detected language.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    try:
        result = json.loads(response.output_text)
    except Exception:
        return jsonify({
            "error": "Invalid AI response",
            "raw": response.output_text
        }), 500

    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
