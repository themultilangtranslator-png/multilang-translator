from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


@app.route("/translate", methods=["POST"])
def translate():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    client = OpenAI(api_key=api_key)

    data = request.json or {}
    author = data.get("author", "Unknown")
    text = data.get("text", "")
    target_languages = data.get("languages", ["en", "fr", "es"])

    if not text:
        return jsonify({"error": "No text provided"}), 400

    translations = {}

    for lang in target_languages:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the text into {lang}."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        translations[lang] = response.choices[0].message.content.strip()

    return jsonify({
        "author": author,
        "original_text": text,
        "translations": translations
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
