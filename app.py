from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200

@app.route("/translate", methods=["POST"])
def translate():
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return jsonify({"error": "OPENAI_API_KEY not set"}), 500

        client = OpenAI(api_key=api_key)

        data = request.get_json(force=True)
        author = data.get("author", "Unknown")
        text = data.get("text", "")
        target_languages = data.get("languages", ["en", "fr", "es"])

        if not text:
            return jsonify({"error": "No text provided"}), 400

        translations = {}

        for lang in target_languages:
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=f"Translate the following text into {lang}:\n{text}"
            )
            translations[lang] = resp.output_text

        return jsonify({
            "author": author,
            "original_text": text,
            "translations": translations
        }), 200

    except Exception as e:
        return jsonify({
            "error": "internal_error",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
