from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def health_check():
    return "Translator service running", 200


@app.route("/translate", methods=["POST"])
def translate():
    data = request.json

    member = data.get("member")
    text = data.get("text")

    if not member or not text:
        return jsonify({"error": "member and text are required"}), 400

    prompt = f"""
You are a professional translator.

Original message from member: {member}

Text:
{text}

Return the result in JSON with this structure:
{{
  "member": "{member}",
  "original": "{text}",
  "translations": {{
    "fr": "...",
    "en": "...",
    "es": "...",
    "fa": "..."
  }}
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You translate text accurately."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = response.choices[0].message.content

    return jsonify({"result": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
