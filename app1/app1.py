from flask import Flask, request, jsonify, render_template
import openai
from flask_cors import CORS

app1 = Flask(__name__)
CORS(app1)  # Enable CORS to allow frontend to interact with the backend

# OpenAI API Key
openai.api_key = "your_openai_api_key"

@app1.route("/")
def home():
    return render_template('index.html')

@app1.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Generate AI response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_input,
            max_tokens=150,
            temperature=0.7,
        )
        return jsonify({"response": response.choices[0].text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app1.run(debug=True)
