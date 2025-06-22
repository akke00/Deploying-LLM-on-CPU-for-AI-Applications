# app.py
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from llama_cpp import Llama
from rag import RAGSearch, load_and_index_file
import os
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
CORS(app)

MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    use_mmap=True,
    use_mlock=True,
    low_vram=True,
    n_gpu_layers=0
)

rag = RAGSearch()

# Session memory
def get_history():
    if "history" not in session:
        session["history"] = []
    return session["history"]

def append_history(role, content):
    history = get_history()
    history.append({"role": role, "content": content})
    session["history"] = history[-5:]

# Prompt formatting

def build_prompt(history, system_prompt="Ты — дружелюбный ИИ, который говорит по-русски и помогает пользователю."):
    prompt = f"<s>[INST] {system_prompt} [/INST]"
    for msg in history:
        if msg['role'] == 'user':
            prompt += f"\n<s>[INST] {msg['content']} [/INST]"
        elif msg['role'] == 'assistant':
            prompt += f" {msg['content']} </s>"
    prompt += "\n"
    return prompt

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    append_history("user", user_input)

    # RAG: поиск контекста
    context_chunks = rag.query(user_input)
    context_text = "\n".join(context_chunks)

    messages = get_history()
    prompt = build_prompt(messages, system_prompt=f"Ты — ИИ, говорящий по-русски. Используй информацию из текста ниже, если она помогает:\n{context_text}")

    output = llm(
        prompt,
        stop=["</s>", "[INST]"],
        max_tokens=256,
        temperature=0.7
    )
    answer = output["choices"][0]["text"].strip()
    append_history("assistant", answer)
    return jsonify({"response": answer})

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Файл не получен"}), 400
    file = request.files["file"]
    filename = file.filename
    path = os.path.join("uploads", filename)
    file.save(path)
    load_and_index_file(path, rag)
    return jsonify({"status": "Файл обработан"})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)


