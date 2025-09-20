# download_and_run_models.py

from gpt4all import GPT4All
import shutil
import os
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# -------------------------------
# Paths
# -------------------------------
MODELS_FOLDER = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\models"
DATA_FOLDER = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\data"
DATA_EMERGENCY_FOLDER = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\data_emergency"

os.makedirs(MODELS_FOLDER, exist_ok=True)

# -------------------------------
# Download models function
# -------------------------------
def download_model(model_name, target_folder):
    print(f"Downloading {model_name}...")
    gpt_model = GPT4All(model_name)  # downloads to ~/.gpt4all
    default_path = os.path.expanduser(f"~/.gpt4all/{model_name}.gguf")
    if os.path.exists(default_path):
        shutil.move(default_path, os.path.join(target_folder, f"{model_name}.gguf"))
        print(f"{model_name} moved to {target_folder}")
    else:
        print(f"Model {model_name} not found at default location. Please check download.")

# Download 1B and 3B models
download_model("orca-mini-1b-q4_0", MODELS_FOLDER)
download_model("orca-mini-3b-q4_0", MODELS_FOLDER)

# -------------------------------
# Choose model to use
# -------------------------------
# Uncomment the one you want to use:
MODEL_PATH = os.path.join(MODELS_FOLDER, "orca-mini-1b-q4_0.gguf")  # phone-friendly
# MODEL_PATH = os.path.join(MODELS_FOLDER, "orca-mini-3b-q4_0.gguf")  # higher accuracy

# Load GPT4All model
llm = GPT4All(MODEL_PATH)

# -------------------------------
# Load embedding model
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return embed_model.encode([text], convert_to_numpy=True).astype('float32')

# -------------------------------
# Ingest data if FAISS indexes don't exist
# -------------------------------
def ingest_folder(folder_path, index_path, json_path):
    import faiss
    texts = []
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read().strip()
                texts.append(text)
                documents.append({"text": text, "source": file_name})

    embeddings = embed_text(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(json_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"Ingested {folder_path}: {len(texts)} documents")

# Paths for FAISS indexes
FAISS_GENERAL = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\faiss.index"
FAISS_EMERGENCY = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\faiss_emergency.index"
DOCS_GENERAL = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\documents.jsonl"
DOCS_EMERGENCY = r"C:\Users\lakis\OneDrive\Desktop\telemedicine-rag\documents_emergency.jsonl"

if not os.path.exists(FAISS_GENERAL):
    ingest_folder(DATA_FOLDER, FAISS_GENERAL, DOCS_GENERAL)
if not os.path.exists(FAISS_EMERGENCY):
    ingest_folder(DATA_EMERGENCY_FOLDER, FAISS_EMERGENCY, DOCS_EMERGENCY)

# -------------------------------
# Load FAISS indexes and documents
# -------------------------------
def load_faiss_and_docs(index_file, doc_file):
    index = faiss.read_index(index_file)
    with open(doc_file, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return index, docs

index_general, docs_general = load_faiss_and_docs(FAISS_GENERAL, DOCS_GENERAL)
index_emergency, docs_emergency = load_faiss_and_docs(FAISS_EMERGENCY, DOCS_EMERGENCY)

# -------------------------------
# Emergency setup
# -------------------------------
EMERGENCY_KEYWORDS = ["chest pain", "fainting", "choking", "allergic reaction",
                      "heatstroke", "hypothermia", "severe bleeding", "breathing difficulty"]

EMERGENCY_FOLLOWUPS = {
    "chest pain": ["On a scale 1-10, how severe is the pain?", "Left or right side?"],
    "fainting": ["Did you lose consciousness?", "Any dizziness before?"],
    "choking": ["Can you breathe?", "How long obstruction?"],
    "allergic reaction": ["Any difficulty breathing?", "Any swelling or hives?"],
}

def check_emergency(question):
    for kw in EMERGENCY_KEYWORDS:
        if kw in question.lower():
            return True, kw
    return False, None

# -------------------------------
# FAISS retrieval
# -------------------------------
def retrieve_topk(text, index, docs, k=3):
    q_emb = embed_text(text)
    D, I = index.search(q_emb, k)
    return [docs[idx] for idx in I[0] if 0 <= idx < len(docs)]

# -------------------------------
# Interactive loop
# -------------------------------
print("Telemedicine assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting. Stay safe!")
        break

    sos_triggered = False
    urgency_score = 0
    followups = {}
    retrieved = []

    is_emergency, emergency_type = check_emergency(user_input)

    if is_emergency:
        followups = EMERGENCY_FOLLOWUPS.get(emergency_type, [])
        retrieved = retrieve_topk(user_input, index_emergency, docs_emergency)
        intensity_q = "On a scale 1-10, how severe is your problem?"
        urgency_input = input(f"{intensity_q} ")
        urgency_score = int(urgency_input) if urgency_input.isdigit() else 7
    else:
        intensity_q = "On a scale 1-10, how severe is your problem?"
        urgency_input = input(f"{intensity_q} ")
        urgency_score = int(urgency_input) if urgency_input.isdigit() else 0
        retrieved = retrieve_topk(user_input, index_general, docs_general)

    if urgency_score >= 5:
        sos_triggered = True

    context_text = "\n".join([r["text"] for r in retrieved])
    prompt = f"Use the following information to give a concise, accurate first aid answer.\n\nContext:\n{context_text}\n\nQuestion: {user_input}\nAnswer:"

    answer = llm.generate(prompt)
    print(f"\nAssistant: {answer}")
    if sos_triggered:
        print("⚠️ SOS triggered due to high urgency!")
