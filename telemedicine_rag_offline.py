import os
import json
from pathlib import Path
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
import faiss
import re  # <-- MODIFICATION: Import the regular expression module

# ------------------ Paths ------------------
# Using relative paths is the best practice and makes your script portable.
DATA_NORMAL = Path("data")
DATA_EMERGENCY = Path("data_emergency")
MODEL_FOLDER = Path("models")
MODEL_FILENAME = "orca-mini-1b.gguf"
MODEL_PATH = MODEL_FOLDER / MODEL_FILENAME
INDEX_NORMAL_FILE = Path("faiss_normal.index")
INDEX_EMERGENCY_FILE = Path("faiss_emergency.index")
SUMMARY_FILE = Path("patient_summary.jsonl")

# ------------------ Load documents ------------------
def load_text_files(folder):
    """Loads all .txt files from a given folder."""
    texts = []
    if not folder.is_dir():
        print(f"Warning: Directory '{folder}' not found.")
        return texts
    for file in folder.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

# ------------------ Chunking ------------------
def chunk_text(text, chunk_size=200):
    """Splits text into chunks of a specified size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ------------------ Build FAISS index ------------------
def build_faiss_index(texts, model_name="all-MiniLM-L6-v2"):
    """Builds a FAISS index from a list of texts."""
    embedder = SentenceTransformer(model_name)
    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))
    
    if not all_chunks:
        print(f"Warning: No text chunks found to build an index.")
        return None, [], embedder
        
    embeddings = embedder.encode(all_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, all_chunks, embedder

# ------------------ Save patient summary ------------------
def save_patient_summary(summary_dict):
    """Appends a patient summary to the JSONL file."""
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_dict, ensure_ascii=False) + "\n")

# ------------------ Emergency detection ------------------
def detect_emergency(symptoms, severity):
    """Detects if a situation is an emergency based on keywords and severity."""
    emergency_keywords = ["chest pain", "shortness of breath", "high fever", "fainting", "severe bleeding", "unconscious"]
    detected = any(k in symptoms.lower() for k in emergency_keywords)
    final_severity = max(severity, 5 if detected else severity)
    return final_severity >= 5

# --- SOS Simulation Function ---
def trigger_sos():
    """Simulates triggering an in-app SOS call to emergency services."""
    print("\n" + "="*50)
    print("!!! EMERGENCY DETECTED !!!")
    print(">>> SIMULATING SOS CALL TO EMERGENCY SERVICES <<<")
    print("Please seek immediate medical help.")
    print("="*50 + "\n")

# ------------------ Main ------------------
print("Loading documents...")
texts_normal = load_text_files(DATA_NORMAL)
texts_emergency = load_text_files(DATA_EMERGENCY)

print("Generating embeddings and building indices...")
index_normal, chunks_normal, embedder_normal = build_faiss_index(texts_normal)
index_emergency, chunks_emergency, embedder_emergency = build_faiss_index(texts_emergency)

print(f"Attempting to load model: {MODEL_PATH}")

if not MODEL_PATH.is_file():
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'")
    print(f"Please make sure the '{MODEL_FILENAME}' model is inside the '{MODEL_FOLDER}' directory.")
    exit()

try:
    llm = GPT4All(
        model_name=MODEL_FILENAME,
        model_path=str(MODEL_FOLDER)
    )
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    exit()

print("\n✅ Telemedicine system loaded. Type 'exit' to quit.\n")

while True:
    user_input = input("Please describe your symptoms or ask a health question:\n> ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # --- START OF MODIFICATION ---
    # This entire block is the change you requested.
    temperature = "Not provided"
    if "fever" in user_input.lower():
        # Search for a number (like 101, 99.5, 38.2) in the initial input.
        match = re.search(r'\d+(\.\d+)?', user_input)
        
        if match:
            # If a number is found, capture it as the temperature.
            temperature = match.group(0)
            print(f"--- Temperature auto-detected: {temperature} ---")
        else:
            # If "fever" is mentioned but no number is found, then ask the user.
            temp_input = input("You mentioned fever. What is your temperature (e.g., 99.5 F or 37.5 C)?\n> ").strip()
            if temp_input:
                temperature = temp_input
    # --- END OF MODIFICATION ---
    
    severity_input = input("On a scale of 1-10, how severe is your problem?\n> ").strip()
    try:
        severity = int(severity_input)
    except ValueError:
        print("Invalid input for severity. Defaulting to 1.")
        severity = 1

    affected_area = input("Specify affected area (if relevant, e.g., head, chest, left arm)\n> ").strip() or "Not specified"

    # Emergency check
    is_emergency = detect_emergency(user_input, severity)

    if is_emergency:
        trigger_sos()
        index, chunks, embedder = index_emergency, chunks_emergency, embedder_emergency
    else:
        index, chunks, embedder = index_normal, chunks_normal, embedder_normal
        
    if index is None or index.ntotal == 0:
        print("\n🤖 Response:\n", "I'm sorry, my knowledge base for this category is empty. I cannot provide advice.", "\n")
        continue

    # Retrieve relevant chunk
    query_emb = embedder.encode([user_input])
    D, I = index.search(query_emb, k=2)
    retrieved_text = "\n".join([chunks[i] for i in I[0]])

    # --- UPDATED PROMPT: Now includes temperature ---
    prompt = f"""SYSTEM: You are a professional medical triage AI. Your sole purpose is to provide direct, actionable advice based on the user's symptoms.

RULES:
- Be direct and concise.
- Do NOT use conversational filler (e.g., "I'm sorry to hear that," "I hope you feel better").
- Base your advice ONLY on the provided context.
- If the situation is an emergency, state clearly to seek immediate medical attention.

PATIENT DATA:
- Symptoms: {user_input}
- Temperature: {temperature}
- Severity (1-10): {severity}
- Affected Area: {affected_area}

RELEVANT MEDICAL INFORMATION:
{retrieved_text}

ADVICE:
"""
    response = llm.generate(prompt)
    print("\n🤖 Response:\n", response.strip(), "\n")

    # --- UPDATED SUMMARY: Now includes temperature ---
    summary = {
        "symptoms": user_input,
        "temperature": temperature,
        "affected_area": affected_area,
        "severity": severity,
        "emergency": is_emergency,
        "advice": response.strip()
    }
    save_patient_summary(summary)
