Telemedicine RAG Chatbot
Overview
This project is a proof-of-concept for an offline, local Telemedicine AI chatbot. It uses a Retrieval-Augmented Generation (RAG) architecture to provide initial medical advice based on user-described symptoms. The system is designed to run entirely on your local machine without needing an internet connection, ensuring data privacy.

The chatbot can differentiate between emergency and non-emergency situations, retrieve relevant information from a local knowledge base, and instruct a Large Language Model (LLM) to generate appropriate and safe advice.

Key Features
Offline & Private: All processing, from embedding generation to language model inference, happens locally. No patient data ever leaves your computer.

RAG Architecture: Uses FAISS vector search to find the most relevant medical information from local text files before generating a response, making the advice more accurate and grounded.

Emergency Triage: Automatically detects high-severity symptoms or keywords (e.g., "chest pain") and switches to an emergency-specific knowledge base.

Simulated SOS Function: In a detected emergency, the system simulates triggering an in-app SOS call to emergency services.

Context-Aware Questions: Dynamically asks for more information when needed, such as asking for a specific temperature if the user mentions "fever".

Symptom Logging: Saves a summary of each interaction (symptoms, severity, advice) to a patient_summary.jsonl file for record-keeping.

How It Works
The system follows a simple but powerful workflow:

User Input: The user describes their symptoms.

Triage: The system assesses the input for emergency keywords and user-provided severity.

Vector Search: The user's query is converted into a vector embedding. This vector is used to search a pre-built FAISS index (either normal or emergency) to find the most similar and relevant text chunks from the local data folders.

Prompt Augmentation: The retrieved text chunks are combined with the user's original query and a strict system prompt.

LLM Generation: This complete "augmented" prompt is sent to the local GPT4All (orca-mini-1b) model.

Actionable Advice: The LLM generates a concise, direct response based on the provided context, which is then displayed to the user.

Project Structure
telemedicine-rag/
│
├── data/
│   └── (add your .txt files for non-emergency info here)
│
├── data_emergency/
│   └── (add your .txt files for emergency info here)
│
├── models/
│   └── orca-mini-1b.gguf  (place the downloaded model here)
│
├── telemedicine_rag_offline.py   # The main application script
├── patient_summary.jsonl       # Log file, created after first run
├── README.md                   # This file
└── requirements.txt            # Python dependencies

Setup and Installation
Clone or Download: Get the project files onto your local machine.

Create a Virtual Environment: It's highly recommended to use a virtual environment.

python -m venv tele_env
source tele_env/bin/activate  # On Windows, use: tele_env\Scripts\activate

Install Dependencies: Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

Download the LLM:

Download the orca-mini-1b.gguf model file. You can find it on the official GPT4All website or other model repositories.

Place the downloaded .gguf file inside the models directory.

Create Knowledge Base:

Create the data and data_emergency folders if they don't exist.

Inside each folder, add one or more .txt files containing medical information. Each file should cover a specific topic (e.g., headaches.txt, burns.txt). The script will read all .txt files in these folders.

How to Run
Once the setup is complete, run the script from your terminal:

python telemedicine_rag_offline.py

The script will first build the FAISS indices from your text files (this may take a moment) and then load the model. Once you see the "Telemedicine system loaded" message, you can start interacting with it.