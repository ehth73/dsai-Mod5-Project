---
title: Bank Contact Centre Multi Agent Demo
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Bank Contact Centre Multi-Agent Demo

This Hugging Face Spaces project is adapted from the uploaded notebook `bank_contact_centre_langgraph_multi_agent_tools_config_ml_enhanced(3).ipynb`.

It hosts a Gradio app that uses:

- LangGraph for routing
- Groq LLM for answer generation
- Hugging Face sentence-transformer embeddings
- ChromaDB for internal document retrieval
- DuckDuckGo search for public/current information where relevant
- Config-driven specialist agents for cards/fraud, payments, complaints, service, and general banking support

## Files included

| File / folder | Purpose |
|---|---|
| `app.py` | Main Hugging Face Spaces entrypoint. Builds the RAG store, LangGraph router, agents, and Gradio UI. |
| `requirements.txt` | Python libraries installed by Hugging Face Spaces. |
| `packages.txt` | Linux packages for document parsing support. |
| `bank_contact_center_config.json` | Config for paths, model names, retrieval settings, web search hints, agents, and classification rules. |
| `input_files/` | Folder for internal bank knowledge files used by Chroma retrieval. |
| `model_artifacts/` | Optional folder for ML artifacts if you later add trained model inference. |
| `.gitignore` | Prevents local cache files and generated vector DB files from being committed. |

## Step-by-step deployment to Hugging Face Spaces

### Step 1 — Create a Hugging Face account

Go to Hugging Face and sign in or create an account.

### Step 2 — Create a new Space

1. Click **New Space**.
2. Enter a Space name, for example `bank-contact-centre-agent`.
3. Select **Gradio** as the SDK.
4. Select hardware. CPU Basic is acceptable for testing, but startup may be slower because embeddings are loaded on startup.
5. Choose public or private visibility.
6. Click **Create Space**.

### Step 3 — Upload all project files

Upload the complete contents of this folder to the Space:

```text
app.py
requirements.txt
packages.txt
bank_contact_center_config.json
README.md
.gitignore
input_files/
model_artifacts/
```

The minimum required files are:

```text
app.py
requirements.txt
bank_contact_center_config.json
input_files/
```

### Step 4 — Add your internal knowledge files

Put approved bank knowledge files into `input_files/`.

Recommended filenames:

```text
01_cards_and_fraud_playbook.md
02_payments_and_transfers_guide.txt
03_complaints_and_service_recovery.csv
04_products_and_accounts_faq.txt
05_regulatory_and_privacy_notes.md
```

The filename matters because the app uses filename keywords to route files into specialist Chroma collections.

### Step 5 — Add the Groq API key as a Space secret

In the Hugging Face Space:

1. Go to **Settings**.
2. Go to **Variables and secrets**.
3. Add a secret:

```text
Name: GROQ_API_KEY
Value: your_groq_api_key_here
```

Do not hard-code the key in `app.py`.

### Step 6 — Check the config file

Open `bank_contact_center_config.json` and confirm:

```json
"paths": {
  "source_data_dir": "input_files",
  "db_path": "chroma_db"
}
```

For Hugging Face Spaces, keep relative paths. Do not use Google Drive paths such as `/content/drive/...`.

### Step 7 — Let the Space build

After upload, Hugging Face Spaces will install dependencies from `requirements.txt`, install Linux packages from `packages.txt`, and start `app.py`.

On startup, the app will:

1. Load `bank_contact_center_config.json`
2. Load files from `input_files/`
3. Split the documents into chunks
4. Create Chroma collections
5. Build the LangGraph workflow
6. Launch the Gradio chat interface

### Step 8 — Test the app

Try these sample prompts:

```text
My debit card was stolen and I shared my OTP. What should I do?
```

```text
My FAST transfer is pending and the beneficiary has not received the funds. What should I do?
```

```text
I want to complain about rude service from a branch officer.
```

```text
Can I repay my personal loan early?
```

```text
Are there any recent public scam alerts in Singapore related to card phishing today?
```

The answer includes an expandable routing trace.

## How the app works

### 1. Ingestion

Files in `input_files/` are loaded and chunked. Each file is assigned to a Chroma collection based on filename keywords.

Examples:

- filenames containing `card`, `fraud`, `scam`, or `otp` go to `cards_fraud_collection`
- filenames containing `payment`, `transfer`, `fast`, `giro`, or `paynow` go to `payments_collection`
- filenames containing `complaint`, `escalation`, or `feedback` go to `complaints_collection`
- filenames containing `account`, `loan`, `deposit`, or `statement` go to `service_collection`

### 2. Routing

The router decides which specialist agent should answer the user question.

### 3. Retrieval

The selected agent retrieves relevant chunks from its Chroma collection.

### 4. Answer generation

The agent prompts Groq with internal context first. DuckDuckGo is only used when the question appears to need current or public information.

## Editing the agents

To change agent behaviour, edit `bank_contact_center_config.json`.

Important sections:

```json
"agents"
```

Controls the agent names, personas, collection names, and descriptions.

```json
"classification_rules"
```

Controls how document filenames are routed into collections.

```json
"web_search"
```

Controls whether public web search is enabled and which keywords trigger it.

## Updating the knowledge base

To update internal knowledge:

1. Upload new or revised files into `input_files/`.
2. Restart the Hugging Face Space.
3. The app rebuilds the Chroma vector store on startup because `rebuild_on_startup` is set to `true`.

## Optional ML model extension

The original notebook included a RandomForest anomaly model workflow for a call-centre database. This Space package focuses on hosting the multi-agent Gradio app. To add ML inference later:

1. Train the model locally or in the notebook.
2. Save the model into `model_artifacts/call_centre_anomaly_model.pkl`.
3. Add a scoring function in `app.py` using `joblib.load()`.
4. Add a Gradio tab for CSV upload and batch anomaly scoring.

## Common errors and fixes

### `GROQ_API_KEY is not configured`

Add `GROQ_API_KEY` as a Hugging Face Space secret and restart the Space.

### App starts but gives weak answers

Add stronger internal documents into `input_files/`. The sample files are only placeholders.

### Build fails on document parsing

Keep knowledge files as `.txt`, `.md`, or `.csv` first. Add `.pdf` and `.docx` only after the basic app works.

### Startup is slow

Use the smaller embedding model already configured:

```json
"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
```

This is lighter than the notebook's `all-mpnet-base-v2` model and better for CPU Spaces.
