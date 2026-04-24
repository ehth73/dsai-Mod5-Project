"""Hugging Face Spaces entrypoint for the Bank Contact Centre LangGraph app.

Upload this folder to a Hugging Face Space with SDK = Gradio.
Set GROQ_API_KEY as a Space secret.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import chromadb
import gradio as gr

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader, UnstructuredFileLoader
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "bank_contact_center_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {"source_data_dir": "input_files", "db_path": "chroma_db"},
    "models": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "groq_model": "llama-3.1-8b-instant",
        "temperature": 0.2,
    },
    "retrieval": {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 3, "rebuild_on_startup": True},
    "web_search": {
        "enabled": True,
        "max_results": 5,
        "hints": ["today", "latest", "recent", "current", "news", "mas", "outage", "scam alert"],
    },
    "agents": {
        "cards_fraud_agent": {
            "collection_name": "cards_fraud_collection",
            "persona": "Cards & Fraud Specialist",
            "description": "Lost cards, card fraud, scams, OTP compromise, unauthorised transactions, card blocking and urgent containment.",
        },
        "payments_agent": {
            "collection_name": "payments_collection",
            "persona": "Payments and Transfers Specialist",
            "description": "FAST, GIRO, PayNow, remittance, transfer delays, payment tracing, and payment recovery.",
        },
        "complaints_agent": {
            "collection_name": "complaints_collection",
            "persona": "Complaints and Service Recovery Specialist",
            "description": "Complaints, service recovery, branch dissatisfaction, rude service, escalation handling.",
        },
        "service_agent": {
            "collection_name": "service_collection",
            "persona": "Bank Products and Account Servicing Specialist",
            "description": "Accounts, cards, loans, deposits, fees, account servicing, and general banking product questions.",
        },
        "general_agent": {
            "collection_name": "general_collection",
            "persona": "General Banking Assistant",
            "description": "Fallback for greetings, broad questions, and topics outside the supported bank contact centre domains.",
        },
    },
    "classification_rules": {
        "cards_fraud_agent": ["card", "fraud", "scam", "otp", "phishing", "stolen", "unauthorised", "chargeback"],
        "payments_agent": ["payment", "transfer", "fast", "giro", "paynow", "remittance", "beneficiary", "swift"],
        "complaints_agent": ["complaint", "escalation", "service recovery", "rude", "dissatisfaction", "feedback"],
        "service_agent": ["account", "loan", "deposit", "fixed deposit", "mortgage", "balance", "statement", "product", "servicing"],
    },
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


cfg = load_config()
SOURCE_DATA_DIR = APP_DIR / cfg["paths"]["source_data_dir"]
DB_PATH = APP_DIR / cfg["paths"]["db_path"]
SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.mkdir(parents=True, exist_ok=True)

TOP_K = int(cfg.get("retrieval", {}).get("top_k", 3))
AGENT_ORDER = ["cards_fraud_agent", "payments_agent", "complaints_agent", "service_agent", "general_agent"]
VALID_ROUTES = list(cfg["agents"].keys())
WEB_SEARCH_HINTS = [h.lower() for h in cfg.get("web_search", {}).get("hints", [])]
WEB_SEARCH_ENABLED = bool(cfg.get("web_search", {}).get("enabled", True))


def build_llm() -> ChatGroq | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    return ChatGroq(
        model=cfg["models"].get("groq_model", "llama-3.1-8b-instant"),
        temperature=float(cfg["models"].get("temperature", 0.2)),
    )


llm = build_llm()
embeddings = HuggingFaceEmbeddings(model_name=cfg["models"].get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
persistent_client = chromadb.PersistentClient(path=str(DB_PATH))
splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(cfg["retrieval"].get("chunk_size", 1000)),
    chunk_overlap=int(cfg["retrieval"].get("chunk_overlap", 200)),
)
ddg_search_results = DuckDuckGoSearchResults(
    output_format="list",
    max_results=int(cfg.get("web_search", {}).get("max_results", 5)),
)


def classify_collection(filename: str) -> str:
    fname = filename.lower()
    for agent_name in AGENT_ORDER:
        for keyword in cfg.get("classification_rules", {}).get(agent_name, []):
            if keyword.lower() in fname:
                return cfg["agents"][agent_name]["collection_name"]
    return cfg["agents"]["general_agent"]["collection_name"]


def load_documents_from_folder(folder_path: Path):
    docs = []
    for path in folder_path.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            suffix = path.suffix.lower()
            if suffix in {".txt", ".md"}:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            elif suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".csv":
                loaded = CSVLoader(str(path), encoding="utf-8").load()
            else:
                loaded = UnstructuredFileLoader(str(path)).load()
            docs.extend(loaded)
        except Exception as exc:
            print(f"Could not load {path.name}: {exc}")
    return docs


def reset_selected_collections() -> None:
    for agent_name in cfg["agents"]:
        collection_name = cfg["agents"][agent_name]["collection_name"]
        try:
            persistent_client.delete_collection(collection_name)
        except Exception:
            pass


def build_knowledge_base() -> str:
    docs = load_documents_from_folder(SOURCE_DATA_DIR)
    if not docs:
        return "No files found in input_files. Add knowledge files and restart the Space."

    reset_selected_collections()
    bucketed_docs = {cfg["agents"][a]["collection_name"]: [] for a in cfg["agents"]}

    for doc in docs:
        source = doc.metadata.get("source", "unknown.txt")
        filename = Path(source).name
        collection_name = classify_collection(filename)
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source_file"] = filename
            chunk.metadata["collection_name"] = collection_name
        bucketed_docs[collection_name].extend(chunks)

    counts = []
    for collection_name, chunk_list in bucketed_docs.items():
        if chunk_list:
            Chroma.from_documents(
                documents=chunk_list,
                embedding=embeddings,
                client=persistent_client,
                collection_name=collection_name,
            )
            counts.append(f"{collection_name}: {len(chunk_list)} chunks")
    return "Knowledge base ready. " + "; ".join(counts)


KB_STATUS = build_knowledge_base() if cfg.get("retrieval", {}).get("rebuild_on_startup", True) else "Startup rebuild disabled."
print(KB_STATUS)


def should_use_web_search(query: str) -> bool:
    q = query.lower()
    return WEB_SEARCH_ENABLED and any(hint in q for hint in WEB_SEARCH_HINTS)


def format_docs_for_answer(docs) -> str:
    if not docs:
        return "No internal documents found."
    formatted = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source_file", "unknown")
        formatted.append(f"[{i}] Source: {src}\n{doc.page_content}")
    return "\n\n".join(formatted)


def retrieve_internal_context(collection_name: str, query: str, k: int = TOP_K):
    try:
        db = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=embeddings)
        return db.similarity_search(query, k=k)
    except Exception as exc:
        print(f"Retrieval error from {collection_name}: {exc}")
        return []


def duckduckgo_search(query: str) -> str:
    if not should_use_web_search(query):
        return "No external web context used."
    try:
        raw_results = ddg_search_results.invoke(query)
        if isinstance(raw_results, str):
            return raw_results
        if not raw_results:
            return "No web results found."
        lines = []
        for idx, item in enumerate(raw_results[: int(cfg["web_search"].get("max_results", 5))], start=1):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            lines.append(f"[{idx}] {title}\nSnippet: {snippet}\nLink: {link}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"


class AgentState(TypedDict):
    query: str
    response: str
    next_node: str
    debug_log: str


def build_router_prompt() -> str:
    labels = "\n".join(f"- {name} = {agent_cfg['description']}" for name, agent_cfg in cfg["agents"].items())
    return (
        "You are a bank contact centre routing classifier.\n\n"
        "Classify the user query into exactly one of these labels:\n"
        f"{labels}\n\n"
        "Return ONLY one label from:\n"
        + "\n".join(VALID_ROUTES)
        + "\n\nIf unsure, return service_agent."
    )


ROUTER_PROMPT = build_router_prompt()


def keyword_route(query: str) -> str:
    q = query.lower()
    for agent_name in AGENT_ORDER:
        for keyword in cfg.get("classification_rules", {}).get(agent_name, []):
            if keyword.lower() in q:
                return agent_name
    return "service_agent"


def router_node(state: AgentState) -> Dict[str, str]:
    query = state["query"]
    decision = keyword_route(query)
    if llm is not None:
        try:
            response = llm.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=query)])
            candidate = response.content.strip().lower().replace('"', "").replace("'", "").strip(".")
            if candidate in VALID_ROUTES:
                decision = candidate
        except Exception:
            pass
    return {"next_node": decision, "debug_log": f"router: {decision}"}


def run_grounded_agent(state: AgentState, agent_name: str) -> Dict[str, str]:
    if llm is None:
        return {
            "response": "GROQ_API_KEY is not configured. Add it as a Hugging Face Space secret, then restart the Space.",
            "debug_log": state.get("debug_log", "") + f" | {agent_name}: missing GROQ_API_KEY",
        }

    query = state["query"]
    agent_cfg = cfg["agents"][agent_name]
    collection_name = agent_cfg["collection_name"]
    docs = retrieve_internal_context(collection_name, query, k=TOP_K)
    internal_context = format_docs_for_answer(docs) if docs else "No internal context found."
    use_web = should_use_web_search(query)
    external_context = duckduckgo_search(query) if use_web else "No external web context used."

    system_prompt = f"""
You are the {agent_cfg['persona']}.

Answering policy:
1. Prioritise INTERNAL CONTEXT for bank procedures, scripts, controls, and service actions.
2. Use EXTERNAL WEB CONTEXT only for public, current, or external information explicitly relevant to the user's question.
3. Do not invent policies, timelines, liabilities, fees, guarantees, or approval outcomes.
4. If internal policy is missing, say that clearly.
5. If web context is used, clearly label it as public web information.
6. Give practical next steps suitable for a bank contact centre.
7. Keep the answer concise, safe, and customer-friendly.

INTERNAL CONTEXT:
{internal_context}

EXTERNAL WEB CONTEXT:
{external_context}
"""
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
        answer = response.content
    except Exception as exc:
        answer = f"The model call failed: {exc}"

    debug_log = " | ".join(
        bit
        for bit in [
            state.get("debug_log", ""),
            f"{agent_name}: collection={collection_name}",
            f"{agent_name}: internal_chunks={len(docs)}",
            f"{agent_name}: web_search_used={use_web}",
        ]
        if bit
    )
    return {"response": answer, "debug_log": debug_log}


def cards_fraud_agent_node(state: AgentState) -> Dict[str, str]:
    return run_grounded_agent(state, "cards_fraud_agent")


def payments_agent_node(state: AgentState) -> Dict[str, str]:
    return run_grounded_agent(state, "payments_agent")


def complaints_agent_node(state: AgentState) -> Dict[str, str]:
    return run_grounded_agent(state, "complaints_agent")


def service_agent_node(state: AgentState) -> Dict[str, str]:
    return run_grounded_agent(state, "service_agent")


def general_agent_node(state: AgentState) -> Dict[str, str]:
    return run_grounded_agent(state, "general_agent")


workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("cards_fraud_agent", cards_fraud_agent_node)
workflow.add_node("payments_agent", payments_agent_node)
workflow.add_node("complaints_agent", complaints_agent_node)
workflow.add_node("service_agent", service_agent_node)
workflow.add_node("general_agent", general_agent_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda state: state["next_node"])
for node in VALID_ROUTES:
    workflow.add_edge(node, END)
compiled_app = workflow.compile()


def chat_fn(message: str, history: List[Dict[str, str]] | None = None) -> str:
    if not message or not message.strip():
        return "Please enter a banking contact-centre question."
    result = compiled_app.invoke({"query": message, "response": "", "next_node": "", "debug_log": ""})
    answer = result.get("response", "No answer returned.")
    log = result.get("debug_log", "No debug log.")
    return f"{answer}\n\n<details><summary>Routing / Agent Trace</summary><pre>{log}</pre></details>"


with gr.Blocks(title="Bank Contact Centre Multi-Agent Demo") as demo:
    gr.Markdown("# 🏦 Bank Contact Centre Multi-Agent Demo")
    gr.Markdown(
        "Ask about cards, fraud, payments, complaints, accounts, deposits, loans, or current public banking information."
    )
    gr.Markdown(f"**Startup status:** {KB_STATUS}")
    gr.ChatInterface(fn=chat_fn, type="messages")

if __name__ == "__main__":
    demo.launch()
