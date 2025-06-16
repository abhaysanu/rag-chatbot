import os
import json
import numpy as np
from typing import List, Dict, Optional, TypedDict
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFaceHub
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class ChatTurn(BaseModel):
    user: str
    bot: str

class ChatInput(BaseModel):
    message: str
    chat_history: Optional[List[ChatTurn]] = []

#=== Define Agent State ===
class AgentState(TypedDict):
    messages: List[str]
    intent: str
    response: str
    chat_history: List[Dict[str, str]]
    chunks: List[str]

client = InferenceClient(
    provider="cohere",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

load_dotenv()

# === Load FAISS Index & Text Chunks ===
print("[INFO] Loading vector index and text chunks...")
index = faiss.read_index("vector.index")
with open("chunks.json", "r") as f:
    chunks = json.load(f)["chunks"]

embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("TEXT_EMBEDDING_MODEL"))
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# === Hugging Face Model Setup ===
print(f"Model: {os.getenv('MODEL')}")
print(f"Huggingface Token: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

@app.post("/chat")
def chat_endpoint(request: ChatInput):
    user_input = request.message.strip()

    # Safe initial state with chat_history fallback
    state = {
        "messages": [user_input],
        "intent": None,
        "chunks":None,
        "response": None,
        "chat_history": request.chat_history or [],  # Ensure it's at least an empty list
    }

    final_state = agent.invoke(state)
    return {"response": final_state["response"], "chat_history": final_state["chat_history"]}

def rephrase_question(state: AgentState) -> AgentState:
    user_input = state["messages"][-1]
    print(state)
    history = state["chat_history"]

    # Construct simple chat history string
    history_str = "\n".join([f"User: {h.user}\nBot: {h.bot}" for h in history[-3:]])

    prompt = f"""You are a chatbot assistant that rephrases follow-up questions using recent chat history.\n
Chat History:
{history_str}

Follow-up Question: {user_input}

Rephrased:"""

    result = query_llm(prompt)
    rephrased = result.strip()

    # Replace the current message with the rephrased one
    state["messages"][-1] = rephrased
    return state

def query_llm(prompt: str) -> str:
    print("Starting to query!")
    # pipe = pipeline(
    #     "text-generation", 
    #     model=model, 
    #     tokenizer=tokenizer, 
    #     token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    #     max_new_tokens=512,
    #     do_sample=True,
    #     temperature=0.7
    # )
    completion = client.chat.completions.create(
        model="CohereLabs/c4ai-command-r-plus",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    result = completion.choices[0].message.content
    return result

def detect_intent(state: AgentState) -> AgentState:
    print("Detect intent started!")
    user_input = state["messages"][-1]
    prompt = f"""You are part of a chatbot which givees information on sales data that in embedded in a vectordb. You are the first step in classifying the user input according to its intent.Classify this user input as one of [greet, fetch_info, unknown]. Respond with only one word, in lowercase, without any period.\n\nInput: {user_input}"""
    result = query_llm(prompt)
    intent = result.strip().lower()

    print(f"[DEBUG] Detected intent: {intent}")
    state["intent"] = intent
    return state


# === Handle Greetings ===
def handle_greet(state: AgentState) -> AgentState:
    user_input = state["messages"][-1]
    prompt = f"You're a polite assistant.\nRespond briefly to: {user_input}"
    result = query_llm(prompt)
    state["response"] = result.strip()
    return state

# === Handle Unknown ===
def handle_unknown(state: AgentState) -> AgentState:
    state["response"] = "Sorry, I couldn't understand your request. Could you please rephrase?"
    return state

# === Vector Search ===
def search_chunks(state: AgentState) -> AgentState:
    """
        Searching for relevant chunks as per the user query
    """
    query = state["messages"][-1] 
    results = vectorstore.similarity_search(query, k=3)  # Returns top3 relevant chunks
    print(f"Relevant Chunks: {results}")
    state["chunks"] = results[0].page_content
    return state

# === Generate Final Answer ===
def answer_query(state: AgentState) -> AgentState:
    """
        For creating a complete answer as per the retrieved chunks of text and based on the user prompt
    """
    user_input = state["messages"][-1]
    context = "\n\n".join(state.get("chunks") or [])
    print(f"Context: {context}")
    prompt = f"You are an assistant answering based on provided document content(chunks).Context:\n{context}\n\nUser query: {user_input}"
    result = query_llm(prompt)
    state["response"] = result.strip()
    return state

# === Save Chat History ===
def save_history(state: AgentState) -> AgentState:
    user_input = state["messages"][-1]
    response = state["response"]
    state["chat_history"].append({"user": user_input, "bot": response})
    return state

# === Build LangGraph ===
graph = StateGraph(AgentState)

graph.add_node("detect_intent", detect_intent)
graph.add_node("greet", handle_greet)
graph.add_node("search_chunks", search_chunks)
graph.add_node("answer_query", answer_query)
graph.add_node("unknown", handle_unknown)
graph.add_node("save_history", save_history)
graph.add_node("rephrase_question", rephrase_question)

graph.set_entry_point("rephrase_question")

graph.add_conditional_edges("detect_intent", lambda s: s["intent"], {
    "greet": "greet",
    "fetch_info": "search_chunks",
    "unknown": "unknown"
})

graph.add_edge("rephrase_question", "detect_intent")
graph.add_edge("greet", "save_history")
graph.add_edge("unknown", "save_history")
graph.add_edge("search_chunks", "answer_query")
graph.add_edge("answer_query", "save_history")
graph.add_edge("save_history", END)

agent = graph.compile()

