# graph.py
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Resolve paths relative to this file, so the app works no matter which folder you run it from
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env.local")


class LocalEmbeddings(Embeddings):
    """Free, local embeddings (all-MiniLM-L6-v2 via ChromaDB's built-in ONNX model).
    No API key needed; the ~80 MB model downloads once on first run."""

    def __init__(self) -> None:
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_llm():
    """DeepSeek chat model. Its API is OpenAI-compatible, so ChatOpenAI works with a different base_url."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env.local.")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
    )


# Retrieval settings. Defaults are the original ones; tune them with `evals/run_evals.py --sweep`
# and override with RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP / RAG_TOP_K in .env.local.
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def build_retriever(embeddings=None, pdf_path=None, persist_directory=None, k=None,
                    chunk_size=None, chunk_overlap=None):
    """Load the company PDF, split it into chunks, index it in ChromaDB and return a retriever."""
    k = k or DEFAULT_TOP_K
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = DEFAULT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    embeddings = embeddings or LocalEmbeddings()

    # this is how the interviewer will know about the company it works for
    pdf_path = pdf_path or os.getenv("COMPANY_PDF_PATH", str(BASE_DIR / "company_profile.pdf"))
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}. Set COMPANY_PDF_PATH in .env.local or put company_profile.pdf in the project folder.")

    pages = PyPDFLoader(pdf_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pages_split = text_splitter.split_documents(pages)

    persist_directory = persist_directory or os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_store"))
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
         persist_directory=persist_directory,
         embedding_function=embeddings,
         # one collection per chunking setting, so changing the settings never reuses a stale index
         collection_name=f"company_info_local_{chunk_size}_{chunk_overlap}",)

    # Only add the PDF if the database is empty
    if vectorstore._collection.count() == 0:
         vectorstore.add_documents(pages_split)

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


# -------------------- Build your Interview RAG pipeline --------------------
def create_workflow(llm=None, retriever=None, answers_path=None):
    """Build the interview graph. Every dependency can be injected, so tests can use a fake LLM
    and retriever, and evals can write answers to a temporary file instead of the real log."""
    llm = llm or build_llm()
    retriever = retriever or build_retriever()
    answers_path = Path(answers_path or BASE_DIR / "interview_answers.txt")

    @tool # tool to search company info
    def company_info_tool(query: str) -> str:
        """Searches the company information document and returns relevant chunks about the company."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the company documents."

        # return every retrieved chunk (top-k), not just the first one
        return "\n\n---\n\n".join(d.page_content for d in docs)
        
    @tool # tool to record answers
    def record_answer_tool(answer: str) -> str:
        """Records the candidate's answer to a text file for later review."""
        # Simply write the answer to the file
        with open(answers_path, "a", encoding="utf-8") as f:
            f.write(f"\nAnswer:\n{answer}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Recorded answer: {answer[:50]}...")
        return "Answer recorded successfully!"

    tools = [company_info_tool, record_answer_tool] # allows the llm to use the tools
    llm = llm.bind_tools(tools)

    class InterviewState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def decide_next_action(state: InterviewState) -> str:
        """Decide what to do next: tool_executor or end"""
        last = state["messages"][-1]
        
        # Check if we need to execute tool calls from the LLM
        if hasattr(last, "tool_calls") and last.tool_calls and len(last.tool_calls) > 0:
            return "tool_executor"
        
        # Default to end if no specific action needed
        return "end"

    def call_llm(state: InterviewState) -> InterviewState: # function that tells the llm what to do
        """Main LLM call that handles the interview conversation."""
        system_prompt = (
              "You are a professional interviewer conducting a job interview. "
            "You will ask structured questions in this order:\n"
            "1. First: 'Hello! Thank you for joining us today. Let's start with the basics - could you tell me about yourself? Please share your background, what you're passionate about, and what brings you here today.'\n"
            "2. After they respond: 'That's great to hear! Now, I'd love to learn about your technical background. Could you tell me about your experience with technology? What technologies, programming languages, or technical projects have you worked with?'\n"
            "3. After they respond: 'Excellent! Now, I'd like to hear about a time when you faced a significant challenge, either technical or professional. Could you walk me through the situation, what obstacles you encountered, and how you overcame them? What did you learn from that experience?'\n"
            "4. After they respond: 'Thank you for sharing that with me. Now, I'd like to give you the opportunity to ask me anything about our company, the role, or anything else you'd like to know. What questions do you have for me?'\n\n"
            "IMPORTANT ROUTING RULES:\n"
            "- When the candidate asks questions about the company (mission, culture, revenue, etc.), use the company_info_tool to find relevant information\n"
            "- Do not repeat the entire company document. Give a concise answer using only the relevant information from the company_info_tool\n"
            "- After EVERY candidate answer to one of the 4 interview questions, you MUST call record_answer_tool first, before acknowledging the answer or asking the next question. Never skip this step, even if you are confident about what to ask next\n"
            "- This is a spoken conversation: keep every reply under 80 words and answer only what was asked\n"
            "- Be conversational, professional, and helpful\n"
            "- If you don't have specific company information, say so honestly and offer to connect them with someone who might know more"
        )
        
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(msgs)
        return {"messages": [message]}

    def tool_executor(state: InterviewState) -> InterviewState: # function that actually executes the tools
        """Execute tool calls from the LLM's response."""
        # Get the tool calls from the last message
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # Go through each tool call one by one
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            print(f"Running tool: {tool_name}")

            # Call the right tool based on its name
            if tool_name == "company_info_tool":
                result = company_info_tool.invoke(tool_args)
            elif tool_name == "record_answer_tool":
                result = record_answer_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            # Create a message with the result
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=str(result),
            )
            results.append(tool_message)

        print("All tools finished running.")
        return {"messages": results}


    # Build the interview graph
    graph = StateGraph(InterviewState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", tool_executor)
    
    # Set up the flow
    graph.set_entry_point("llm")
    
    # Conditional edges from LLM
    graph.add_conditional_edges(
        "llm", 
        decide_next_action, 
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )
    
    # From tool_executor back to LLM
    graph.add_edge("tool_executor", "llm")

    return graph.compile()