"""Unit tests for the LangGraph interview workflow (graph.py)."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import graph
from conftest import FakeLLM, FakeRetriever, KeywordEmbeddings, ai_tool_calls, tool_call

PDF = Path(graph.__file__).parent / "company_profile.pdf"


def run(llm, retriever=None, answers_path=None, user_text="Hello"):
    wf = graph.create_workflow(llm=llm, retriever=retriever or FakeRetriever(), answers_path=answers_path)
    return wf.invoke({"messages": [HumanMessage(content=user_text)]})["messages"]


# ---------- graph structure ----------

def test_graph_has_llm_and_tool_nodes(answers_file):
    wf = graph.create_workflow(llm=FakeLLM([]), retriever=FakeRetriever(), answers_path=answers_file)
    assert {"llm", "tool_executor"} <= set(wf.get_graph().nodes)


def test_both_tools_are_bound_to_the_llm(answers_file):
    llm = FakeLLM([])
    graph.create_workflow(llm=llm, retriever=FakeRetriever(), answers_path=answers_file)
    assert sorted(t.name for t in llm.bound_tools) == ["company_info_tool", "record_answer_tool"]


# ---------- routing ----------

def test_plain_reply_ends_without_tools(answers_file):
    llm = FakeLLM([AIMessage(content="Could you tell me about yourself?")])
    messages = run(llm, answers_path=answers_file)
    assert messages[-1].content == "Could you tell me about yourself?"
    assert not any(isinstance(m, ToolMessage) for m in messages)
    assert len(llm.calls) == 1


def test_system_prompt_is_sent_first_with_the_question_order(answers_file):
    llm = FakeLLM([AIMessage(content="ok")])
    run(llm, answers_path=answers_file)
    first = llm.calls[0][0]
    assert isinstance(first, SystemMessage)
    for stage in ["1. First", "2. After they respond", "3. After they respond", "4. After they respond"]:
        assert stage in first.content


def test_company_question_runs_retrieval_and_loops_back_to_llm(answers_file):
    retriever = FakeRetriever()
    llm = FakeLLM([
        ai_tool_calls(tool_call("company_info_tool", {"query": "paid time off"})),
        AIMessage(content="You get 20 days of paid time off."),
    ])
    messages = run(llm, retriever=retriever, answers_path=answers_file, user_text="How much PTO?")

    assert retriever.queries == ["paid time off"]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and "20 days" in tool_msgs[0].content
    assert tool_msgs[0].tool_call_id == "call_1"
    # the second LLM call must see the tool result
    assert any(isinstance(m, ToolMessage) for m in llm.calls[1])
    assert messages[-1].content == "You get 20 days of paid time off."


def test_empty_retrieval_returns_a_clear_message(answers_file):
    llm = FakeLLM([
        ai_tool_calls(tool_call("company_info_tool", {"query": "CEO salary"})),
        AIMessage(content="I don't have that information."),
    ])
    messages = run(llm, retriever=FakeRetriever(docs=[]), answers_path=answers_file)
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.content == "No relevant information found in the company documents."


def test_record_answer_appends_to_the_answers_file(answers_file):
    llm = FakeLLM([
        ai_tool_calls(tool_call("record_answer_tool", {"answer": "I build LLM agents in Python."})),
        AIMessage(content="Great! Next question..."),
    ])
    run(llm, answers_path=answers_file)
    run(FakeLLM([
        ai_tool_calls(tool_call("record_answer_tool", {"answer": "Second answer"})),
        AIMessage(content="Thanks."),
    ]), answers_path=answers_file)

    text = answers_file.read_text(encoding="utf-8")
    assert "I build LLM agents in Python." in text
    assert "Second answer" in text
    assert text.index("I build") < text.index("Second answer")  # appended, not overwritten


def test_multiple_tool_calls_in_one_turn_are_all_executed(answers_file):
    retriever = FakeRetriever()
    llm = FakeLLM([
        ai_tool_calls(
            tool_call("record_answer_tool", {"answer": "My answer"}, call_id="a"),
            tool_call("company_info_tool", {"query": "benefits"}, call_id="b"),
        ),
        AIMessage(content="Done."),
    ])
    messages = run(llm, retriever=retriever, answers_path=answers_file)
    ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert ids == ["a", "b"]
    assert retriever.queries == ["benefits"]
    assert "My answer" in answers_file.read_text()


def test_unknown_tool_name_does_not_crash(answers_file):
    llm = FakeLLM([
        ai_tool_calls(tool_call("delete_database", {})),
        AIMessage(content="Sorry about that."),
    ])
    messages = run(llm, answers_path=answers_file)
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.content == "Unknown tool: delete_database"
    assert messages[-1].content == "Sorry about that."


# ---------- configuration and errors ----------

def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        graph.build_llm()


def test_llm_uses_deepseek_endpoint_and_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")
    llm = graph.build_llm()
    assert llm.model_name == "deepseek-test"
    assert "api.deepseek.com" in str(llm.openai_api_base)


def test_missing_pdf_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="COMPANY_PDF_PATH"):
        graph.build_retriever(embeddings=KeywordEmbeddings(), pdf_path=str(tmp_path / "nope.pdf"),
                              persist_directory=str(tmp_path / "db"))


# ---------- retrieval over the real company PDF (offline embeddings) ----------

def test_pdf_is_indexed_and_retrievable(tmp_path):
    retriever = graph.build_retriever(embeddings=KeywordEmbeddings(), pdf_path=str(PDF),
                                      persist_directory=str(tmp_path / "db"))
    docs = retriever.invoke("What benefits like dental insurance and parental leave do you offer?")
    assert len(docs) == 1
    assert "dental" in docs[0].page_content.lower()


def test_reindexing_does_not_duplicate_chunks(tmp_path):
    db = str(tmp_path / "db")
    first = graph.build_retriever(embeddings=KeywordEmbeddings(), pdf_path=str(PDF), persist_directory=db)
    count = first.vectorstore._collection.count()
    second = graph.build_retriever(embeddings=KeywordEmbeddings(), pdf_path=str(PDF), persist_directory=db)
    assert count > 0
    assert second.vectorstore._collection.count() == count
