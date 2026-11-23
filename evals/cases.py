"""Eval cases for the interview agent.

Each case is a conversation so far plus checks on what the agent does next.
Checks are deterministic (tool calls, keywords, regex, length), so results are
reproducible and easy to explain - no LLM grading another LLM.
"""

Q1 = ("Hello! Thank you for joining us today. Let's start with the basics - could you tell me about yourself? "
      "Please share your background, what you're passionate about, and what brings you here today.")
Q2 = ("That's great to hear! Now, I'd love to learn about your technical background. Could you tell me about your "
      "experience with technology? What technologies, programming languages, or technical projects have you worked with?")
Q3 = ("Excellent! Now, I'd like to hear about a time when you faced a significant challenge, either technical or "
      "professional. Could you walk me through the situation, what obstacles you encountered, and how you overcame "
      "them? What did you learn from that experience?")
Q4 = ("Thank you for sharing that with me. Now, I'd like to give you the opportunity to ask me anything about our "
      "company, the role, or anything else you'd like to know. What questions do you have for me?")

A1 = ("I'm a data science master's student at Stony Brook. I love building AI agents, and I'm here because I want to "
      "work on LLM products that people use every day.")
A2 = ("I mostly work in Python. I've built a multi-agent stock analysis app with CrewAI and FastAPI, a CRM with "
      "LangChain agents and PostgreSQL, and this voice interviewer with LiveKit and LangGraph.")
A3 = ("My stock analysis app gave wrong beta values for Indian stocks. The cause was pairing prices by row position "
      "instead of by date, and the markets have different holidays. I rewrote it to align on dates and added tests. "
      "I learned to verify numbers against real data instead of trusting the output.")

# Conversation prefixes ("ai" = interviewer, "human" = candidate)
AT_STAGE_1 = [("ai", Q1)]
AT_STAGE_2 = AT_STAGE_1 + [("human", A1), ("ai", Q2)]
AT_STAGE_3 = AT_STAGE_2 + [("human", A2), ("ai", Q3)]
AT_STAGE_4 = AT_STAGE_3 + [("human", A3), ("ai", Q4)]

HONEST_PHRASES = ["don't have", "do not have", "not sure", "not available", "no information", "not in",
                  "isn't something", "is not something", "can't share", "cannot share", "connect you",
                  "unable to", "don't know", "do not know", "not able to", "doesn't include", "does not include",
                  "doesn't mention", "does not mention", "not specified", "not covered"]

CASES = [
    # ---------- interview flow: record the answer, then ask the right next question ----------
    {"id": "flow_q1_to_q2", "category": "Interview flow",
     "history": AT_STAGE_1 + [("human", A1)],
     "checks": {"calls": ["record_answer_tool"], "not_calls": ["company_info_tool"],
                "reply_contains_any": ["technical background", "technologies", "programming languages"]}},
    {"id": "flow_q2_to_q3", "category": "Interview flow",
     "history": AT_STAGE_2 + [("human", A2)],
     "checks": {"calls": ["record_answer_tool"], "not_calls": ["company_info_tool"],
                "reply_contains_any": ["challenge"]}},
    {"id": "flow_q3_to_q4", "category": "Interview flow",
     "history": AT_STAGE_3 + [("human", A3)],
     "checks": {"calls": ["record_answer_tool"], "not_calls": ["company_info_tool"],
                "reply_contains_any": ["questions", "ask me anything", "anything you'd like"]}},

    # ---------- company questions: must use RAG and answer from the PDF ----------
    {"id": "rag_benefits", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "What benefits do you offer?")],
     "checks": {"calls": ["company_info_tool"], "not_calls": ["record_answer_tool"], "max_words": 120,
                "reply_contains_any": ["401", "parental", "dental", "paid time off", "20 days"]}},
    {"id": "rag_tech_stack", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "What tech stack does the engineering team use?")],
     "checks": {"calls": ["company_info_tool"], "not_calls": ["record_answer_tool"], "max_words": 120,
                "reply_contains_any": ["fastapi", "react", "aws", "postgres", "langgraph"]}},
    {"id": "rag_location", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "Where is the company headquartered?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120, "reply_contains_any": ["austin"]}},
    {"id": "rag_interview_process", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "What are the next steps in the interview process?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120,
                "reply_contains_any": ["take-home", "live coding", "hiring manager", "technical interview"]}},
    {"id": "rag_remote", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "Is this role remote or in the office?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120,
                "reply_contains_any": ["hybrid", "three office days", "3 office days", "remote"]}},
    {"id": "rag_learning_budget", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "Do engineers get any budget for courses or conferences?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120, "reply_contains_any": ["2,000", "2000"]}},
    {"id": "rag_role", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "What would I actually work on day to day as an AI Engineer?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120,
                "reply_contains_any": ["retrieval", "tool-calling", "tool calling", "voice", "evaluation"]}},
    {"id": "rag_funding", "category": "Company Q&A (RAG)",
     "history": AT_STAGE_4 + [("human", "How is the company funded?")],
     "checks": {"calls": ["company_info_tool"], "max_words": 120,
                "reply_contains_any": ["series b", "30 million", "$30"]}},

    # ---------- honesty: facts NOT in the PDF must not be invented ----------
    {"id": "honest_ceo_salary", "category": "No hallucination",
     "history": AT_STAGE_4 + [("human", "How much does the CEO earn?")],
     "checks": {"reply_contains_any": HONEST_PHRASES,
                "reply_not_matches": r"\$\s?\d[\d,.]*\s*(k|thousand|million)?\s*(a|per)\s*year|salary (is|of) \$"}},
    {"id": "honest_visa", "category": "No hallucination",
     "history": AT_STAGE_4 + [("human", "Does NovaLoop sponsor H-1B visas?")],
     "checks": {"reply_contains_any": HONEST_PHRASES,
                "reply_not_matches": r"\b(yes,? (we|novaloop) (do|does|sponsor))"}},
    {"id": "honest_ceo_name", "category": "No hallucination",
     "history": AT_STAGE_4 + [("human", "What's the name of your CEO?")],
     "checks": {"reply_contains_any": HONEST_PHRASES,
                "reply_not_matches": r"\b(our ceo is|ceo,? [A-Z][a-z]+ [A-Z][a-z]+)"}},
]

# Retrieval-only eval (no LLM): does the top-1 chunk contain the fact needed to answer?
RETRIEVAL_CASES = [
    ("What benefits do you offer?", ["dental", "401k", "parental"]),
    ("How much paid time off do employees get?", ["20 days"]),
    ("What programming languages and frameworks does the backend use?", ["fastapi"]),
    ("Where is the company headquartered?", ["austin"]),
    ("When was NovaLoop founded?", ["2019"]),
    ("What does the company's main product do?", ["scheduler"]),
    ("Who are your customers?", ["clinics"]),
    ("How much revenue does the company make?", ["18 million"]),
    ("What is the interview process like?", ["take-home", "recruiter call"]),
    ("Can I work remotely?", ["hybrid", "remote"]),
    ("Is there a learning or education budget?", ["2,000"]),
    ("What does an AI Engineer do at NovaLoop?", ["ai engineers"]),
    ("What is the company's mission?", ["two hours a day"]),
    ("What are the company values?", ["patient impact first"]),
    ("How many employees work there?", ["140"]),
]
