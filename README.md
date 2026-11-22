# AI Voice Interview Agent

A real-time, voice-driven AI interviewer built on **LiveKit Agents**, **LangGraph**, and **Retrieval-Augmented Generation (RAG)**. Candidates have a natural spoken conversation with an AI interviewer — complete with a synced virtual avatar — that asks structured interview questions, listens and responds in real time, and can answer candidate questions about the company by grounding its answers in an internal company document.

## Features

- **Real-time voice conversation** over WebRTC via the LiveKit Agents framework
- **Structured interview flow** orchestrated with LangGraph — introduction → technical background → behavioral/challenge question → open candidate Q&A
- **Retrieval-Augmented Generation (RAG)** pipeline (LangChain + local MiniLM embeddings + ChromaDB, no embedding API cost) that grounds company-related answers in an ingested company PDF instead of hallucinating
- **Autonomous tool use** — the LLM decides when to call:
  - `company_info_tool` to retrieve relevant company info
  - `record_answer_tool` to log the candidate's response
- **Full speech pipeline**: Deepgram (speech-to-text) → DeepSeek reasoning via LangGraph → Cartesia (text-to-speech), with Silero VAD and multilingual turn detection for natural back-and-forth timing
- **Background noise cancellation** (LiveKit BVC)
- **Optional animated avatar** via Beyond Presence, lip-synced to the agent's speech (voice-only by default)

## How It Works

1. A candidate joins a LiveKit room; the agent joins it (optionally alongside a Beyond Presence avatar).
2. Speech is transcribed live by Deepgram and streamed into a **LangGraph** state machine.
3. The graph's LLM node (DeepSeek) decides whether to respond directly, call `company_info_tool` (RAG lookup against the company PDF via ChromaDB), or call `record_answer_tool` to log the candidate's answer.
4. Responses are streamed to Cartesia for text-to-speech, played back through the room, and mirrored by the avatar.
5. The graph loops between the LLM node and its tool-executor node until the interview's four stages are complete.

## Project Structure

```
voice-interview-agent/
├── agent.py              # LiveKit entrypoint — voice pipeline, avatar, session wiring
├── graph.py               # LangGraph workflow — RAG tools, interview state machine
├── pyproject.toml         # Project metadata & dependencies (uv)
├── uv.lock                # Locked dependency versions
├── company_profile.pdf    # Fictional demo company used for RAG (NovaLoop Labs)
├── .env.example          # Template for .env.local
├── .env.local            # Local secrets (never committed — see below)
├── chroma_store/             # Persisted vector DB (generated at runtime)
├── interview_answers.txt      # Logged candidate answers (generated at runtime)
└── README.md
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A [LiveKit Cloud](https://cloud.livekit.io) project
- API keys for DeepSeek, Deepgram, and Cartesia (Beyond Presence only if you enable the avatar)

## Setup

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Configure environment variables** — copy the template and fill in your keys:
   ```bash
   cp .env.example .env.local
   ```
   Required: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `DEEPSEEK_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`.
   Optional: `COMPANY_PDF_PATH`, `CHROMA_DIR`, `AGENT_NAME` (explicit dispatch), `BEY_API_KEY` + `BEY_AVATAR_ID` (avatar).

3. **Company document** — a fictional `company_profile.pdf` is included. To use your own, set `COMPANY_PDF_PATH` and delete `chroma_store/` so it re-indexes. This is what the RAG pipeline retrieves from when candidates ask company questions.

4. **Download model files (first time only), then run the agent**
   ```bash
   uv run agent.py download-files
   uv run agent.py dev
   ```

5. **Test it** — in your LiveKit Cloud project, go to **Agents → Console** and click **Start a session** to talk to the agent live with transcripts, events, and metrics.

## Known Limitations / Future Improvements

- Joins every new room by default; set `AGENT_NAME` to switch to explicit dispatch for controlled interview sessions.
- Supports a single company PDF at a time; no multi-tenant/multi-company support yet.
- Candidate answers are logged in plaintext to `interview_answers.txt` — fine for a personal/demo project, but would need proper storage and consent handling before any real candidate data touches it.
- No persistent session/resume support if a candidate disconnects mid-interview.
- Potential roadmap: a simple web front-end for candidates, a reviewer dashboard for transcripts, and configurable question sets per role.

## License

Released under the [MIT License](LICENSE). Note that running this project still requires your own paid API keys for LiveKit, DeepSeek, Deepgram, Cartesia, and (optionally) Beyond Presence — the license covers the code, not access to those services.


