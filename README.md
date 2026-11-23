# AI Voice Interview Agent

A real-time AI interviewer you talk to out loud. You open a web page, click **Start interview**, and an AI interviewer greets you, works through a structured screening interview, and answers your questions about the company using retrieval over a company document instead of guessing.

**Stack:** Python · LiveKit Agents (WebRTC) · LangGraph · LangChain · ChromaDB (RAG) · DeepSeek · Deepgram · Cartesia · Next.js / React / TypeScript

## Features

- **Real-time voice conversation** over WebRTC with LiveKit Agents, including voice activity detection, multilingual turn detection and background noise cancellation
- **Structured interview flow** run by a LangGraph state machine: introduction → technical background → challenge question → candidate Q&A
- **Retrieval-Augmented Generation:** company questions are answered from an ingested PDF (LangChain + ChromaDB + local MiniLM embeddings, so retrieval has no embedding API cost)
- **Tool calling:** the LLM decides when to call `company_info_tool` (RAG lookup) or `record_answer_tool` (logs the candidate's answer for review)
- **Web frontend** (Next.js + React) with a start screen, live transcripts and an audio visualizer
- **Proactive greeting:** the interviewer opens the conversation as soon as the candidate joins
- **Optional animated avatar** via Beyond Presence (off by default)

## How It Works

```
Browser (Next.js) ──WebRTC──▶ LiveKit Cloud ◀──▶ Python agent
                                                   │
   mic audio ─▶ Deepgram STT ─▶ LangGraph (DeepSeek) ─▶ Cartesia TTS ─▶ speaker
                                    │
                                    ├─ company_info_tool ─▶ ChromaDB (company PDF)
                                    └─ record_answer_tool ─▶ interview_answers.txt
```

1. The frontend requests a token from its `/api/token` route and joins a LiveKit room.
2. The Python agent joins the same room and greets the candidate with the first question.
3. Deepgram transcribes the candidate's speech live, and the full conversation is passed to the LangGraph workflow.
4. The graph's LLM node (DeepSeek) either replies directly or calls a tool; tool results loop back to the LLM node.
5. Replies are spoken by Cartesia and streamed back to the browser with live transcripts.

## Project Structure

```
voice-interview-agent/
├── agent.py               # LiveKit entrypoint: voice pipeline, greeting, optional avatar
├── graph.py               # LangGraph workflow: RAG pipeline, tools, interview prompt
├── company_profile.pdf    # Fictional demo company (NovaLoop Labs) used for RAG
├── pyproject.toml         # Python dependencies (uv)
├── uv.lock
├── .env.example           # Template for .env.local
├── frontend/              # Next.js web app (start screen, live transcript, visualizer)
│   ├── app/               # Pages and the /api/token route
│   ├── components/        # UI components
│   └── package.json
├── chroma_store/          # Vector DB (generated at runtime, not committed)
└── interview_answers.txt  # Logged answers (generated at runtime, not committed)
```

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node.js 22+ and pnpm (`npm install -g pnpm@9`)
- A [LiveKit Cloud](https://cloud.livekit.io) project (free tier)
- API keys for DeepSeek, Deepgram and Cartesia (Deepgram and Cartesia have free tiers)

## Setup

1. **Install the agent's dependencies and download the voice models (first time only)**
   ```bash
   uv sync
   uv run agent.py download-files
   ```

2. **Add your keys.** Copy the template and fill it in:
   ```bash
   cp .env.example .env.local
   ```
   Required: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `DEEPSEEK_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`.
   Optional: `COMPANY_PDF_PATH`, `CHROMA_DIR`, `AGENT_NAME` (explicit dispatch), `CARTESIA_MODEL`, `CARTESIA_VOICE_ID`, `BEY_API_KEY` + `BEY_AVATAR_ID` (avatar).

3. **Set up the frontend.** It only needs the three LiveKit values:
   ```bash
   cd frontend
   pnpm install
   grep '^LIVEKIT_' ../.env.local > .env.local
   ```

## Running

Use two terminals:

```bash
# Terminal 1: the agent
uv run agent.py dev

# Terminal 2: the website
cd frontend && pnpm dev
```

Open **http://localhost:3000**, click **Start interview** and allow the microphone. Use `localhost` rather than the network address, because browsers only allow microphone access on secure origins.

To try the agent without the website, run `uv run agent.py console` and talk through your computer's microphone.

**Using your own company document:** set `COMPANY_PDF_PATH` in `.env.local` and delete `chroma_store/` so it re-indexes.

## Known Limitations / Next Steps

- Runs locally; deploying needs a hosted agent (e.g. LiveKit Cloud Agents) and a hosted frontend (e.g. Vercel).
- One company PDF at a time.
- Answers are logged in plain text to `interview_answers.txt`. That's fine for a demo, but real candidate data would need proper storage and consent.
- No resume support if a candidate disconnects mid-interview.
- Ideas: a reviewer dashboard for transcripts and scores, and question sets per role.

## License

MIT, see [LICENSE](LICENSE). Running the project needs your own API keys for LiveKit, DeepSeek, Deepgram and Cartesia. The license covers the code, not those services.
