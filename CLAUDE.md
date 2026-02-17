# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Callbuddy Service is a Python-based **internal tooling and testing** service for the CallBuddy platform. It is NOT a production service — it is used for offline experimentation, campaign onboarding, dataset generation, and prompt iteration. The production middleware that Flutter talks to is `callbuddy_render` (Node.js/Express on Fly.io).

It processes sales call audio recordings into structured datasets, manages campaigns via Supabase, and integrates with Langfuse for prompt management and LLM evaluation.

The platform is primarily Danish-language focused (default language: "Dansk").

## Architecture

### Two Python packages with shared code

- **`src/`** — Core library with models, services, and utilities. All imports use bare module names (e.g., `from services.llm_client import ...`, `from models.conversation import ...`). Run scripts from within `src/` or add it to `PYTHONPATH`.
- **`scripts/`** — CLI tools for onboarding and dataset generation. The main entry point `scripts/onboard.py` imports from `src/services/` using `from services.xxx import ...`, so it expects `src/` on the Python path.

### Key data flow

1. **Audio ingestion**: Sales call audio files are transcribed via Soniox STT API (`src/services/soniox.py`)
2. **Conversation modeling**: Transcriptions are parsed into `Conversation` objects with speaker-diarized `Utterance`s (`src/models/conversation.py`)
3. **Scenario detection**: Gemini LLM analyzes conversations to identify action points (objections, closes, openings, price comparisons) and produces `ScenarioTestCase` objects (`src/services/llm.py`, `src/models/scenario_test_case.py`)
4. **VAD-based detection**: Silero VAD v5 detects speech/silence patterns to identify AI suggestion trigger points (`src/services/vad.py`)
5. **Dataset management**: Results are pushed to Langfuse as dataset items for evaluation (`src/services/langfuse.py`, `src/sample_dataset.py`)
6. **Campaign onboarding**: Full pipeline orchestrated by `scripts/onboard.py` — syncs firms/campaigns/users to Supabase, generates campaign info via Gemini, creates prompts in Langfuse

### External services

| Service | Purpose | Env vars |
|---------|---------|----------|
| **Supabase** | Database for firms, campaigns, users, prompt templates | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` |
| **Langfuse** | Prompt management, dataset storage, evaluation | `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` |
| **Soniox** | Speech-to-text transcription | `SONIOX_API_KEY` |
| **Gemini** | LLM for scenario detection and campaign info generation | `GEMINI_API_KEY` |

### Campaign info

Each campaign has a **campaign info** text prompt (stored in Langfuse as `<campaign_id>_info`). This is the single source of truth for everything about the campaign: company description, products/services, pricing (abonnement, tillæg, oprettelsesgebyr, binding), target audience, sales approach, and competitive positioning. It is injected into state patcher, responder, and competitor fetcher prompts via `{{CAMPAIGN_INFO}}`.

### customer_data/ directory

Per-client campaign data following the structure: `customer_data/<firm>/<campaign>/`. Each campaign has a `config.json` (campaign settings, prompt configs, action types), optional `campaign_info/` (raw materials), `conversations/` (audio files), and auto-generated `processed_conversations/` and `dataset/` directories. Firm-level config lives in `customer_data/<firm>/firm_config.json`. See `docs/ONBOARDING.md` for the full schema.

## Commands

### Campaign onboarding
```bash
cd scripts
python onboard.py --customer-data-dir ../customer_data/<firm>/<campaign> --push-to-langfuse
python onboard.py --customer-data-dir ../customer_data/<firm>/<campaign> --dry-run
python onboard.py --customer-data-dir ../customer_data/<firm>/<campaign> --skip-dataset --skip-prompts
python onboard.py --customer-data-dir ../customer_data/<firm>/<campaign> --create-users-only
python onboard.py --customer-data-dir ../customer_data/<firm>/<campaign> --local  # use local Supabase
```

### Dataset sampling
```bash
cd src
python sample_dataset.py --customer-data-dir ../customer_data/<firm>/<campaign> --type objection --amount 10 --name "test_v1" --push-to-langfuse
```

### Install dependencies
```bash
pip install -r src/requirements.txt
```

### Environment setup
All API keys are loaded from a `.env` file (searched in CWD, then project root). Required keys depend on which services you're using — see the external services table above.

## Langfuse Prompt System

Prompts are managed in Langfuse and use **composite prompts (composability)** — a modular system where prompts reference other prompts via tags:
```
@@@langfusePrompt:name=PromptName|label=production@@@
```
When fetched via the SDK, tags are replaced with the referenced prompt content. This lets us maintain campaign context, state patcher instructions, and responder logic as separate reusable text prompts that compose into full prompts automatically. See `src/services/prompt_generator.py`.

### Prompt versioning

Local prompt version files are saved at `prompts/<prompt_name>/v<N>_prompt.txt` and `v<N>_config.json`. Use `/langfuse <prompt-name>` to fetch, inspect, or save prompt versions.

### Prompt discussion notes

Design discussions and draft prompts live at `/Users/mikkeldahl/Notes/new_backend_discussion/` with subdirectories for each prompt type (analysis, feedback, responder, state_updater, summary). Additional drafts may be in `new_backend_discussion/` at the project root.

## Key Patterns

- **`load_env_value(key)`** (`src/utils.py`): Lightweight `.env` loader used everywhere instead of `python-dotenv`. Searches CWD then two parents up.
- **Action types**: `objection`, `close`, `opening`, `price_comparison`, `ai_suggestion`. The first four use LLM-based detection; `ai_suggestion` uses VAD-based detection.
- **LLM clients**: `GeminiChatCompletionClient` (primary, used for scenario detection) and `BasetenChatCompletionClient` in `src/services/llm_client.py`. Both follow the `ChatCompletionClient` protocol.
- **Supabase tables**: `firm`, `campaigns`, `campaigns_prompt_templates`, `user`, `user_campaign`. Managed via `src/services/supabase_client.py`.
