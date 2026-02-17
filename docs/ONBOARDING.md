# Campaign Onboarding Guide

This document describes how to create and configure new campaigns for the Callbuddy service.

## Directory Structure

Each campaign requires the following directory structure:

```
customer_data/<client>/
├── firm_config.json         # Firm configuration (name)
├── users.csv                # List of users for all campaigns (optional)
└── <campaign>/
    ├── config.json          # Campaign configuration (required)
    ├── users.csv            # Campaign-specific users (optional, overrides parent)
    ├── campaign_info/       # Raw campaign materials (optional)
    │   ├── product_info.txt
    │   ├── sales_script.txt
    │   └── objection_guide.txt
    ├── conversations/       # Audio files for dataset generation (optional)
    ├── speaker_map.json     # Speaker mapping for audio files (optional)
    ├── processed_conversations/ # Auto-generated transcriptions
    └── dataset/             # Auto-generated test cases
```

---

## Configuration Schema

### config.json

```json
{
  "campaign": {
    "id": "my_campaign",
    "name": "My Campaign Display Name",
    "description": "Optional description",
    "language": "Dansk",
    "status": "draft"
  },
  "api_keys": {
    "gemini": "sk-...",
    "soniox": "..."
  },
  "admin_users": [
    {
      "email": "admin@example.com",
      "name": "Admin Name"
    }
  ],
  "campaign": {
    "id": "my_campaign",
    "name": "My Campaign Display Name",
    "description": "Optional description",
    "language": "Dansk",
    "status": "draft"
  },
  "dataset": {
    "name": "my_campaign",
    "action_types": ["close", "objection", "opening", "price_comparison"]
  },
  "vad": {
    "speech_minimum_duration_ms": 2000,
    "silence_before_ai_ms": 500
  },
  "prompts": {
    "generate_campaign_info": true,
    "base_system_prompt": "salescall_system1",
    "output_format_prompt": "output_markdown_suggestions",
    "actions": {
      "objection": {
        "langfuse_name": "my_campaign_objection",
        "action_prompt": "objection_prompt",
        "enabled": true
      }
    }
  },
  "campaign_info_files": ["product_info.txt", "sales_script.txt"]
}
```

### Configuration Sections

| Section | Description |
|---------|-------------|
| `api_keys` | API keys to be stored securely in Supabase (Gemini, Soniox) |
| `admin_users` | List of admins to be created and linked to the campaign |
| `campaign` | Basic campaign identification and metadata |
| `dataset` | Dataset generation settings and action types |
| `vad` | Voice Activity Detection configuration |
| `prompts` | Langfuse prompt configuration |
| `campaign_info_files` | List of files in `campaign_info/` to process |

### firm_config.json (in parent directory)

```json
{
  "name": "My Firm Name"
}
```

The firm configuration is placed in the parent directory (e.g., `customer_data/daica/firm_config.json`) and is shared across all campaigns for that firm. This avoids duplicating firm info in each campaign.

---

## Available Action Types

| Action Type | Description | Structure | Default Internet |
|-------------|-------------|-----------|-----------------|
| `objection` | Customer objection handling | standard | No |
| `close` | Closing the sale | standard | No |
| `opening` | Call opening/hook | standard | No |
| `price_comparison` | Price comparison scenarios | custom | Yes |
| `ai_suggestion` | AI suggestion trigger points | N/A (VAD-based) | No |

---

## Prompt Structures

### Standard Structure
Uses component arrays to build prompts from Langfuse:
```json
{
  "structure": "standard",
  "config": {
    "use_reasoning": false,
    "use_internet": false,
    "output_type": "text"
  },
  "components": {
    "system": ["salescall_system1", "{campaign_id}_info", "output_markdown_suggestions"],
    "user": ["objection_prompt"]
  }
}
```

### Custom Structure
Uses inline prompt content with custom features:
```json
{
  "structure": "custom",
  "config": {
    "use_reasoning": false,
    "use_internet": true,
    "output_type": "text",
    "search_sources": ["strømligning.dk"]
  },
  "prompt_content": "## Opgave\n\nFind kundens selskab...\n\n{{transcript}}"
}
```

### Frontend Config Options
| Option | Type | Description |
|--------|------|-------------|
| `use_reasoning` | boolean | Enable LLM reasoning/thinking |
| `use_internet` | boolean | Enable web search |
| `output_type` | string | Response format ("text", "markdown", etc.) |
| `search_sources` | array | Preferred web sources (when use_internet=true) |

---

## Running the Onboarding Script

### Full Onboarding
```bash
cd scripts
python onboard.py --customer-data-dir ../customer_data/hello_sales/my_campaign --push-to-langfuse
```

### Dry Run (Preview)
```bash
python onboard.py --customer-data-dir ../customer_data/hello_sales/my_campaign --dry-run
```

### Skip Options
```bash
# Skip dataset generation (no audio files)
python onboard.py --customer-data-dir ... --skip-dataset

# Skip prompt creation
python onboard.py --customer-data-dir ... --skip-prompts

# Skip campaign info generation
python onboard.py --customer-data-dir ... --skip-campaign-info
```

---

## Prompt Structure

Campaign prompts in Langfuse combine multiple components:

```
System Prompt:
├── salescall_system1         (base instructions)
├── {campaign_id}_info        (campaign-specific information)
└── output_markdown_suggestions (output format)

User Prompt:
└── {action}_prompt           (action-specific instructions)
```

---

## Step-by-Step: Creating a New Campaign

1. **Create directory structure**
   ```bash
   mkdir -p customer_data/client_name/campaign_name/{campaign_info,conversations}
   ```

2. **Add campaign materials** to `campaign_info/`:
   - Product information
   - Sales scripts
   - Objection handling guides

3. **Create `config.json`** using the schema above

4. **Add audio files** to `conversations/` (optional)

5. **Create `speaker_map.json`** if audio files exist:
   ```json
   {
     "audio_filename_without_extension": true
   }
   ```
   (`true` = speaker 1 is seller, `false` = speaker 1 is customer)

6. **Run onboarding**:
   ```bash
   python scripts/onboard.py --customer-data-dir customer_data/client_name/campaign_name --push-to-langfuse
   ```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No audio files found" | This is normal if you're creating a campaign without training data |
| "Invalid action types" | Check that action types in config match the valid types above |
| "Langfuse prompt not found" | Ensure the base prompts exist in Langfuse (salescall_system1, etc.) |
| "GEMINI_API_KEY missing" | Set the environment variable or add to scripts/.env |
| "Supabase sync failed" | Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in scripts/.env |

## User Access (users.csv)

To bulk onboard users and assign them to the campaign, create a `users.csv` file in the campaign directory (or firm directory for shared users):

```csv
email
john@example.com
jane@example.com
```

- Users are identified by email only
- Firm is automatically assigned from `firm_config.json` in the parent directory
- Running the script multiple times is safe (idempotent) - existing users won't be duplicated
