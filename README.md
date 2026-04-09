---
title: Invoice Processing Environment
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
---


# Invoice Processing Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent reads invoices, extracts fields, validates them, and detects fraud.

## Setup
```bash
uv sync
```

## Run Tests
```bash
python test_env.py
```

## Quick Start
```python
from env import InvoiceEnv
from models import Action

env = InvoiceEnv()
obs = env.reset(mode="easy")
print(obs.invoice_text)

obs, reward, done, info = env.step(Action(action_type="extract_field", field_name="amount", value="5000"))
print(reward, done)
```

## How It Works

### Actions
| Action | Description | Reward |
|---|---|---|
| `extract_field` | Extract amount, date, or vendor | +0.2 correct / -0.2 wrong |
| `validate` | Check all extracted fields | +0.3 correct / -0.1 errors |
| `flag_fraud` | Flag invoice as fraudulent | +0.5 correct / -0.5 wrong |
| `finish` | Complete the episode | +0.99 + efficiency bonus / -0.3 wrong |

### Efficiency Bonus
Finishing in fewer steps earns extra reward: `(max_steps - step_count) * 0.05`

### Difficulty Modes
| Mode | Description |
|---|---|
| `easy` | Clean invoice, no fraud |
| `medium` | Noisier invoice, extra fields, no fraud |
| `hard` | Fraudulent invoice with embedded fraud signals |

## REST API Server

The environment is accessible via a FastAPI server.

### Start the server
```bash
python server/app.py
```

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/reset` | Reset environment, returns observation |
| `POST` | `/step` | Take an action, returns obs + reward |
| `GET` | `/state` | Get full current state |

## Graders

Each difficulty mode has a grader that scores field extraction accuracy from 0.0 to 1.0.
```python
from graders.easy_grader import grade_easy

score = grade_easy(env.state_data)
print(f"Extraction accuracy: {score:.2f}")
```

| Grader | File | Mode |
|---|---|---|
| `grade_easy` | `graders/easy_grader.py` | Easy |
| `grade_medium` | `graders/medium_grader.py` | Medium |
| `grade_hard` | `graders/hard_grader.py` | Hard |

## Baseline Agent
```bash
python agent.py
```

### Sample output
```
--- Episode 1 | Mode: easy ---
  extract_field   → reward: +0.20 | done: False
  extract_field   → reward: +0.20 | done: False
  extract_field   → reward: +0.20 | done: False
  validate        → reward: +0.30 | done: False
  finish          → reward: +1.25 | done: True
Total reward: 2.15

--- Episode 1 | Mode: hard ---
  extract_field   → reward: +0.20 | done: False
  extract_field   → reward: +0.20 | done: False
  extract_field   → reward: +0.20 | done: False
  validate        → reward: +0.30 | done: False
  flag_fraud      → reward: +0.50 | done: False
  finish          → reward: +1.20 | done: True
Total reward: 2.60
```

## Reward Range
`[-2, 2.59]`

## Max Steps
`10`

## Notes
- Tasks are randomised on every `reset()` so the agent cannot memorise answers
- Fuzzy field matching handles minor formatting differences in extracted values  

## Baseline Agent

A rule-based agent is included to demonstrate the environment is learnable.
```bash
python agent.py
```

## LLM Agent (Groq + LLaMA)

A real AI-powered agent that reads raw invoice text and extracts fields using LLaMA 3.1 via Groq API — no peeking at answers.

### Environment
Set `GROQ_API_KEY` before running the LLM agent:
```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

You can also create a local `.env` file with:
```env
GROQ_API_KEY=your_groq_api_key
```

## Hackathon Inference Runner (LiteLLM Proxy)

`inference.py` is configured for hackathon validation and must use the injected proxy variables.

### Required environment variables
```powershell
$env:API_BASE_URL="https://your-litellm-proxy/v1"
$env:API_KEY="your_proxy_api_key"
```

### Run inference
```bash
python inference.py
```

Notes:
- Do not hardcode provider keys in `inference.py`.
- If `API_BASE_URL` or `API_KEY` is missing, the script falls back safely instead of crashing.

### How it works
1. Receives raw invoice text as observation
2. Sends it to LLaMA 3.1 to extract amount, vendor, date
3. On hard mode — LLaMA also decides if the invoice is fraudulent
4. Takes actions based on LLM decisions and collects reward

### Run the LLM agent
```bash
python agent.py
```

### Sample output
```
--- Episode 1 | Mode: easy ---
Invoice:
Invoice from XYZ Corp. Amount: $4500. Date: 2 Jan 2024.

LLM extracting fields...
LLM extracted: {'amount': '4500', 'vendor': 'XYZ Corp', 'date': '2 Jan'}
  extract amount     = '4500' -> reward: +0.20
  extract vendor     = 'XYZ Corp' -> reward: +0.20
  extract date       = '2 Jan' -> reward: +0.20
  validate           -> reward: +0.30
  finish             -> reward: +1.25
Total reward: 2.15

--- Episode 1 | Mode: hard ---
LLM extracted: {'amount': '12800', 'vendor': 'Apex Global Traders', 'date': '7 Jun'}
  extract amount     = '12800' -> reward: +0.20
  extract vendor     = 'Apex Global Traders' -> reward: +0.20
  extract date       = '7 Jun' -> reward: +0.20
  validate           -> reward: +0.30
LLM checking for fraud...
LLM fraud decision: True
  flag_fraud         -> reward: +0.50
  finish             -> reward: +1.20
Total reward: 2.60
```
