import os
import json
from openai import OpenAI
from env import InvoiceEnv
from models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = "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"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or GROQ_API_KEY
)


def extract_fields(invoice_text):
    prompt = f"""You are an invoice processing assistant.
Read the following invoice and extract these three fields:
- amount: just the number, no $ or commas (e.g. "5000")
- vendor: the company name only
- date: day and month only (e.g. "10 Oct")

Invoice:
{invoice_text}

Respond ONLY with a JSON object like this:
{{"amount": "5000", "vendor": "ABC Ltd", "date": "10 Oct"}}
No explanation. No markdown. Just the JSON."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    fields = json.loads(raw)
    fields = {k: str(v) for k, v in fields.items()}
    if "date" in fields:
        parts = fields["date"].split()
        if len(parts) >= 2:
            fields["date"] = f"{parts[0]} {parts[1]}"
    return fields


def is_fraud(invoice_text):
    prompt = f"""You are a fraud detection expert. You MUST check the math.

Step 1: Find any line showing rate and quantity (e.g. "30 hrs @ $150/hr")
Step 2: Calculate: quantity x rate = expected amount
Step 3: Compare expected amount to "Amount Due"
Step 4: If Amount Due is MORE than 1.5x the expected amount -> FRAUD

Also check: duplicate invoice numbers, changed bank accounts.

Invoice:
{invoice_text}

Is this invoice fraudulent? Respond ONLY with true or false."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return "true" in response.choices[0].message.content.strip().lower()


def run_episode(mode="easy"):
    env = InvoiceEnv()
    obs = env.reset(mode=mode)
    total_reward = 0.0
    step_count = 0

    print(f"[START] task={mode}", flush=True)

    try:
        fields = extract_fields(obs.invoice_text)
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        print(f"[END] task={mode} score=0.0 steps=0", flush=True)
        return 0.0

    for field_name, value in fields.items():
        obs, reward, done, _ = env.step(Action(
            action_type="extract_field",
            field_name=field_name,
            value=value
        ))
        step_count += 1
        total_reward += reward
        print(f"[STEP] step={step_count} action=extract_{field_name} reward={round(reward, 2)}", flush=True)

    obs, reward, done, _ = env.step(Action(action_type="validate"))
    step_count += 1
    total_reward += reward
    print(f"[STEP] step={step_count} action=validate reward={round(reward, 2)}", flush=True)

    if mode == "hard":
        fraud = is_fraud(obs.invoice_text)
        if fraud:
            obs, reward, done, _ = env.step(Action(action_type="flag_fraud"))
            step_count += 1
            total_reward += reward
            print(f"[STEP] step={step_count} action=flag_fraud reward={round(reward, 2)}", flush=True)

    obs, reward, done, _ = env.step(Action(action_type="finish"))
    step_count += 1
    total_reward += reward
    print(f"[STEP] step={step_count} action=finish reward={round(reward, 2)}", flush=True)

    score = round(total_reward, 2)
    print(f"[END] task={mode} score={score} steps={step_count}", flush=True)
    return total_reward


def main():
    for mode in ["easy", "medium", "hard"]:
        run_episode(mode=mode)


if __name__ == "__main__":
    main()