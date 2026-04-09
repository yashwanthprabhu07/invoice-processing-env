import os
import json
from groq import Groq
from env import InvoiceEnv
from models import Action

MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError(
        "Missing GROQ_API_KEY. Set GROQ_API_KEY to a valid Groq API key before running agent.py."
    )

client = Groq(api_key=GROQ_API_KEY)


def extract_fields_with_llm(invoice_text):
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
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    fields = json.loads(raw)
    if "date" in fields:
        parts = fields["date"].split()
        if len(parts) >= 2:
            fields["date"] = f"{parts[0]} {parts[1]}"
    return fields


def is_fraud_with_llm(invoice_text):
    prompt = f"""You are a fraud detection expert.
Read this invoice and decide if it is fraudulent.
Fraud signals: duplicate invoices, urgent payment, bank changes, vendor mismatch, missing docs.

Invoice:
{invoice_text}

Respond ONLY with true or false."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    raw = (response.choices[0].message.content or "").strip().lower()
    return "true" in raw


def run_llm_agent(mode="easy", episodes=1):
    env = InvoiceEnv()
    for ep in range(episodes):
        obs = env.reset(mode=mode)
        total_reward = 0.0
        print(f"\n--- Episode {ep+1} | Mode: {mode} ---")
        print(f"Invoice:\n{obs.invoice_text.strip()}\n")
        print("LLM extracting fields...")
        try:
            fields = extract_fields_with_llm(obs.invoice_text)
            print(f"LLM extracted: {fields}")
        except Exception as e:
            print(f"Extraction failed: {e}")
            continue
        for field_name, value in fields.items():
            obs, reward, done, _ = env.step(Action(
                action_type="extract_field",
                field_name=field_name,
                value=value
            ))
            print(f"  extract {field_name:10} = '{value}' -> reward: {reward:+.2f}")
            total_reward += reward
        obs, reward, done, _ = env.step(Action(action_type="validate"))
        print(f"  validate           -> reward: {reward:+.2f}")
        total_reward += reward
        if mode == "hard":
            print("LLM checking for fraud...")
            fraud = is_fraud_with_llm(obs.invoice_text)
            print(f"LLM fraud decision: {fraud}")
            if fraud:
                obs, reward, done, _ = env.step(Action(action_type="flag_fraud"))
                print(f"  flag_fraud         -> reward: {reward:+.2f}")
                total_reward += reward
        obs, reward, done, _ = env.step(Action(action_type="finish"))
        print(f"  finish             -> reward: {reward:+.2f}")
        total_reward += reward
        print(f"\nTotal reward: {total_reward:.2f}")
        print(f"Done: {done}")


if __name__ == "__main__":
    for mode in ["easy", "medium", "hard"]:
        run_llm_agent(mode=mode, episodes=1)
