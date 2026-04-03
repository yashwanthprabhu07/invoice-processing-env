import os
import json
from groq import Groq
from env import InvoiceEnv
from models import Action

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    fields = json.loads(raw)
    if "date" in fields:
        parts = fields["date"].split()
        if len(parts) >= 2:
            fields["date"] = f"{parts[0]} {parts[1]}"
    return fields


def is_fraud_with_llm(invoice_text):
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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    raw = response.choices[0].message.content.strip().lower()
    return "true" in raw


def run_agent(mode="easy", episodes=1):
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
            fraud = is_fraud_with_llm(obs.invoice_text)
            print(f"  fraud decision: {fraud}")
            if fraud:
                obs, reward, done, _ = env.step(Action(action_type="flag_fraud"))
                print(f"  flag_fraud         -> reward: {reward:+.2f}")
                total_reward += reward
        obs, reward, done, _ = env.step(Action(action_type="finish"))
        print(f"  finish             -> reward: {reward:+.2f}")
        total_reward += reward
        print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    for mode in ["easy", "medium", "hard"]:
        run_agent(mode=mode, episodes=1)