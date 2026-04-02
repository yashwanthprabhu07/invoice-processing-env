import os
import json
from openai import OpenAI
from env import InvoiceEnv
from models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
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
    if "date" in fields:
        parts = fields["date"].split()
        if len(parts) >= 2:
            fields["date"] = f"{parts[0]} {parts[1]}"
    return fields


def is_fraud(invoice_text):
    prompt = f"""You are a fraud detection expert reviewing invoices.

The following invoice contains ONE OR MORE of these fraud signals:
- "Duplicate" or "resubmitted"
- "Urgent" payment request
- "Different account" or bank details changed
- "Vendor name does not match"
- "Amount billed is 4x" or discrepancy note
- "No supporting documents"
- "Previous invoices already paid"

Read this invoice carefully:
{invoice_text}

Does this invoice contain ANY fraud signal? Respond ONLY with true or false."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return "true" in response.choices[0].message.content.strip().lower()


def run_episode(mode="easy"):
    env = InvoiceEnv()
    obs = env.reset(mode=mode)
    total_reward = 0.0

    print(f"\n--- Mode: {mode} ---")
    print(f"Invoice:\n{obs.invoice_text.strip()}\n")

    try:
        fields = extract_fields(obs.invoice_text)
        print(f"LLM extracted: {fields}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return 0.0

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
        fraud = is_fraud(obs.invoice_text)
        print(f"  fraud decision: {fraud}")
        if fraud:
            obs, reward, done, _ = env.step(Action(action_type="flag_fraud"))
            print(f"  flag_fraud         -> reward: {reward:+.2f}")
            total_reward += reward

    obs, reward, done, _ = env.step(Action(action_type="finish"))
    print(f"  finish             -> reward: {reward:+.2f}")
    total_reward += reward

    print(f"Total reward: {total_reward:.2f}")
    return total_reward


def main():
    print("=== Invoice Processing Environment — Baseline Inference ===\n")
    scores = {}
    for mode in ["easy", "medium", "hard"]:
        score = run_episode(mode=mode)
        scores[mode] = score

    print("\n=== Final Scores ===")
    for mode, score in scores.items():
        print(f"  {mode:10} -> {score:.2f}")


if __name__ == "__main__":
    main()