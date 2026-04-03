from env import InvoiceEnv
from models import Action

env = InvoiceEnv()

# TEST 1: BASIC FLOW
print("\n--- BASIC FLOW TEST ---")
obs = env.reset()
fields = env.state_data["true_fields"]
actions = [
    Action(action_type="extract_field", field_name="amount", value=fields["amount"]),
    Action(action_type="extract_field", field_name="vendor", value=fields["vendor"]),
    Action(action_type="extract_field", field_name="date",   value=fields["date"]),
    Action(action_type="validate"),
    Action(action_type="finish"),
]
for a in actions:
    obs, reward, done, _ = env.step(a)
    print(f"Reward: {reward}, Done: {done}")

# TEST 2: WRONG FIELD
print("\n--- WRONG FIELD TEST ---")
env.reset()
obs, reward, done, _ = env.step(Action(action_type="extract_field", field_name="wrong_field", value="123"))
print("Reward:", reward, "Done:", done)

# TEST 3: WRONG EXTRACTION
print("\n--- WRONG EXTRACTION TEST ---")
env.reset()
obs, reward, done, _ = env.step(Action(action_type="extract_field", field_name="amount", value="WRONG"))
print("Reward:", reward, "Done:", done)

# TEST 4: SKIP VALIDATION
print("\n--- SKIP VALIDATION TEST ---")
env.reset()
fields = env.state_data["true_fields"]
env.step(Action(action_type="extract_field", field_name="amount", value=fields["amount"]))
obs, reward, done, _ = env.step(Action(action_type="finish"))
print("Reward:", reward, "Done:", done)

# TEST 5: FRAUD DETECTION
print("\n--- FRAUD TEST ---")
env.reset(mode="hard")
obs, reward, done, _ = env.step(Action(action_type="flag_fraud"))
print("Reward:", reward, "Done:", done)

# TEST 6: MULTI-MODE
print("\n--- MULTI MODE TEST ---")
for mode in ["easy", "medium", "hard"]:
    obs = env.reset(mode=mode)
    print(f"{mode.upper()} MODE:", obs.invoice_text.strip())

# TEST 7: STEP LIMIT
print("\n--- STEP LIMIT TEST ---")
env.reset()
for i in range(12):
    obs, reward, done, _ = env.step(Action(action_type="extract_field", field_name="amount", value="WRONG"))
    if done:
        print("Stopped at step:", i + 1)
        break