from models import Observation, Action
from typing import Tuple, Dict, Any
from tasks.easy import get_easy_task
from tasks.medium import get_medium_task
from tasks.hard import get_hard_task


def normalize(value: str) -> str:
    return value.strip().lower().replace(",", "").replace("$", "").replace(".", "")


class InvoiceEnv:
    def __init__(self):
        self.max_steps = 10
        self.reset()

    def seed(self, seed=None):
        import random
        random.seed(seed)
        return [seed]

    def render(self, mode="human"):
        print(f"\n--- Invoice Env State ---")
        print(f"Step     : {self.state_data['step_count']} / {self.max_steps}")
        print(f"Invoice  : {self.state_data['invoice_text'].strip()}")
        print(f"Fields   : {self.state_data['extracted_fields']}")
        print(f"Errors   : {self.state_data['validation_errors']}")
        print(f"Fraud    : {self.state_data['fraud']} | Detected: {self.state_data['fraud_detected']}")
        print(f"Done     : {self.state_data['done']}")
        print(f"-------------------------\n")

    def reset(self, mode="easy") -> Observation:
        if mode == "easy":
            task = get_easy_task()
        elif mode == "medium":
            task = get_medium_task()
        else:
            task = get_hard_task()

        self.state_data = {
            "invoice_text": task["invoice_text"],
            "true_fields": task["true_fields"],
            "fraud": task["fraud"],
            "fraud_detected": False,
            "extracted_fields": {
                "amount": None,
                "date": None,
                "vendor": None
            },
            "validation_errors": [],
            "step_count": 0,
            "done": False
        }

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state_data["done"]:
            return self._get_observation(), 0.0, True, {}

        self.state_data["step_count"] += 1
        reward = 0.0
        info = {}

        if action.action_type == "extract_field":
            field = action.field_name
            value = action.value

            if field in self.state_data["true_fields"]:
                if normalize(value) == normalize(self.state_data["true_fields"][field]):
                    self.state_data["extracted_fields"][field] = self.state_data["true_fields"][field]
                    reward += 0.2
                else:
                    reward -= 0.2
            else:
                reward -= 0.2

        elif action.action_type == "validate":
            errors = []
            for k, v in self.state_data["true_fields"].items():
                if self.state_data["extracted_fields"][k] != v:
                    errors.append(f"{k} incorrect")

            self.state_data["validation_errors"] = errors

            if len(errors) == 0:
                reward += 0.3
            else:
                reward -= 0.1

        elif action.action_type == "flag_fraud":
            if self.state_data["fraud"]:
                reward += 0.5
                self.state_data["fraud_detected"] = True
            else:
                reward -= 0.5

        elif action.action_type == "finish":
            if len(self.state_data["validation_errors"]) > 0:
                reward -= 0.5
            elif self.state_data["step_count"] < 4:
                reward -= 0.5
            elif all(
                self.state_data["extracted_fields"][k] == v
                for k, v in self.state_data["true_fields"].items()
                if v is not None
            ) and (not self.state_data["fraud"] or self.state_data["fraud_detected"]):
                efficiency_bonus = round(max(0, (self.max_steps - self.state_data["step_count"]) * 0.05), 2)
                reward += 1.0 + efficiency_bonus
                self.state_data["done"] = True
            else:
                reward -= 0.3

        else:
            reward -= 0.2

        if self.state_data["step_count"] >= self.max_steps:
            self.state_data["done"] = True

        return self._get_observation(), reward, self.state_data["done"], info

    def state(self) -> Dict:
        return self.state_data

    def _get_observation(self) -> Observation:
        return Observation(
            invoice_text=self.state_data["invoice_text"],
            extracted_fields=self.state_data["extracted_fields"],
            validation_errors=self.state_data["validation_errors"],
            fraud_detected=self.state_data["fraud_detected"],
            step_count=self.state_data["step_count"],
            goal="Extract all fields and validate invoice correctly"
        )