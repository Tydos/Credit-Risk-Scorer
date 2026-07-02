import random
from locust import HttpUser, between, task

def random_application() -> dict:
    """Generate a valid LoanApplicationPayload on every call."""
    PURPOSES = [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "medical", "small_business", "car", "vacation",
        "moving", "house", "wedding", "renewable_energy", "educational",
    ]
    GRADES = ["A", "B", "C", "D", "E", "F", "G"]
    TERMS = ["36 months", "60 months"]

    loan_amount = round(random.uniform(1_000, 50_000), 2)
    interest_rate = round(random.uniform(0.05, 0.25), 4)
    return {
        "loan_amount": loan_amount,
        "annual_income": round(random.uniform(25_000, 250_000), 2),
        "debt_to_income_ratio": round(random.uniform(0.05, 0.65), 4),
        "credit_score": random.randint(300, 850),
        "interest_rate": interest_rate,
        "installment": round(loan_amount * interest_rate / 12, 2),
        "revol_util": round(random.uniform(0.0, 1.0), 4),
        "grade": random.choice(GRADES),
        "term": random.choice(TERMS),
        "loan_purpose": random.choice(PURPOSES),
    }

class User(HttpUser):
    "Simulate a user using the API"
    wait_time = between(0.1,0.5) # wait 100<500ms before sending the next response

    @task(10)
    def predict(self):
        payload = random_application()
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True
        ) as resp:
            if resp.status_code != 200:
                  resp.failure(f"status={resp.status_code}, body={resp.text[:200]}")
                  return
            try:
                body = resp.json()
            except Exception as exc:
                resp.failure(f"invalid json: {exc}")
                return
            if body.get("prediction") not in (0, 1):
                resp.failure(f"unexpected prediction: {body.get('prediction')}")
                return
            if "inference_latency_ms" not in body:
                resp.failure("missing inference_latency_ms")
