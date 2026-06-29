import random
from locust import HttpUser, between, task

def random_application() -> dict:
    """Generate a valid LoanApplicationPayload on every call."""
    GENDER = ["Female", "Male", "Other"]
    MARITAL = ["Single", "Married", "Divorced", "Widowed"]
    EDUCATION = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    EMPLOYMENT = ["Employed", "Self-employed", "Unemployed", "Retired", "Student"]
    PURPOSE = [
        "Debt consolidation",
        "Home improvement",
        "Medical",
        "Education",
        "Business",
        "Other",
    ]
    GRADES = [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5",
        "G1", "G2", "G3", "G4", "G5",
    ]
    return {
          "annual_income": round(random.uniform(25_000, 250_000), 2),
          "debt_to_income_ratio": round(random.uniform(0.05, 0.65), 4),
          "credit_score": random.randint(300, 850),
          "loan_amount": round(random.uniform(1_000, 50_000), 2),
          "interest_rate": round(random.uniform(0.05, 0.25), 4),
          "gender": random.choice(GENDER),
          "marital_status": random.choice(MARITAL),
          "education_level": random.choice(EDUCATION),
          "employment_status": random.choice(EMPLOYMENT),
          "loan_purpose": random.choice(PURPOSE),
          "grade_subgrade": random.choice(GRADES),
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
