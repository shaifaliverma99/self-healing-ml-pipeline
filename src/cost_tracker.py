"""Simulated cloud cost accounting, so drift-detector comparisons can report
$-per-run alongside accuracy -- the evaluation axis that makes this an applied
systems paper rather than a pure ML paper.

Rates approximate real 2026 list pricing (AWS Lambda + a small SageMaker/EC2
training instance); swap in your own account's rates in deploy/ once you have
real cloud billing to calibrate against.
"""

INFERENCE_COST_PER_CALL = 0.0000002       # ~ AWS Lambda: $0.20 per 1M invocations
RETRAIN_COST_PER_JOB = 0.05               # ~ small ml.m5.large training job, few minutes


class CostTracker:
    def __init__(self):
        self.n_inferences = 0
        self.n_retrains = 0

    def record_inference(self):
        self.n_inferences += 1

    def record_retrain(self):
        self.n_retrains += 1

    @property
    def total_cost(self):
        return (self.n_inferences * INFERENCE_COST_PER_CALL
                + self.n_retrains * RETRAIN_COST_PER_JOB)

    def summary(self):
        return {
            "n_inferences": self.n_inferences,
            "n_retrains": self.n_retrains,
            "inference_cost_usd": round(self.n_inferences * INFERENCE_COST_PER_CALL, 6),
            "retrain_cost_usd": round(self.n_retrains * RETRAIN_COST_PER_JOB, 4),
            "total_cost_usd": round(self.total_cost, 4),
        }
