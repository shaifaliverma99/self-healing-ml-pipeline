"""Run this ON CAMERA against the live deployed API to show the self-healing
loop actually happening: predictions, a real drift event, a diagnosis of
which feature is implicated, a validation-gated retrain (committed or
rejected), and the final cost summary.

Usage:
    python demo/live_demo.py --base https://self-healing-ml-pipeline-7ejk.onrender.com
    python demo/live_demo.py --base http://localhost:8000   # fast local rehearsal
"""
import argparse
import random
import time
import urllib.request
import json

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--n", type=int, default=900, help="total requests to send")
parser.add_argument("--drift-at", type=int, default=300, help="sample index where the concept flips")
parser.add_argument("--seed", type=int, default=42, help="fixed seed for a reproducible take")
args = parser.parse_args()
random.seed(args.seed)


def call(method, path, payload=None):
    url = args.base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def make_sample(i):
    """Same 3-feature SEA-style rule the API's baseline was warm-started on
    (label = f0+f1 > 8); at --drift-at the rule INVERTS, matching the SINE
    generator's abrupt-reversal drift used in the paper's experiments. This
    guarantees the old model's accuracy craters right at the drift point,
    giving a fast, clearly visible detection instead of a subtle one."""
    f = [random.uniform(0, 10) for _ in range(3)]
    base = int(f[0] + f[1] > 8.0)
    label = base if i < args.drift_at else 1 - base
    return f, label


print(f"Resetting service at {args.base} ...")
call("POST", "/reset")
time.sleep(1)

print(f"Streaming {args.n} samples, concept drift injected at sample {args.drift_at}\n")
window = []
last_version = 0

for i in range(args.n):
    feats, label = make_sample(i)
    resp = call("POST", "/predict", {"features": feats, "true_label": label})
    window.append(int(resp["prediction"] == label))
    if len(window) > 50:
        window.pop(0)

    if resp.get("recovery") == "committed":
        print(f"\n*** DRIFT DETECTED at sample {i} -- diagnosis implicates feature "
              f"{resp['diagnosis_feature']} -- retrain validated and committed "
              f"(v{last_version} -> v{resp['model_version']}) ***\n")
        last_version = resp["model_version"]
    elif resp.get("recovery") == "rejected":
        print(f"\n*** DRIFT DETECTED at sample {i} -- diagnosis implicates feature "
              f"{resp['diagnosis_feature']} -- candidate retrain REJECTED by validation "
              f"gate (did not beat current model on held-out data) ***\n")

    if (i + 1) % 25 == 0:
        acc = sum(window) / len(window)
        print(f"sample {i + 1:4d}/{args.n}  recent_acc(last 50)={acc:.2f}  "
              f"model_version={resp['model_version']}")

print("\nFinal status:")
print(json.dumps(call("GET", "/status"), indent=2))
