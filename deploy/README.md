# Cloud deployment

The API (`src/api.py`) is a plain FastAPI app, so any of these work. Pick based
on what free-tier/student credits you have access to — none of this needs
provisioning until you're ready to actually deploy.

## Option A — AWS Lambda (serverless, cheapest for a student demo)
1. `pip install mangum` and wrap the app: `handler = Mangum(app)` in a new `src/lambda_handler.py`.
2. Build the container image from the existing `Dockerfile` (Lambda supports container images up to 10GB).
3. `aws ecr create-repository`, push the image, then `aws lambda create-function --package-type Image`.
4. Put API Gateway (HTTP API) in front for a public URL.
5. This is what makes the "cost-aware" story real: Lambda bills per-invocation, matching `cost_tracker.py`'s per-call assumption directly — you can pull actual CloudWatch billing data to replace the simulated rate.

## Option B — GCP Cloud Run (also serverless, simplest CLI)
```
gcloud run deploy self-healing-ml --source . --port 8000 --allow-unauthenticated
```
Cloud Run's per-request billing plays the same role as Lambda above.

## Option C — Render / Railway (no cloud account needed, fastest to a public URL)
Good if you want a demo link for your resume/interview without setting up AWS/GCP
billing at all. Push this repo, connect it, point it at the `Dockerfile`.

## What to actually measure once deployed
- Real dollar cost per 1,000 inferences and per retrain job (replace the constants
  in `src/cost_tracker.py` with your actual billing).
- Cold-start latency if you go serverless (Lambda/Cloud Run) — worth reporting
  in the paper as an applied-systems tradeoff against a warm always-on instance.
