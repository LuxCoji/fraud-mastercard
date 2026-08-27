# Deploying to Cloud Run

Models live in GCS, not in this repository. The image fetches them at startup,
so promoting a new model is an upload rather than a rebuild.

## 1. Upload the models

```bash
BUCKET=your-bucket
gsutil -m cp artifacts/card_lightgbm.txt \
             artifacts/wire_xgboost_medium.json \
             artifacts/wire_feature_names_medium.json \
             artifacts/wire_model_card_medium.json \
             gs://$BUCKET/fraud-mastercard/artifacts/
```

## 2. Build and deploy

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/repo/fraud-router

gcloud run deploy fraud-router \
  --image REGION-docker.pkg.dev/PROJECT/repo/fraud-router \
  --region REGION \
  --no-allow-unauthenticated \
  --set-env-vars MODEL_BUCKET=$BUCKET \
  --memory 2Gi --cpu 2 --concurrency 40
```

## Settings that matter, and why

**`--no-allow-unauthenticated`.** This endpoint takes payment-shaped payloads
and returns block/allow decisions. There is no reason for it to be reachable
from the open internet.

**`--memory 2Gi`.** The card feature builder holds its lookup tables in memory -
about 1.2 GB fitted. Below 2 GB the container is killed during startup.

**Do not set `--min-instances` above 0 without deciding to.** A previous
deployment on this project ran `min-instances=1` with CPU throttling disabled,
which billed a full vCPU continuously for weeks. Cold starts here are a few
seconds, dominated by the model fetch.

**Concurrency 40.** Scoring is measured at 5.53 ms p99 for the card rail, so a
single instance has ample headroom; the binding constraint is memory, not CPU.

## Promoting a new model

Upload the new file, then restart the service. The registry's promote/rollback
tracks which version is active - see `src/monitoring/registry.py`.

```bash
gsutil cp artifacts/wire_xgboost_v3.json gs://$BUCKET/fraud-mastercard/artifacts/
gcloud run services update fraud-router --region REGION
```
