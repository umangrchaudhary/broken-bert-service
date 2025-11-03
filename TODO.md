# ML Engineer Debugging & API Integration Challenge

A pragmatic, production-style task to assess your debugging, refactoring, and API integration skills across an ML microservice.

---

## Overview

You will review and repair a real-world microservice for **sentiment analysis** (fine‑tuned DistilBERT) and **product recommendations** (optional **Qdrant** vector search). The codebase runs end‑to‑end but contains deliberate bugs—some obvious, others subtle—that impact **training**, **inference**, **batching**, and **vector search**.

Your job: diagnose, fix, and ship a working pipeline.

---

## Tech Stack

* **Model**: DistilBERT (fine‑tuned)
* **Service**: FastAPI
* **Vector Store**: Qdrant (optional, for product similarity)
* **Runtime**: Python, PyTorch, Uvicorn

---

## What You Need to Do

1. **Train**: Run the training script

   ```bash
   python -m ml.train
   ```
2. **Inspect API**: Verify `/predict` and `/recommend` behaviors (inputs/outputs, status codes, latency, logs).
3. **Debug & Fix** across:

   * Model **training & saving**
   * **Prediction** correctness (single & batch)
   * **Batch inference** alignment (IDs ↔ predictions)
   * **Qdrant** product vector search
   * Fragile logic & API contract mismatches
4. **Refactor** where necessary (don’t wallpaper over bad design).
5. **Write** a short `FIX_REPORT.md` describing what you fixed and why.

---

## Symptoms You’ll See

* Wrong/unstable predictions
* Confidence scores that don’t make sense
* Batch API returns misaligned results
* Empty or low‑quality recommendations
* Flaky behaviors due to device, dtype, or hidden state issues

Some bugs won’t throw exceptions—expect silent logic errors (e.g., incorrect loss, device mismatch, vector corruption).

---

## Acceptance Criteria

A ✅ **pass** means:

* **Training** completes and the model is saved correctly.
* **/predict** returns correct **label** and **confidence** per input.
* **/predict (batch)** returns outputs aligned to the input order.
* **/recommend** returns non‑empty, relevant products (when Qdrant is running & loaded).
* `FIX_REPORT.md` is clear, specific (file + line ranges), and justifies changes.

Bonus (optional, earns extra points):

* Unit tests for `/predict` and `/recommend`.
* Optimized model load time or batch throughput.
* Retry/fallback for vector store downtime.

---

## API Contract (expected)

> Implementations may vary; fix code to comply with this contract.

### `POST /predict`

**Request (single):**

```json
{ "text": "Great product!" }
```

**Response:**

```json
{ "label": "positive", "confidence": 0.97 }
```

**Request (batch):**

```json
{ "texts": ["Great", "Terrible"] }
```

**Response:**

```json
{
  "predictions": [
    { "label": "positive", "confidence": 0.97 },
    { "label": "negative", "confidence": 0.95 }
  ]
}
```

### `POST /recommend`

**Request:**

```json
{ "product_id": "SKU-123", "k": 5 }
```

**Response:**

```json
{
  "product_id": "SKU-123",
  "neighbors": [
    { "id": "SKU-456", "score": 0.83 },
    { "id": "SKU-789", "score": 0.81 }
  ]
}
```

**Notes**

* Return **HTTP 400** on invalid input schema; **HTTP 503** when vector store is unavailable (with a graceful message).
* Batch prediction order **must match** input order.
* Confidence must be derived from **softmax probabilities** of the returned class.

---

## Environment & Runbook

* Start API:

  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
  ```
* Health check: `GET /healthz` → `{ "status": "ok" }` (implement if missing)
* Qdrant local:

  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
  ```
* Seed vectors (if required by codebase): provide a script or update README.

---

## Debugging Checklist

**Training**

* [ ] Correct loss for classification (e.g., CrossEntropy vs. BCE)
* [ ] Model set to **train()** during training, **eval()** during inference
* [ ] Device consistency (CPU/GPU) & dtype
* [ ] Proper saving: weights + tokenizer + label map

**Inference**

* [ ] Tokenization params (truncation, padding, max_length)
* [ ] No gradient tracking during inference (`torch.no_grad()`)
* [ ] Softmax on logits; pick argmax per sample
* [ ] Batch ordering preserved
* [ ] Confidence from predicted class prob

**API Layer**

* [ ] Input schema validation (Pydantic models)
* [ ] Distinguish single vs batch payloads
* [ ] Deterministic JSON keys & types
* [ ] Robust error handling and status codes

**Qdrant**

* [ ] Collection created with correct **vector size** and **distance** metric
* [ ] Vectors normalized if cosine distance is used
* [ ] Upsert pipeline doesn’t corrupt vector ordering/IDs
* [ ] Query returns non‑empty results for known IDs

**Observability**

* [ ] Structured logs (request_id, path, latency)
* [ ] Basic metrics (requests/sec, p95 latency) if feasible

---

## Deliverables

1. Working pipeline with:

   * ✅ Correct training & saved artifacts
   * ✅ Correct predictions & confidence
   * ✅ Batch alignment
   * ✅ Meaningful recommendations
2. `FIX_REPORT.md` with:

   * Bugs found (file + line ranges)
   * What changed & why
   * Notes on production‑hardening

---

## Submission

* Commit your fixes and **squash** noisy WIP commits.
* Include both files at repo root:

  * `README.md` (this document)
  * `FIX_REPORT.md` (see template below)
* Provide commands to run training, API, and (optionally) Qdrant seeding.

---

## Scoring Rubric (what we evaluate)

* **Debugging Depth**: Did you find silent logic issues, not just syntax?
* **ML Understanding**: Loss, softmax, device/dtype, batching
* **API Quality**: Contracts, status codes, schemas, deterministic outputs
* **Vector Search**: Correct collection config & meaningful neighbors
* **Code Quality**: Clarity, refactors, tests (bonus)
* **Report Quality**: Specificity, rationale, and trade‑offs

---

## Tips

* Reproduce first. Add a failing test or script that demonstrates the bug.
* Log intermediate tensors (shape, min/max) to catch dtype/device issues.
* Compare single vs. batch outputs on the same inputs.
* For Qdrant, verify vector dimensionality and distance metric match your embeddings.

---

# FIX_REPORT.md — Template

Copy this into `FIX_REPORT.md` and fill it in.

```md
# FIX_REPORT

## Summary
Concise overview of what was broken and how you fixed it.

## Bug Log

### 1) Title of Bug
- **Symptoms:** What you observed (wrong outputs, empty results, etc.)
- **Root Cause:** Why it happened
- **Location:** `path/file.py` (lines A–B)
- **Fix:** What you changed (+ why)
- **Validation:** How you verified the fix (commands, tests, metrics)

### 2) Title of Bug
- **Symptoms:** ...
- **Root Cause:** ...
- **Location:** ...
- **Fix:** ...
- **Validation:** ...

> Add more sections as needed

## API Behavior After Fixes
- **/predict (single):** sample request/response
- **/predict (batch):** sample request/response ordered alignment shown
- **/recommend:** sample request/response and explanation of vector config

## Production Notes
- Observability: logs/metrics you added or recommend
- Reliability: retries/backoff for Qdrant, graceful 503s
- Performance: batching, model warmup, lazy vs eager load
- Security: input size caps, timeouts

## Future Work (Optional)
- Tests you’d add, refactors you deferred, known limitations
```

---

## Example Commands (Sanity Checks)

```bash
# Train
python -m ml.train --epochs 1 --batch_size 16 --lr 3e-5

# Run API
uvicorn app.main:app --port 8080 --reload

# Predict
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is fantastic"}' | jq

# Batch Predict
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Great", "Awful"]}' | jq

# Recommend (when Qdrant running and seeded)
curl -s -X POST localhost:8080/recommend \
  -H 'Content-Type: application/json' \
  -d '{"product_id": "SKU-123", "k": 5}' | jq
```

---

## License / Attribution

Internal challenge material. Do not distribute externally without permission.

