# RelianceML

**Behavioral over-reliance prediction for Human–LLM interaction**

RelianceML is a compact ML engineering project that extends the ReflectAI idea from a hand-crafted heuristic toward a trainable prediction pipeline.

> Goal: turn interaction traces into an estimated probability of over-reliance, then expose that prediction through an API for adaptive HCI interventions.

## Engineering capabilities demonstrated

- synthetic data generation for controlled prototyping
- feature engineering from interaction traces
- reproducible train/test splitting
- model comparison (Logistic Regression vs Random Forest)
- ROC-AUC / F1 / precision / recall evaluation
- model serialization with `joblib`
- typed FastAPI inference endpoint
- unit tests
- clear separation between data, training, serving, and artifacts

## Important research note

The included dataset is **synthetic** and exists only to demonstrate the engineering pipeline. The resulting classifier must **not** be interpreted as a scientifically validated detector of human over-reliance. A real study would require behaviorally grounded labels and external validation.

## Architecture

```text
Interaction traces
      │
      ▼
Feature table
(latency, revision, verification, confidence, ...)
      │
      ▼
Train / validation split
      │
      ├── Logistic Regression
      └── Random Forest
      │
      ▼
Model evaluation
(ROC-AUC, F1, precision, recall)
      │
      ▼
Serialized best model
      │
      ▼
FastAPI /predict endpoint
      │
      ▼
Adaptive interface trigger
```

## Quick start

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
uvicorn src.api:app --reload
```

Then open:

- API docs: `http://127.0.0.1:8000/docs`
- health: `http://127.0.0.1:8000/health`

Example request:

```json
{
  "latency_sec": 5.2,
  "revision_delta": 0.03,
  "verification_clicks": 0,
  "self_reported_confidence": 0.91,
  "task_difficulty": 2,
  "action_accept": 1,
  "ai_correct": 0
}
```

## Reproducible demo metrics

The committed `artifacts/metrics.json` records one reproducible run on synthetic demonstration data.

- Logistic Regression ROC-AUC: **0.7769**
- Random Forest ROC-AUC: **0.7577**
- Random Forest F1: **0.8214**

These values are engineering-demo results only, not empirical findings about human behavior.

## Research roadmap

1. Replace synthetic labels with ground-truth behavioral outcomes from controlled tasks.
2. Add calibration analysis and threshold selection.
3. Add SHAP or coefficient-based explanations.
4. Integrate the prediction API with ReflectAI.
5. Evaluate whether adaptive triggering reduces harmful over-reliance without creating excessive interaction friction.

## Author

Lujie Huang  
https://lujiehuang.github.io/
