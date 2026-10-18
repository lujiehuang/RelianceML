from pathlib import Path
import numpy as np
import pandas as pd


def make_dataset(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latency = np.clip(rng.gamma(shape=2.3, scale=12.0, size=n), 1, 120)
    revision_delta = np.clip(rng.beta(1.7, 4.2, size=n), 0, 1)
    verification_clicks = rng.poisson(0.9, size=n)
    confidence = np.clip(rng.normal(0.68, 0.18, size=n), 0, 1)
    task_difficulty = rng.integers(1, 6, size=n)
    ai_correct = rng.binomial(1, 0.78, size=n)
    action_accept = rng.binomial(1, np.clip(0.25 + 0.55*confidence - 0.35*revision_delta, 0.05, 0.95))

    logit = (
        1.6
        - 0.035*latency
        - 2.4*revision_delta
        - 0.65*verification_clicks
        + 1.2*confidence
        + 0.22*(6-task_difficulty)
        + 1.15*action_accept
        + 0.8*(1-ai_correct)
        + rng.normal(0, 0.75, size=n)
    )
    prob = 1/(1+np.exp(-logit))
    label = rng.binomial(1, np.clip(prob, 0.02, 0.98))

    return pd.DataFrame({
        'latency_sec': latency.round(2),
        'revision_delta': revision_delta.round(3),
        'verification_clicks': verification_clicks.astype(int),
        'self_reported_confidence': confidence.round(3),
        'task_difficulty': task_difficulty.astype(int),
        'action_accept': action_accept.astype(int),
        'ai_correct': ai_correct.astype(int),
        'overreliance_label': label.astype(int),
    })


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'data' / 'synthetic_interactions.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(out, index=False)
    print(f'wrote {len(df)} rows -> {out}')
