from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'synthetic_interactions.csv'
ART = ROOT / 'artifacts'
FEATURES = [
    'latency_sec','revision_delta','verification_clicks','self_reported_confidence',
    'task_difficulty','action_accept','ai_correct'
]
TARGET = 'overreliance_label'


def evaluate(model, X_test, y_test):
    p = model.predict_proba(X_test)[:,1]
    y = (p >= 0.5).astype(int)
    return {
        'roc_auc': round(float(roc_auc_score(y_test, p)), 4),
        'f1': round(float(f1_score(y_test, y)), 4),
        'accuracy': round(float(accuracy_score(y_test, y)), 4),
        'precision': round(float(precision_score(y_test, y)), 4),
        'recall': round(float(recall_score(y_test, y)), 4),
    }


def make_preprocessor():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    return ColumnTransformer([('num', numeric, FEATURES)], remainder='drop')


def main():
    ART.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    candidates = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            class_weight='balanced', random_state=42
        ),
    }

    scores = {}
    trained = {}
    for name, clf in candidates.items():
        pipe = Pipeline([
            ('prep', make_preprocessor()),
            ('clf', clf)
        ])
        pipe.fit(Xtr, ytr)
        scores[name] = evaluate(pipe, Xte, yte)
        trained[name] = pipe

    best_name = max(scores, key=lambda k: scores[k]['roc_auc'])
    best = trained[best_name]
    joblib.dump(best, ART / 'overreliance_model.joblib')
    (ART / 'metrics.json').write_text(
        json.dumps({'best_model': best_name, 'models': scores}, indent=2),
        encoding='utf-8'
    )
    (ART / 'feature_schema.json').write_text(
        json.dumps({'features': FEATURES, 'target': TARGET}, indent=2),
        encoding='utf-8'
    )
    print(json.dumps({'best_model': best_name, 'models': scores}, indent=2))


if __name__ == '__main__':
    main()
