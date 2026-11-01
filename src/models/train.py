"""Entrena la referencia NLP/Naive Bayes sin leer el conjunto test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import joblib
import nltk
import sklearn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.nlp.preprocessing import SpanishTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/model'
LABELS = ['Atención al Cliente', 'Finanzas', 'Logística', 'Marketing', 'Soporte TI', 'Ventas']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'text', 'department'}.issubset(reader.fieldnames or ()):
            raise ValueError(f'{path}: faltan columnas text y department')
        rows = list(reader)
    if not rows or any(not r['text'].strip() or r['department'] not in LABELS for r in rows):
        raise ValueError(f'{path}: filas vacías o etiquetas desconocidas')
    return [r['text'] for r in rows], [r['department'] for r in rows]


def build_pipeline(method: str = 'stemming', ngrams: int = 1) -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=SpanishTokenizer(method), token_pattern=None,
                                  lowercase=False, ngram_range=(1, ngrams), min_df=1, max_df=1.0)),
        ('classifier', MultinomialNB(alpha=1.0)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['stemming', 'lemmatization'], default='stemming')
    parser.add_argument('--ngrams', type=int, choices=[1, 2], default=1)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/baseline')
    args = parser.parse_args()
    train_path, val_path = DATA / 'train.csv', DATA / 'validation.csv'
    train_texts, train_labels = load_data(train_path)
    val_texts, val_labels = load_data(val_path)
    if set(train_labels) != set(LABELS):
        raise ValueError('Entrenamiento debe contener las seis áreas.')
    model = build_pipeline(args.method, args.ngrams)
    model.fit(train_texts, train_labels)
    predictions = model.predict(val_texts)
    metrics = {
        'evaluation_split': 'validation',
        'input_sha256': {'train': digest(train_path), 'validation': digest(val_path)},
        'model': 'MultinomialNB',
        'parameters': {'method': args.method, 'remove_stopwords': True, 'ngrams': [1, args.ngrams],
                       'min_df': 1, 'max_df': 1.0, 'alpha': 1.0},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'nltk': nltk.__version__, 'spacy': spacy.__version__, 'joblib': joblib.__version__,
                     'spacy_model': 'es_core_news_sm 3.8.0' if args.method == 'lemmatization' else None},
        'counts': {'train': len(train_texts), 'validation': len(val_texts)},
        'vocabulary_size': len(model.named_steps['tfidf'].vocabulary_),
        'f1_macro': f1_score(val_labels, predictions, labels=LABELS, average='macro', zero_division=0),
        'classification_report': classification_report(val_labels, predictions, labels=LABELS,
                                                        output_dict=True, zero_division=0),
        'confusion_matrix_labels': LABELS,
        'confusion_matrix': confusion_matrix(val_labels, predictions, labels=LABELS).tolist(),
        'limitations': 'Referencia preliminar sobre validación sintética; test y evaluación humana no utilizados.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / 'pipeline.joblib')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"F1 macro (validación): {metrics['f1_macro']:.4f}")
    print(f"Vocabulario: {metrics['vocabulary_size']} términos; artefactos: {args.output_dir}")


if __name__ == '__main__':
    main()
