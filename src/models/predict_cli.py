"""Clasifica mensajes y compara pipelines entrenados con probabilidades y trazas NLP."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib

from src.nlp.preprocessing import normalize

MODEL = Path(__file__).resolve().parents[2] / 'artifacts/baseline/pipeline.joblib'
GREETINGS = {'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'ayuda'}


def load_model(path: Path):
    if not path.is_file():
        raise ValueError(f'No se encontró el modelo: {path}. '
                         'Indica uno existente con --model o entrena la referencia con: '
                         'python -m src.models.train')
    model = joblib.load(path)
    try:
        tfidf = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        if not (callable(getattr(tfidf.tokenizer, 'trace', None))
                and callable(getattr(classifier, 'predict_proba', None))
                and len(classifier.classes_) > 0):
            raise ValueError('Faltan trazas NLP, probabilidades o clases entrenadas.')
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('Se requiere un pipeline entrenado con los pasos tfidf y classifier, '
                         'trazas NLP y predict_proba.') from exc
    return model


def predict_message(model, message: str) -> dict:
    """Usa el preprocesamiento y las clases del artefacto, sin ajustar el modelo."""
    tfidf = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    trace = tfidf.tokenizer.trace(message)
    features = tfidf.transform([message])
    result = {'label': None, 'probabilities': {}, 'preprocessing': trace,
              'recognized_features': features.nnz, 'reason': None}
    if features.nnz == 0:
        result['reason'] = 'No se reconocieron términos del vocabulario; agrega más detalles.'
        return result
    probabilities = classifier.predict_proba(features)[0]
    result['probabilities'] = {str(label): float(probability)
                               for label, probability in zip(classifier.classes_, probabilities)}
    result['label'] = str(classifier.classes_[probabilities.argmax()])
    return result


def print_result(index: int, path: Path, model, result: dict) -> None:
    tfidf = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    trace = result['preprocessing']
    method = {'stemming': 'Stemming', 'lemmatization': 'Lematización'}[tfidf.tokenizer.method]
    print(f'\nModelo {index}: {type(classifier).__name__} | {path}')
    print('Preprocesamiento:')
    print(f"  Normalización: {trace['normalized']}")
    print(f"  Tokens: {trace['tokens']}")
    print(f"  Stop words retiradas: {trace['stopwords_removed']}")
    print(f"  Tokens conservados: {trace['tokens_kept']}")
    print(f"  {method}: {trace['tokens_transformed']}")
    print(f"  TF-IDF: n-gramas {tfidf.ngram_range}; "
          f"{result['recognized_features']} términos reconocidos.")
    if result['label'] is None:
        print(f"Sin área sugerida: {result['reason']}")
        return
    print(f"Área sugerida: {result['label']}")
    print('Probabilidades por área:')
    for label, probability in sorted(result['probabilities'].items(), key=lambda item: -item[1]):
        print(f'  {label:<20} {probability:7.2%}')


def print_comparison(results: list[dict]) -> None:
    print('\nComparación para el mismo mensaje:')
    print(f"{'Modelo':<10} {'Área sugerida':<22} {'Probabilidad':>12}")
    for index, result in enumerate(results, 1):
        label = result['label']
        probability = f"{result['probabilities'][label]:.2%}" if label is not None else '—'
        print(f"{f'Modelo {index}':<10} {label or 'Sin clasificación':<22} {probability:>12}")
    labels = {result['label'] for result in results}
    if None in labels:
        print('Hay modelos sin clasificación; revisa sus mensajes de diagnóstico.')
    elif len(labels) == 1:
        print(f'Los modelos coinciden en el área: {results[0]["label"]}.')
    else:
        print('Los modelos sugieren áreas diferentes.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', metavar='RUTA',
                        help='Ruta al pipeline.joblib; repetir para comparar modelos. '
                             'Por defecto usa artifacts/baseline/pipeline.joblib.')
    args = parser.parse_args(argv)
    models = []
    for path in args.model or [MODEL]:
        try:
            models.append((path, load_model(path)))
        except Exception as exc:
            print(f'No se pudo cargar {path}: {exc}', file=sys.stderr)
            return 1
    print('Clasificador de mensajes. Escribe salir para terminar.')
    print('Las probabilidades son estimaciones no calibradas; '
          'una cifra mayor no demuestra que un modelo sea mejor.')
    while True:
        try:
            message = input('\nEscribe tu mensaje: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nHasta luego.')
            return 0
        if message.casefold() == 'salir':
            print('Hasta luego.')
            return 0
        normalized = normalize(message)
        if not normalized or normalized in GREETINGS:
            print('Por favor, describe tu solicitud con más detalle.')
            continue
        results = []
        for index, (path, model) in enumerate(models, 1):
            try:
                result = predict_message(model, message)
                print_result(index, path, model, result)
            except Exception as exc:
                print(f'\nModelo {index} ({path}): no se pudo procesar el mensaje: {exc}')
                result = {'label': None, 'probabilities': {}}
            results.append(result)
        if len(results) > 1:
            print_comparison(results)


if __name__ == '__main__':
    raise SystemExit(main())
