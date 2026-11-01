"""Clasifica mensajes introducidos en la terminal con el modelo entrenado."""
from pathlib import Path
import sys

import joblib

from src.nlp.preprocessing import normalize

MODEL = Path(__file__).resolve().parents[2] / 'artifacts/baseline/pipeline.joblib'
GREETINGS = {'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'ayuda'}


def main() -> int:
    if not MODEL.is_file():
        print('No se encontró el modelo. Entrénalo con: python -m src.models.train', file=sys.stderr)
        return 1
    model = joblib.load(MODEL)
    print('Clasificador de mensajes. Escribe salir para terminar.')
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
        print(f'Área sugerida: {model.predict([message])[0]}')


if __name__ == '__main__':
    raise SystemExit(main())
