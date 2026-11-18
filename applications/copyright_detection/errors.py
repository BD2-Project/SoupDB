"""Errores del servicio de análisis que la API REST traduce a HTTP.

Módulo ligero (sin librosa ni SoupDB) para que ``app.py`` pueda capturarlos sin
cargar el pipeline multimedia al importarse.
"""


class AnalysisError(Exception):
    """Base de los errores esperados del análisis."""


class InvalidAudioError(AnalysisError):
    """El archivo del usuario no sirve: vacío, no decodificable o sin segmentos (HTTP 422)."""


class AnalysisUnavailableError(AnalysisError):
    """La infraestructura no está lista: índice, perfil o servicio sin configurar (HTTP 503)."""
