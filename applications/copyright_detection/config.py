"""Configuración prevista del análisis de copyright.

Son decisiones ya tomadas para las siguientes issues, **no** una implementación:
ningún endpoint usa todavía el índice ni la métrica.

- ``SEGMENT_SECONDS``: duración de los bloques de audio. Debe coincidir con
  ``soupdb_vector.multimedia.audio_features.SEGMENT_SECONDS`` (un test lo
  verifica). No se importa de ahí para no arrastrar librosa/numpy a la base del
  API.
- ``INDEX_KIND`` / ``METRIC``: HNSW con distancia euclidiana, elegido por los
  benchmarks de la Parte 4 sobre 100 000 segmentos (Recall@10 ≈ 0.990,
  ≈ 12.3 ms/consulta).
"""

from typing import Final

SERVICE_NAME: Final = "soupdb-copyright-api"
API_PREFIX: Final = "/api/v1"

SEGMENT_SECONDS: Final = 5.0
INDEX_KIND: Final = "HNSW"
METRIC: Final = "euclidean"
