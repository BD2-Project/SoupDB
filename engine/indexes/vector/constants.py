"""Tunable parameters of the vector indexes.

Kept apart so the benchmarks can vary them without touching the structures.
"""

#: Métrica por defecto cuando la consulta no indica ``metric``.
DEFAULT_METRIC = "euclidean"

#: Páginas en el buffer pool de cada archivo cuando el índice abre sus propios managers.
VECTOR_BUFFER_CAPACITY = 64
