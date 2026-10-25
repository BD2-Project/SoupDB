"""Tunable parameters of the vector indexes.

Kept apart so the benchmarks can vary them without touching the structures.
"""

#: Métrica por defecto cuando la consulta no indica ``metric``.
DEFAULT_METRIC = "euclidean"

#: Páginas en el buffer pool de cada archivo cuando el índice abre sus propios managers.
VECTOR_BUFFER_CAPACITY = 64

#: Número de listas (centroides) del IVF; la regla habitual es ~sqrt(N).
IVF_NLIST = 32

#: Listas que se visitan por consulta.
IVF_NPROBE = 4

#: Iteraciones de Lloyd al entrenar los centroides.
IVF_KMEANS_ITERATIONS = 20

#: Tope de la muestra con la que se entrena K-Means; acota la memoria del entrenamiento.
IVF_TRAIN_SAMPLE = 10_000
