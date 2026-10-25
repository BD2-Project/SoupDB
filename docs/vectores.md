# Índices vectoriales (Parte 4)

Código en `engine/indexes/vector/`. Las tres estructuras implementan
`RankedIndex` y se validan con la misma suite de conformidad y contra la fuerza
bruta como oráculo de recall.

| Módulo | Contenido |
|---|---|
| `metrics.py` | Euclidiana, coseno y producto punto, sin numpy |
| `_record_file.py` | Arreglo de registros de tamaño fijo sobre páginas |
| `store.py` | `VectorStore`: vectores float32 con su RID y su norma |
| `brute_force.py` | `BruteForceIndex`: k-NN exacto, memoria O(k) |
| `kmeans.py` | K-Means (Lloyd + k-means++) y muestreo por reservorio |
| `ivf.py` | `IVFIndex`: centroides + listas invertidas paginadas |
| `hnsw.py` | `HNSWIndex`: grafo por capas persistido en páginas |
| `constants.py` | `IVF_NLIST`, `IVF_NPROBE`, `HNSW_M`, `HNSW_EF_*`, ... |

Toda la E/S pasa por `DiskManager` y `BufferManager`, así que los contadores
`reads`/`writes` miden el costo de cada consulta.

## Métricas y score

El score es siempre "mayor es mejor":

| Métrica | Score |
|---|---|
| `euclidean` | `1 / (1 + ‖q − v‖)` |
| `cosine` | `q·v / (‖q‖ ‖v‖)` (0 si alguna norma es 0) |
| `inner_product` | `q·v` |

## Uso

```python
from engine.common.rid import RID
from engine.indexes.vector import HNSWIndex, IVFIndex

# IVF: dos pasadas en streaming sobre una fuente re-iterable (entrenar, insertar).
ivf = IVFIndex.build("data/img.ivf", vectores, dim=128, nlist=316)
ivf.top_k(consulta, 10, metric="cosine", nprobe=8)

# HNSW: inserción incremental; el path es un directorio.
hnsw = HNSWIndex("data/img.hnsw", dim=128, m=16, ef_construction=100, metric="cosine")
for vector, rid in vectores:
    hnsw.insert(vector, rid)
hnsw.top_k(consulta, 10, ef=64)
```

## Layout en disco

**IVF** (un archivo):

```
página 0           cabecera: dim, nlist, count, inicio de centroides y directorio
centroides         (norma, c_0..c_{K-1}) por centroide
directorio         (head, tail, size) por lista
listas             (next, count) + registros (rid, norma, vector)
```

**HNSW** (un directorio):

```
vectors.db   VectorStore; el vid del vector es el id del nodo
layer0.db    por nodo: (nivel, inicio en upper.db, count, 2M vecinos)
upper.db     por (nodo, capa >= 1): (count, M vecinos)
```

Las razones de cada decisión están en [Decisiones de diseño](decisiones.md).

## Benchmark (2.4.4)

```bash
uv run python -m benchmarks.scripts.compare_vector_indexes
uv run python -m benchmarks.scripts.compare_vector_indexes --sizes 1000 10000 100000
```

Escribe `benchmarks/results/vector_indexes_<fecha>.csv` y dos gráficas SVG de
tiempo contra tamaño (consulta y construcción). Mientras `vectores.bin` (#108)
no tenga descriptores reales (#120), usa vectores sintéticos con semilla.
