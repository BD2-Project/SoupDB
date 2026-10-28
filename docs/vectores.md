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

Compara fuerza bruta (exacta), IVF y HNSW, con euclidiana y coseno, k = 10.

**Sintético** (vectores gaussianos con semilla, consultas perturbadas):

```bash
uv run python -m benchmarks.scripts.compare_vector_indexes
uv run python -m benchmarks.scripts.compare_vector_indexes --sizes 1000 10000 100000
```

**Real** (#129): los `vectores.bin` de imágenes COCO y segmentos de audio FMA
(ver [Datasets multimedia](datasets_multimedia.md)):

```powershell
uv run python -m benchmarks.scripts.compare_vector_indexes `
    --real-data-dir D:/SoupDB-data/multimedia --sizes 1000 10000 100000 `
    --image-root D:/datasets/coco/train2017 --audio-root D:/datasets/fma_medium
```

Salidas en `benchmarks/results/`:

| Archivo | Contenido |
|---|---|
| `vector_indexes[_real]_<fecha>.csv` | una fila por medición |
| `…_<medio>_build.svg`, `…_<medio>_knn.svg` | tiempo contra tamaño (log-log); color = técnica, trazo = métrica |
| `…_real_<fecha>_gallery.html` | galería de aciertos y errores (solo modo real) |

Columnas del CSV, iguales en los dos modos (`media_type` = `synthetic`, `image` o `audio`):
`media_type, dataset_size, technique, metric, operation, run, elapsed_ms, disk_reads,
disk_writes, bytes_on_disk, memory_bytes, recall_at_k, dim, queries, k, params`.

### Cómo se mide en modo real

- **RIDs:** `RID(i // 64, i % 64)` por la posición `i` del registro en `vectores.bin`.
  Los mismos vectores y RIDs para las tres técnicas. Se guarda RID → (id, path, medio)
  para la galería.
- **Consultas:** 50 objetos del propio dataset, elegidos con
  `random.Random(20261024).sample`. Son las mismas para todas las técnicas y métricas.
- **Sin el propio objeto:** a cada técnica se le piden `k + 1` vecinos, se quita el RID
  de la consulta y se conservan los 10 primeros (`knn_ids`). El tiempo de consulta
  incluye pedir ese vecino de más, igual para las tres técnicas.
- **Recall@10** = |top-10 aproximado ∩ top-10 exacto| / 10, contra la fuerza bruta
  **de la misma métrica**: hay una verdad euclidiana y otra coseno.
- **HNSW** guarda la métrica de construcción: se construye un grafo por métrica. IVF
  y la fuerza bruta también se reconstruyen por métrica, como en el modo sintético.
- **Galería:** 3 consultas por dataset, elegidas de forma determinista entre las que
  más MISS tienen (a igualdad, la de menor posición). Muestra la imagen o el audio
  original (`file://` a COCO/FMA, sin copiar nada), el top-10 exacto y el de IVF/HNSW,
  cada resultado marcado HIT o MISS. Los audios suenan solo su segmento (`#t=inicio,fin`).

### Euclidiana y coseno sobre vectores reales

Los vectores TF-IDF están L2-normalizados (norma 1). En ese caso
`|a − b|² = 2 − 2·cos(a, b)`, así que las dos métricas producen **el mismo ranking**,
salvo empates y redondeo. Que la fuerza bruta euclidiana y la coseno coincidan es el
resultado esperado, no un error. Las diferencias de tiempo entre métricas vienen del
costo de cada fórmula. En IVF/HNSW pueden variar algo los candidatos visitados.

### Empates

Audio 10k contiene 28 vectores duplicados (segmentos casi idénticos, p. ej. silencio).
Con empates exactos, el orden entre ellos depende de la técnica, y una técnica puede
devolver un empatado distinto del que eligió la fuerza bruta. Eso puede bajar el
recall un poco sin que haya error.
