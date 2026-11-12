# API de detección de copyright (Parte 5)

API REST de la aplicación de la Parte 5 (opción C): detectar fragmentos de un
audio que coinciden con canciones de un catálogo (copyright/remixes). El
frontend la consume por HTTP.

Es una **capa de aplicación** (`applications/copyright_detection/`), separada
del núcleo:

- no forma parte del wheel `soupdb` (que solo empaqueta `engine`) y el núcleo
  sigue sin dependencias externas;
- no toca el servidor TCP (`python -m engine`), el protocolo binario ni `rsoup`.

## Instalación y arranque

Las dependencias (FastAPI, python-multipart, uvicorn) están en el grupo
opcional `copyright-api` de `pyproject.toml`. El grupo `dev` lo incluye, así
que `uv sync` ya basta para correr los tests.

```bash
uv sync --group copyright-api
uv run --group copyright-api uvicorn applications.copyright_detection.app:app --reload
```

Por defecto escucha en `http://127.0.0.1:8000`. La documentación interactiva
queda en `/docs` y el esquema en `/openapi.json`.

## Endpoints

| Método | Ruta | Estado |
|---|---|---|
| `GET` | `/api/v1/health` | Funcional: `200 {"status": "ok", "service": "soupdb-copyright-api"}` |
| `POST` | `/api/v1/analyze` | Contrato definido; responde `501` (ver abajo) |

### `POST /api/v1/analyze`

Recibe un archivo de audio como `multipart/form-data` en el campo `file`.

!!! warning "Todavía no ejecuta detección"
    Mientras no exista el motor de detección responde
    `501 Not Implemented` con
    `{"detail": "El motor de detección de copyright aún no está implementado."}`.
    No devuelve coincidencias simuladas. Sin el campo `file` responde `422`.

La respuesta satisfactoria futura (`AnalysisResult`, documentada en OpenAPI):

```json
{
  "analysis_id": "string",
  "filename": "query.mp3",
  "segment_seconds": 5.0,
  "matches": [
    {
      "query_start": 0.0,
      "query_end": 15.0,
      "song_id": 1,
      "title": "Canción",
      "artist": "Artista",
      "source_start": 35.0,
      "source_end": 50.0,
      "score": 0.94
    }
  ]
}
```

`score` es un **score de similitud** en `[0, 1]` (mayor = más parecido). **No**
es una probabilidad de plagio, copyright ni infracción: todavía no hay un umbral
de copyright validado experimentalmente.

## Configuración prevista

Definida en `applications/copyright_detection/config.py` para las siguientes
issues; ningún endpoint la usa todavía:

| Parámetro | Valor | Origen |
|---|---|---|
| `SEGMENT_SECONDS` | `5.0` | Segmentación de audio de la Parte 4 (un test verifica que coincide con `soupdb_vector`) |
| `INDEX_KIND` | `HNSW` | Benchmarks de la Parte 4 sobre 100 000 segmentos |
| `METRIC` | `euclidean` | Recall@10 ≈ 0.990, ≈ 12.3 ms/consulta |

## Catálogo relacional

`applications/copyright_detection/catalog.py` (`CopyrightCatalog`) guarda el
catálogo en el **motor relacional real de SoupDB**: un `engine.common.catalog.Catalog`
abierto sobre una ruta explícita, que persiste en disco y se recarga al reabrir.
No usa otra base de datos. Ningún endpoint lo usa todavía.

```python
from pathlib import Path

from engine.common.catalog import Catalog
from applications.copyright_detection.catalog import CopyrightCatalog, Song

catalog = Catalog(Path("data/copyright"))
db = CopyrightCatalog(catalog)
db.initialize()  # crea las tablas que falten; nunca borra datos
db.add_song(Song(1, "Canción", "Artista", "tracks/song001.mp3", 183.5))
catalog.close()
```

| Tabla | Columnas |
|---|---|
| `songs` | `song_id INT`, `title TEXT`, `artist TEXT`, `audio_path TEXT`, `duration_seconds FLOAT` |
| `audio_segments` | `segment_id INT`, `song_id INT`, `segment_index INT`, `start_seconds FLOAT`, `end_seconds FLOAT`, `audio_ref TEXT` |

Hay índices B+ sobre `songs.song_id`, `audio_segments.segment_id` y
`audio_segments.song_id`. Los ids los asigna quien llama (SoupDB no tiene
autoincremento).

`audio_ref` identifica un segmento con la misma convención que la Parte 4
(`soupdb_vector.multimedia.audio_features.segment_id`):
`<audio_path>#segment=NNNNNN`, por ejemplo `tracks/song001.mp3#segment=000001`.
`audio_path` es relativo a la raíz del dataset. Esa columna es la que usará el
pipeline multimedia; **todavía no hay índice HNSW ni vectores sobre la tabla**.

SoupDB no tiene `UNIQUE`, `FOREIGN KEY`, `IF NOT EXISTS` ni `UPDATE` ejecutable,
y su `PRIMARY KEY` no impide duplicados. Por eso `CopyrightCatalog` comprueba
antes de cada inserción que los ids no estén repetidos, que el segmento apunte a
una canción existente, que una canción no repita `segment_index` y que
`audio_ref` sea la referencia canónica del segmento. Si una tabla ya existe con
otro schema, `initialize()` falla con `IncompatibleSchemaError`.

Limitación actual: esas comprobaciones se serializan con un lock de la propia
instancia, que solo protege un proceso. No sustituye restricciones del motor, no
coordina varios procesos y registrar una canción con sus segmentos no es atómico;
se resolverá al integrar la ingesta con transacciones.

## Ingesta de canciones

`applications/copyright_detection/ingestion.py` registra una canción original del
dataset en el catálogo. Requiere `soupdb-vector` (librosa), incluido en el grupo
`dev`; la API REST no importa este módulo.

```python
from applications.copyright_detection.ingestion import SongMetadata, ingest_song

result = ingest_song(
    db,  # CopyrightCatalog ya inicializado
    Path("datasets/originales"),  # raíz del dataset
    SongMetadata(1, "Canción", "Artista", "tracks/song001.wav"),
    first_segment_id=1,
)
```

- **Raíz del dataset:** `audio_path` es relativo a ella, con `/` y en forma
  canónica. Se rechazan rutas absolutas, `#`, `..` u otras rutas que salgan de la
  raíz, archivos inexistentes y directorios.
- **Segmentación:** la misma de la Parte 4, `iter_audio_segments` de
  `soupdb_vector.multimedia.audio_features`: bloques consecutivos de 5 s sin
  solapamiento, audio en mono a 22 050 Hz. El último bloque solo se conserva si
  dura al menos `MIN_SEGMENT_RATIO` (la mitad, 2.5 s): una pista de 13 s da
  0–5, 5–10 y 10–13; una de 12 s da 0–5 y 5–10, y la cola de 2 s no tiene fila.
  Si no sale ningún segmento, la ingesta falla y no registra la canción.
- **Intervalos:** `start_seconds = segment_index × 5.0` y
  `end_seconds = start_seconds + muestras_del_segmento / 22050`.
- **Duración:** `songs.duration_seconds` es la duración real decodificada
  (`len(load_audio(path)) / 22050`), incluida la cola descartada; no la suma de
  segmentos. El archivo se decodifica dos veces (duración y segmentos) porque la
  API pública del plugin no da ambos a la vez.
- **Ids:** el caller da `song_id` y `first_segment_id`; cada segmento recibe
  `first_segment_id + segment_index`. Antes de escribir se comprueba que ningún id
  esté ocupado, que la canción y su `audio_path` no estén ya registrados y que el
  audio se decodifique. Si algo falla no se escribe nada: `AudioIngestionError`
  (o `ValueError` si los metadatos o `first_segment_id` no son válidos).
- **Atomicidad:** la escritura posterior (canción y luego cada segmento) no es
  transaccional; un fallo a mitad puede dejar la canción con parte de sus segmentos.

Esta capa todavía **no extrae MFCC, no genera vectores ni construye HNSW**.

## Índice HNSW del catálogo

`applications/copyright_detection/indexing.py` indexa `audio_segments.audio_ref`
con el **HNSW real de SoupDB**, usando la integración SQL de `soupdb-vector`
(`MultimediaProfile` + `OnDiskVectorFactory` + `Catalog(..., ranked_factory=...)`):

```
audio_ref → MFCC (por segmento de 5 s) → codebook K=128 → TF-IDF L2 → HNSW (euclidean)
```

Orden de construcción, con rutas que da el caller (`CopyrightIndexConfig`:
raíz del dataset, `codebook.npz`, caché de descriptores y carpeta del índice):

1. Ingestar **todo** el catálogo (sección anterior).
2. `train_codebook(...)`: genera desde el catálogo un manifest `audio_segment`
   (rutas relativas, ordenado por `segment_id`; solo artefacto de construcción) y
   entrena con `build_vectors.train_from_manifest` un codebook de **K=128**
   palabras acústicas, que se guarda una vez y nunca se sobrescribe.
3. `open_catalog(db_path, config)`: abre la base con el perfil
   `copyright-audio-v1` (`kind="audio_segment"`) y valida que el codebook dé
   vectores de dimensión 128.
4. `ensure_index(catalog, config)`: si no existe, ejecuta

   ```sql
   CREATE INDEX copyright_audio_hnsw ON audio_segments (audio_ref) USING HNSW
   WITH METRIC=euclidean, DIMENSION=128, PROFILE='copyright-audio-v1'
   ```

   Si ya existe con esa configuración lo reutiliza sin reconstruirlo; si existe
   con otra, falla.

No se genera `vectores.bin`: la factoría vectoriza cada fila al crear el índice.
El HNSW guarda el **RID** de cada fila; una consulta resuelve cada hit a su fila
real de `audio_segments`, de donde salen `segment_id` y `song_id`:

```sql
SELECT segment_id, song_id, audio_ref, SCORE() AS score FROM audio_segments
WHERE SIMILAR_TO('tracks/a.wav#segment=000001', k=3) USING HNSW WITH METRIC=euclidean
ORDER BY score DESC
```

Con métrica euclidean, `SCORE()` es `1 / (1 + distancia)`: mayor es más parecido
y el propio segmento da 1.0. Es una similitud, no una probabilidad de copyright.

**Persistencia:** el HNSW y su metadata (`sql_index.json`, con la huella del
codebook, la raíz del dataset y el perfil) quedan en disco. Al reabrir con
`open_catalog` y la misma configuración se recargan sin reconstruir; si el
codebook, la raíz o el perfil cambiaron, la apertura falla.

**Limitaciones:**

- Con el índice creado, el motor bloquea `INSERT`/`DELETE` en `audio_segments`
  fuera de una `TransactionalSession`, así que la ingesta rechaza canciones
  nuevas. Para ampliar el catálogo habrá que integrar la ingesta con
  transacciones o reconstruir el índice.
- La base indexada debe abrirse siempre con `open_catalog`.

Todavía **no hay detector de un archivo completo, agregación temporal ni umbral
de copyright**; `/api/v1/analyze` sigue respondiendo 501.

## Búsqueda por segmentos de un candidato

`applications/copyright_detection/search.py` busca un audio candidato contra el
catálogo indexado:

```python
from applications.copyright_detection.indexing import open_catalog
from applications.copyright_detection.search import search_audio

catalog = open_catalog(db_path, config)  # índice ya creado (sección anterior)
result = search_audio(catalog, config, "queries/remix.wav", k=5)
```

- **Ubicación del candidato:** en esta etapa, `audio_path` es relativo a la raíz
  del perfil (`config.dataset_root`), porque el vectorizador del perfil solo
  resuelve rutas ahí. Se valida igual que los originales (relativo, `/`, sin `#`,
  dentro de la raíz, archivo existente). El candidato **no** se inserta en
  `songs`, `audio_segments` ni en el HNSW. Cada candidato distinto debe tener su
  propia ruta: la caché de descriptores usa la ruta como clave.
- **Segmentación:** la misma que los originales (`iter_audio_segments`): bloques
  de 5 s y el último solo si dura al menos 2.5 s. Un candidato de 13 s da 0–5,
  5–10 y 10–13; uno de 12 s da 0–5 y 5–10. Si no hay ningún segmento, falla
  antes de consultar.
- **Una consulta HNSW por segmento**, por SQL y el planner real:

  ```sql
  SELECT segment_id, song_id, segment_index, start_seconds, end_seconds, audio_ref,
         SCORE() AS score FROM audio_segments
  WHERE SIMILAR_TO('queries/remix.wav#segment=000000', k=5) USING HNSW WITH METRIC=euclidean
  ORDER BY score DESC
  ```

  El vectorizador aplica al segmento el mismo MFCC → codebook → TF-IDF L2 del
  índice, y cada hit se resuelve por RID a su fila real de `audio_segments`.
- **Resultado:** `AudioSearchResult(audio_path, k, segments)`. Cada
  `QuerySegmentResult` tiene `query_segment_index`, `query_start_seconds`,
  `query_end_seconds` y sus `matches` propios (no se mezclan entre segmentos).
  Cada `SegmentMatch` tiene `segment_id`, `song_id`, `title`, `artist`,
  `source_segment_index`, `source_start_seconds`, `source_end_seconds`,
  `audio_ref` y `score`.
- **`k`:** entero positivo; son los vecinos por segmento, no un número de
  detecciones. Si el catálogo tiene menos segmentos, se devuelven los que haya.
- **`score`:** `1 / (1 + distancia euclidiana)`; mayor es más parecido y un
  segmento idéntico da ≈ 1.0. **No es una probabilidad de copyright.**

Todavía no hay umbral, agregación temporal por canción ni decisión de copyright,
y `/api/v1/analyze` sigue respondiendo 501.
