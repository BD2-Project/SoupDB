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
queda en `/docs` y el esquema en `/openapi.json`. La `app` global no tiene
analizador configurado (no abre bases al importarse), así que `/analyze` responde
`503`; para analizar se usa `create_app(CopyrightAnalyzer(...))`.

## Frontera REST de la aplicación

La API REST es la frontera de integración de la aplicación demo: el frontend
solo consume `/api/v1/...` por HTTP. No accede al `Catalog`, a los archivos del
dataset ni al índice; SoupDB sigue siendo la base multimodal subyacente
(relacional `songs`/`audio_segments` + HNSW vectorial) detrás de la API.

## Endpoints

| Método | Ruta | Respuesta |
|---|---|---|
| `GET` | `/api/v1/health` | `200 {"status": "ok", "service": "soupdb-copyright-api"}` |
| `POST` | `/api/v1/analyze` | `200 AnalysisResult` con la evidencia temporal (ver abajo) |

### `POST /api/v1/analyze`

Recibe un archivo de audio como `multipart/form-data` en el campo `file`:

```bash
curl -F "file=@remix.wav" http://127.0.0.1:8000/api/v1/analyze
```

Flujo:

```
POST multipart
→ staging único  <dataset_root>/queries/<analysis_id>/candidate.wav
→ segmentación 5 s (iter_audio_segments)
→ una consulta HNSW euclidean por segmento (k vecinos)
→ evidencia temporal (runs por canción)
→ JSON AnalysisResult
→ borrado del archivo y de su carpeta
```

La lógica vive en `applications/copyright_detection/analysis.py`
(`CopyrightAnalyzer`); FastAPI solo recibe el multipart, llama al servicio y
traduce errores. `create_app(analyzer)` recibe el servicio ya configurado con un
`Catalog` abierto por `open_catalog`, la configuración del índice, `search_k`
(vecinos por segmento, por defecto 5; no es un umbral) y la carpeta de staging.

Respuesta real (candidato idéntico a una canción de 13 s del catálogo, `search_k=2`):

```json
{
  "analysis_id": "4ff2b1526cd0458085ecc0f4720271b6",
  "filename": "remix.wav",
  "segment_seconds": 5.0,
  "matches": [
    {
      "query_start": 0.0, "query_end": 13.0,
      "song_id": 1, "title": "Alpha", "artist": "Ana",
      "source_start": 0.0, "source_end": 13.0,
      "score": 0.9999999752196517,
      "segment_count": 3,
      "min_score": 0.9999999708697074, "max_score": 0.9999999794376813,
      "alignment_offset_seconds": 0.0,
      "is_best_run": true
    },
    {
      "query_start": 0.0, "query_end": 5.0,
      "song_id": 1, "title": "Alpha", "artist": "Ana",
      "source_start": 5.0, "source_end": 10.0,
      "score": 0.41421356298841117,
      "segment_count": 1,
      "min_score": 0.41421356298841117, "max_score": 0.41421356298841117,
      "alignment_offset_seconds": 5.0,
      "is_best_run": false
    }
  ]
}
```

(Recortado: la respuesta real trae dos runs más de un segmento con score ≈ 0.414.)

Cada elemento de `matches` es una **run de evidencia temporal** (ver "Evidencia
temporal"):

| Campo | Significado |
|---|---|
| `query_start`, `query_end` | Tramo del audio subido (s); el último segmento puede durar menos de 5 s |
| `song_id`, `title`, `artist` | Canción del catálogo |
| `source_start`, `source_end` | Tramo alineado de la canción (s) |
| `score` | **Media de los scores de similitud** de los segmentos de la run (euclidean: `1 / (1 + distancia)`, en `[0, 1]`) |
| `segment_count` | Segmentos consecutivos de 5 s que forman la run |
| `min_score`, `max_score` | Menor y mayor score de similitud de la run |
| `alignment_offset_seconds` | `source_start - query_start` |
| `is_best_run` | `true` en exactamente una run por canción: la de más evidencia temporal |

Se devuelven **todas** las runs, sin filtrar por score ni por número de
segmentos: canciones de más a menos coherentes y, dentro de cada una, en orden
temporal del audio subido. Si no hay evidencia, `matches` es `[]` (HTTP 200).

`analysis_id` es un `uuid4` y es el mismo de la carpeta de staging; `filename`
es solo el nombre presentado por el cliente, sin componentes de ruta
(`../../x.wav` → `x.wav`), y nunca decide dónde se escribe el archivo.

!!! warning "Evidencia, no veredicto"
    La API devuelve evidencia de similitud acústica y temporal. **No determina
    copyright**, no prueba una infracción ni da una probabilidad. No hay umbral:
    cuál evidencia es significativa se decidirá aparte y deberá calibrarse
    experimentalmente. El frontend puede mostrar `score * 100` siempre como
    "similitud", no como probabilidad.

**Errores:**

| Código | Cuándo |
|---|---|
| `422` | Falta `file` (detalle de validación de FastAPI), o el audio está vacío, no se puede decodificar o no produce ningún segmento (≥ 2.5 s) |
| `503` | Análisis no configurado (`create_app()` sin analizador) o índice/perfil no disponible |
| `500` | Error interno inesperado; respuesta genérica sin traceback (el detalle queda en el log) |

**Staging y limpieza.** Cada análisis escribe el upload por bloques en una
carpeta nueva `queries/<analysis_id>/`, porque la caché de MFCC del perfil usa la
ruta absoluta como clave: reutilizar una ruta devolvería descriptores de un audio
anterior. El archivo y su carpeta se borran siempre al terminar, también si el
audio es inválido o la búsqueda falla. La carpeta raíz `queries/`, la caché, el
codebook y el índice no se tocan. La caché conserva entradas de esas rutas
temporales ya borradas (nunca se reutilizan); limpiarlas es una mejora futura.
El candidato nunca se inserta en `songs`, `audio_segments` ni en el HNSW.

**Concurrencia.** El `Catalog` compartido no está documentado como thread-safe:
el servicio serializa la búsqueda y la evidencia con un `RLock`. Las escrituras de
staging de análisis simultáneos van en paralelo y nunca colisionan.

## Configuración prevista

Definida en `applications/copyright_detection/config.py`:

| Parámetro | Valor | Origen |
|---|---|---|
| `SEGMENT_SECONDS` | `5.0` | Segmentación de audio de la Parte 4 (un test verifica que coincide con `soupdb_vector`) |
| `INDEX_KIND` | `HNSW` | Benchmarks de la Parte 4 sobre 100 000 segmentos |
| `METRIC` | `euclidean` | Recall@10 ≈ 0.990, ≈ 12.3 ms/consulta |

## Catálogo relacional

`applications/copyright_detection/catalog.py` (`CopyrightCatalog`) guarda el
catálogo en el **motor relacional real de SoupDB**: un `engine.common.catalog.Catalog`
abierto sobre una ruta explícita, que persiste en disco y se recarga al reabrir.
No usa otra base de datos.

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
pipeline multimedia; el índice HNSW se crea aparte (ver "Índice HNSW del catálogo").

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

Esta capa **no extrae MFCC, no genera vectores ni construye HNSW**.

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

El índice no aplica umbral ni decide copyright; el análisis completo de un
archivo se expone en `POST /api/v1/analyze`.

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

Esta capa no aplica umbral ni agrega por canción (eso lo hace la evidencia
temporal) y no decide copyright.

## Evidencia temporal

`applications/copyright_detection/evidence.py` convierte el resultado de la
búsqueda por segmentos en evidencia temporal por canción. Es una transformación
pura: no lee audio, no ejecuta SQL y no vuelve a buscar.

```python
from applications.copyright_detection.evidence import build_temporal_evidence

evidence = build_temporal_evidence(search_audio(catalog, config, "queries/remix.wav", k=5))
```

- **Alineamiento:** para cada match,
  `alignment_offset_segments = source_segment_index - query_segment_index`.
- **Agrupación:** por canción (`song_id`) y, dentro de ella, por offset.
- **Runs:** dentro de cada grupo, ordenado por `query_segment_index`, una run
  continúa solo si el siguiente segmento de consulta es el inmediato (`+1`), y
  con él el de origen; un hueco o un cambio de offset empiezan otra run.
- **Intervalos:** candidato desde el inicio del primer punto hasta el fin del
  último; igual en el original. Los segmentos finales cortos conservan su fin
  real (por ejemplo 10–13 s, no 10–15).
- **Métricas por run:** `segment_count`, `mean_score` (media aritmética),
  `min_score` y `max_score`, sin redondear.
- **`best_run` de cada canción:** mayor `segment_count`; luego mayor
  `mean_score`; mayor `min_score`; menor `query_start_seconds`; menor
  `source_start_seconds`. La continuidad pesa más que un pico de score aislado.
- **Orden de canciones:** por su `best_run` (`segment_count`, `mean_score`,
  `min_score` descendentes) y luego `song_id`. Significa "evidencia temporal más
  coherente primero", **no** "más probable que infrinja".
- Las runs de una canción van en orden temporal del candidato.

Ejemplo: matches `q0→A#10 (0.94)`, `q1→A#11 (0.92)`, `q2→A#12 (0.95)` forman una
run de la canción A con offset 10: candidato 0–15 s ↔ original 50–65 s,
`segment_count=3`, `mean_score≈0.9367`, `min_score=0.92`, `max_score=0.95`.

**Sin umbral, a propósito.** Se usan todos los matches recibidos, también los de
score bajo; una run con scores bajos sigue siendo una run estructural. La
decisión de qué evidencia es significativa (y cualquier umbral, que habrá que
calibrar experimentalmente) pertenece a una capa posterior. `score` sigue siendo
la similitud del motor, no una probabilidad de copyright.
