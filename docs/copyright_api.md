# Aplicación de similitud de audio (Parte 5)

## Propósito

La aplicación de la Parte 5 (opción C) analiza la **similitud entre un audio consultado
y las canciones originales del catálogo** para detectar posibles coincidencias
(copyright/remixes). Para cada tramo del audio subido indica qué canciones y qué tramos
de ellas se parecen, con qué similitud y con qué coherencia temporal.

!!! warning "Evidencia, no veredicto"
    La aplicación devuelve **similitud acústica y evidencia temporal**. No determina
    copyright, no prueba una infracción, no da probabilidades y no aplica ningún umbral.
    Qué evidencia es significativa se decidirá aparte y deberá calibrarse
    experimentalmente.

Integra dos capacidades de SoupDB:

- **Relacional:** tablas `songs` y `audio_segments` en el motor SoupDB.
- **Multimedia/vectorial:** audio → MFCC → codebook K=128 → TF-IDF L2 → HNSW (euclidean)
  sobre `audio_segments.audio_ref`.

## Arquitectura

```mermaid
flowchart TD
    FE["Frontend"] -->|"HTTP REST: multipart / JSON<br/>/api/v1/..."| API["FastAPI (app.py)"]
    API --> AN["CopyrightAnalyzer (analysis.py)"]
    AN --> SE["search_audio: una consulta SQL SIMILAR_TO por segmento"]
    AN --> EV["build_temporal_evidence: runs por canción"]
    SE --> CAT[("SoupDB Catalog<br/>songs / audio_segments")]
    SE --> HN[("HNSW sobre audio_ref<br/>MFCC → K=128 → TF-IDF L2")]
    AN -->|"guarda el resultado"| HI["AnalysisHistory (history.py)"]
    HI --> HT[("SoupDB Catalog<br/>analyses / analysis_matches")]
    API --> LI["CatalogLibrary (library.py)<br/>solo lectura + audio original"]
    LI --> CAT
    API --> EX["export.py: JSON / CSV"]
```

La API FastAPI es el adaptador REST de la aplicación sobre las capacidades relacionales y
vectoriales de SoupDB.

- El **núcleo SoupDB** (`engine/`) no tiene REST ni GraphQL. Su interfaz remota nativa es
  el protocolo binario sobre TCP/IP en el puerto `55432` (`python -m engine`), que usa el
  driver `rsoup`; ver [Driver](driver.md) y [Protocolo](protocolo.md). La aplicación no
  lo modifica.
- La **aplicación de la Parte 5** (`applications/copyright_detection/`) usa SoupDB dentro
  del mismo proceso (`Catalog` + SQL + HNSW) y publica una API REST. Esa API es la única
  frontera del frontend: solo consume `/api/v1/...` y nunca toca el `Catalog`, el HNSW,
  los archivos internos, el protocolo TCP ni SQL.

La aplicación no forma parte del wheel `soupdb`, que solo empaqueta `engine`, y el núcleo
sigue sin dependencias externas.

| Módulo | Papel |
|---|---|
| `settings.py` | Variables `COPYRIGHT_*` → `RuntimeSettings` (solo parsing, stdlib) |
| `runtime.py` | Servidor ejecutable: `lifespan` que abre y cierra el `Catalog` |
| `app.py` | `create_app`: rutas, errores HTTP y CORS (import ligero) |
| `analysis.py` | `CopyrightAnalyzer`: staging → búsqueda → evidencia → historial → JSON |
| `history.py` | `AnalysisHistory`: tablas `analyses`/`analysis_matches` en SoupDB |
| `library.py` | `CatalogLibrary`: catálogo de solo lectura y resolución segura del audio |
| `export.py` | Exportación de un análisis guardado a JSON o CSV |
| `catalog.py`, `ingestion.py` | Catálogo relacional e ingesta de originales |
| `indexing.py` | Codebook K=128 y HNSW por SQL |
| `search.py`, `evidence.py` | Búsqueda por segmentos y evidencia temporal |
| `demo.py` | Preparación de una demo sintética |
| `fma_catalog.py` | Preparación del catálogo real FMA Small |

## Instalación

```bash
uv sync --group copyright-api
```

El grupo `copyright-api` instala todo lo que necesita el servidor real: FastAPI,
python-multipart, uvicorn y `soupdb-vector`, que a su vez trae librosa, numpy,
scikit-learn, soundfile y el resto. No depende del grupo `dev`:
`uv sync --no-dev --group copyright-api` instala solo el runtime. El grupo `dev` incluye
`copyright-api`, así que `uv sync` basta para correr los tests.

## Preparar la demo

La demo es autocontenida. Genera en runtime tres canciones sintéticas deterministas
(Alpha 20 s, Beta 15 s y Gamma 12.5 s), sin material con copyright y sin WAV
versionados, y prepara la base con las APIs reales:

```
WAV originales → CopyrightCatalog.initialize() → ingest_song()
→ train_codebook(K=128) → open_catalog() → ensure_index(HNSW euclidean) → close()
```

```bash
uv run --group copyright-api python -m applications.copyright_detection.demo prepare --root .demo/copyright
```

Estructura (`.demo/` está en `.gitignore`):

```
.demo/copyright/
  dataset/tracks/{alpha,beta,gamma}.wav   originales
  dataset/queries/                        staging de uploads (vacío)
  db/                                     base SoupDB (songs: 3, audio_segments: 10)
  artifacts/codebook.npz                  codebook K=128 + IDF
  artifacts/cache/                        caché de MFCC
  artifacts/vectors/                      HNSW persistente
  artifacts/training_manifest.jsonl       manifest de entrenamiento
  sample_query.wav                        audio de prueba (idéntico a Alpha)
  demo.json                               marca de demo completa
```

`sample_query.wav` es solo un input de usuario: está fuera del dataset y no se registra
en SoupDB. El comando imprime las rutas y las variables `COPYRIGHT_*` listas para copiar.

Si `--root` ya contiene una demo completa, se reutiliza sin modificarla. Si existe con
otro contenido o con una preparación interrumpida, el comando falla y pide borrarlo a
mano: la herramienta nunca borra directorios.

## Catálogo real: FMA Small

Hay dos fuentes de canciones con propósitos distintos:

| Uso | Audio | Comando |
|---|---|---|
| Tests automáticos | sintético (Alpha/Beta/Gamma), generado en runtime | `...demo prepare` |
| Demo académica | real: subconjunto balanceado de **FMA Small** | `...fma_catalog prepare` |

La demo sintética sigue siendo la base determinista de los tests; el catálogo FMA es una
capa adicional (`applications/copyright_detection/fma_catalog.py`) que reutiliza las mismas APIs (`ingest_song`, `train_codebook`, `open_catalog`,
`ensure_index`, `require_index`) sin reimplementar MFCC, K-Means, TF-IDF ni HNSW.

### Dataset

[Free Music Archive](https://github.com/mdeff/fma) (FMA) es un dataset de investigación
musical con audio publicado bajo licencias Creative Commons; la licencia **varía por
pista** (columna `track/license` de `tracks.csv`, que el manifest conserva). Se usa
**FMA Small**: 8 000 clips de ≈30 s repartidos en 8 géneros balanceados (Electronic,
Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock).

El dataset **no se versiona ni se descarga automáticamente**. Hay que bajar de la fuente
oficial (repositorio `mdeff/fma`) y descomprimir:

- `fma_small.zip` (≈7.2 GiB): audio en `fma_small/<3 dígitos>/<track_id de 6 dígitos>.mp3`
  (`track_id=2` → `000/000002.mp3`).
- `fma_metadata.zip`: `tracks.csv`, con cabecera doble (`header=[0, 1]` en pandas).

`--audio-root` es la carpeta que contiene `000/`, `001/`, ...; esa misma carpeta es el
`dataset_root` del perfil y la variable `COPYRIGHT_DATASET_ROOT` del servidor. Los MP3
no se copian.

### Selección balanceada

Por defecto se seleccionan **10 pistas de cada uno de los 8 géneros (80 pistas)**;
`--per-genre` lo cambia. Es nuestra configuración de demo, no un requisito del
enunciado. La selección es determinista (sin azar):

1. se leen las filas de `tracks.csv` con `set/subset == small` (módulo `csv` de la stdlib);
2. se agrupan por `track/genre_top` y se ordenan por `track_id`;
3. en cada género se toman las primeras pistas cuyo MP3 existe **y** se ingesta bien.

Una pista cuyo MP3 falta (`missing_file`) o no se decodifica (`ingestion_failed`) se
registra en `skipped_tracks` y se prueba la siguiente del mismo género. `ingest_song`
valida y decodifica antes del primer `INSERT`, así que una pista descartada no deja
filas. Si un género no llega a `per_genre` pistas válidas, el comando aborta con un
mensaje claro.

`song_id` es el `track_id` de FMA (trazabilidad; cabe en el `INT` de 32 bits). Los
`segment_id` son un contador global desde 1: tras cada pista,
`next_segment_id = último segment_id + 1`, sin suponer 6 segmentos por pista.

### Preparar el catálogo

Las canciones se cargan **antes** de arrancar el backend:

```bash
uv run --group copyright-api python -m applications.copyright_detection.fma_catalog prepare \
  --audio-root D:/datasets/fma_small \
  --tracks-csv D:/datasets/fma_metadata/tracks.csv \
  --output-root .demo/fma --per-genre 10
```

```
FMA catalog prepared
Songs: 80
Segments: ~480
Genres: 8
Skipped tracks: 0
HNSW: ready
DB: ...
```

Un clip de ≈30 s da 6 segmentos de 5 s, así que 80 pistas dan **≈480 segmentos**; el
número exacto depende de la duración real de cada clip y aparece en el resumen y en el
manifest. Tras el resumen se imprimen las variables `COPYRIGHT_*` para Bash y
PowerShell. `--seed` (por defecto 0) solo afecta a K-Means.

Estructura (`.demo/` ya está en `.gitignore`):

```
.demo/fma/
  db/                                 base SoupDB (songs, audio_segments)
  artifacts/codebook.npz              codebook K=128 + IDF
  artifacts/cache/                    caché de MFCC
  artifacts/vectors/                  HNSW persistente
  artifacts/training_manifest.jsonl   manifest de entrenamiento
  catalog_manifest.json               resumen (se escribe al final)
  sample_query.mp3                    solo tras make-query
```

`catalog_manifest.json` no guarda rutas absolutas. Incluye `dataset`, `source`,
`number_of_songs`, `number_of_segments`, `genres`, `per_genre`, `segment_seconds`,
`index`, `metric`, `codebook_k`, `skipped_tracks` y, por pista seleccionada, `track_id`,
`song_id`, `title`, `artist`, `genre`, `audio_path`, `duration_seconds`,
`segment_count`, el rango de `segment_id` y `license`. Si `--output-root` ya tiene un
catálogo completo con el mismo `--per-genre`, se reutiliza; con otro contenido el
comando falla y pide borrarlo a mano.

El HNSW contiene los segmentos de esas canciones. Como el índice bloquea nuevas
inserciones (ver Limitaciones), cambiar el catálogo exige prepararlo de nuevo en otro
`--output-root`.

### Consultas: smoke test y demo real

Cada upload a `POST /api/v1/analyze` se segmenta y se compara contra los segmentos ya
indexados del catálogo. El staging de uploads se escribe en
`<COPYRIGHT_DATASET_ROOT>/queries/` (configurable con `COPYRIGHT_STAGING_DIR`), así que
la carpeta de FMA debe ser escribible.

**Smoke test.** `make-query` copia una pista del catálogo a `output_root/sample_query.mp3`
(fuera del `dataset_root`; no se registra en SoupDB). Por defecto usa la primera pista
seleccionada; `--track-id` elige otra y `--overwrite` reemplaza un `sample_query.mp3`
existente:

```bash
uv run --group copyright-api python -m applications.copyright_detection.fma_catalog make-query \
  --audio-root D:/datasets/fma_small --output-root .demo/fma
curl -F "file=@.demo/fma/sample_query.mp3" http://127.0.0.1:8000/api/v1/analyze
```

Esperado: esa canción sale primera, con score muy alto y una run temporal continua
(offset ≈ 0). Es una comprobación de que el pipeline funciona, **no** una evaluación de
copyright: la consulta es idéntica a una pista indexada.

**Demo real.** Se sube una canción, remix o cover externo que **no** está registrado
(también puede sustituirse `sample_query.mp3` por un archivo propio). El sistema compara
sus segmentos contra las pistas indexadas. No hay garantía de coincidencia: si el audio
no contiene material del catálogo, los resultados serán solo los vecinos más cercanos,
con scores bajos.

## Ejecutar el servidor

El runtime asume una base ya preparada: **no entrena el codebook ni crea o reconstruye
el índice**.

- **Al arrancar** (`lifespan` de FastAPI) abre el `Catalog` con `open_catalog`, comprueba
  con `require_index` que el HNSW existe y está cargado, crea si faltan las tablas del
  historial (`analyses`, `analysis_matches`) y construye el `CopyrightAnalyzer`, el
  catálogo de solo lectura y el historial, que comparten un único `RLock`.
- **Al apagar** cierra el `Catalog` una sola vez; al cerrarse, escribe en disco el
  historial pendiente.
- **Si falta** una variable, un directorio o el índice, el servidor no arranca y cierra lo
  que hubiera abierto.

Bash:

```bash
export COPYRIGHT_DB_PATH=.demo/copyright/db
export COPYRIGHT_DATASET_ROOT=.demo/copyright/dataset
export COPYRIGHT_CODEBOOK_PATH=.demo/copyright/artifacts/codebook.npz
export COPYRIGHT_DESCRIPTOR_CACHE=.demo/copyright/artifacts/cache
export COPYRIGHT_VECTOR_STORAGE=.demo/copyright/artifacts/vectors
uv run --group copyright-api python -m applications.copyright_detection.runtime
```

PowerShell:

```powershell
$env:COPYRIGHT_DB_PATH=".demo/copyright/db"
$env:COPYRIGHT_DATASET_ROOT=".demo/copyright/dataset"
$env:COPYRIGHT_CODEBOOK_PATH=".demo/copyright/artifacts/codebook.npz"
$env:COPYRIGHT_DESCRIPTOR_CACHE=".demo/copyright/artifacts/cache"
$env:COPYRIGHT_VECTOR_STORAGE=".demo/copyright/artifacts/vectors"
uv run --group copyright-api python -m applications.copyright_detection.runtime
```

También se puede arrancar con uvicorn. En ese caso el host y el puerto salen de las
opciones de uvicorn, no de `COPYRIGHT_HOST`/`COPYRIGHT_PORT`:

```bash
uv run --group copyright-api uvicorn applications.copyright_detection.runtime:app --port 8000
```

Prueba:

```bash
curl -F "file=@.demo/copyright/sample_query.wav" http://127.0.0.1:8000/api/v1/analyze
```

Responde HTTP 200. La primera run es la de Alpha (0–20 s del audio ↔ 0–20 s de la
canción), con `segment_count=4`, `is_best_run=true` y score ≈ 1.0; ver el ejemplo de la
sección API.

**Variables de entorno.** Las rutas relativas se resuelven contra el directorio de
trabajo del proceso. Una variable opcional vacía o ausente toma su valor por defecto. Un
valor inválido detiene el arranque; nunca se sustituye en silencio por el valor por
defecto.

| Variable | Por defecto | Validación |
|---|---|---|
| `COPYRIGHT_DB_PATH` | obligatoria | directorio existente con la base |
| `COPYRIGHT_DATASET_ROOT` | obligatoria | directorio del dataset (escribible: ahí va el staging) |
| `COPYRIGHT_CODEBOOK_PATH` | obligatoria | `.npz` existente que produzca vectores de dimensión 128 |
| `COPYRIGHT_DESCRIPTOR_CACHE` | obligatoria | carpeta de la caché de MFCC |
| `COPYRIGHT_VECTOR_STORAGE` | obligatoria | directorio existente con el HNSW |
| `COPYRIGHT_SEARCH_K` | `5` | entero decimal entre 1 y 1000 |
| `COPYRIGHT_STAGING_DIR` | `queries` | ruta relativa canónica con `/` |
| `COPYRIGHT_HOST` | `127.0.0.1` | interfaz de escucha (solo con `python -m ...runtime`) |
| `COPYRIGHT_PORT` | `8000` | entero entre 1 y 65535 (solo con `python -m ...runtime`) |
| `COPYRIGHT_CORS_ORIGINS` | vacío | orígenes `http(s)://host[:puerto]` separados por comas |

**CORS.** Con la variable vacía no se habilita CORS. Para un frontend de desarrollo en
otro origen:

```bash
export COPYRIGHT_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

Se permiten los métodos `GET` y `POST` y las cabeceras `Accept` y `Content-Type`, que es
lo que necesita `POST multipart /api/v1/analyze`. El valor `*` se rechaza: los orígenes
se enumeran uno a uno.

`applications.copyright_detection.app:app` sigue existiendo como app sin analizador: al
importarse no lee el entorno ni abre nada, y responde 503 en `/analyze`. El servidor
ejecutable es `runtime`.

## API

La documentación interactiva está en `/docs` y el esquema en `/openapi.json`.

| Método | Ruta | Respuesta |
|---|---|---|
| `GET` | `/api/v1/health` | `200 {"status": "ok", "service": "soupdb-copyright-api"}` |
| `POST` | `/api/v1/analyze` | `200 AnalysisResult` con la evidencia temporal; se guarda en el historial |
| `GET` | `/api/v1/catalog` | `200 CatalogResponse`: canciones del catálogo (solo lectura) |
| `GET` | `/api/v1/catalog/{song_id}` | `200 CatalogSongDetail`: canción y sus segmentos; `404` si no existe |
| `GET` | `/api/v1/catalog/{song_id}/audio` | `200`/`206` audio original (admite `Range`); `404` si no existe |
| `GET` | `/api/v1/analyses` | `200 AnalysisSummary[]`: historial, el más reciente primero |
| `GET` | `/api/v1/analyses/{analysis_id}` | `200 StoredAnalysis`: resultado guardado; `404` si no existe |
| `GET` | `/api/v1/analyses/{analysis_id}/export?format=json\|csv` | descarga JSON o CSV; `404` si no existe |

Todos los endpoints nuevos son de lectura. Sin servicio configurado (la `app` de
`app.py` sin `lifespan`) responden `503`. Ver [API para el frontend](#api-para-el-frontend).

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
traduce errores. `search_k` (vecinos por segmento, `COPYRIGHT_SEARCH_K`, por
defecto 5) no es un umbral.

Respuesta real del servidor de la demo (`sample_query.wav`, idéntico a Alpha,
`search_k=5`), recortada a la primera run de dos canciones:

```json
{
  "analysis_id": "aeb1eb30c00940d196be845ac2adf00f",
  "filename": "sample_query.wav",
  "segment_seconds": 5.0,
  "matches": [
    {
      "query_start": 0.0, "query_end": 20.0,
      "song_id": 1, "title": "Alpha", "artist": "Demo Ensemble",
      "source_start": 0.0, "source_end": 20.0,
      "score": 0.999999976788879,
      "segment_count": 4,
      "min_score": 0.9999999735362719, "max_score": 0.9999999802395061,
      "alignment_offset_seconds": 0.0,
      "is_best_run": true
    },
    {
      "query_start": 0.0, "query_end": 10.0,
      "song_id": 3, "title": "Gamma", "artist": "Demo Ensemble",
      "source_start": 0.0, "source_end": 10.0,
      "score": 0.41421356292998734,
      "segment_count": 2,
      "min_score": 0.41421356229651984, "max_score": 0.4142135635634548,
      "alignment_offset_seconds": 0.0,
      "is_best_run": true
    }
  ]
}
```

(La respuesta completa trae 13 runs: las de Alpha con otros offsets y las de Gamma y
Beta, todas con score ≈ 0.414. Ese valor es `1 / (1 + √2)`, la distancia entre vectores
TF-IDF L2 sin palabras acústicas en común. Aparecen porque no hay umbral.)

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
| `503` | Análisis no configurado (app sin analizador, como `app.py:app`) o índice/perfil no disponible |
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

## API para el frontend

Lo que el frontend puede hacer con capacidades reales del backend:

| Función del frontend | Cómo |
|---|---|
| Escuchar el audio analizado | Con el `File` local que eligió el usuario (`URL.createObjectURL`). El backend no lo guarda. |
| Comparar con la canción original | `<audio src="/api/v1/catalog/{song_id}/audio">` y posicionar en `source_start`…`source_end` de cada run. |
| Historial | `GET /analyses` y `GET /analyses/{analysis_id}`. |
| Exportar | `GET /analyses/{analysis_id}/export?format=json` o `format=csv`. |
| Consultar el catálogo | `GET /catalog` y `GET /catalog/{song_id}`. |

### Catálogo (solo lectura)

`GET /api/v1/catalog` devuelve las canciones que hay realmente en `songs`:

```json
{
  "song_count": 80,
  "segment_count": 480,
  "catalog_mutable": false,
  "songs": [
    {"song_id": 2, "title": "...", "artist": "...", "duration_seconds": 29.98,
     "segment_count": 6, "audio_url": "/api/v1/catalog/2/audio"}
  ]
}
```

`GET /api/v1/catalog/{song_id}` añade `segments` (`segment_id`, `segment_index`,
`start_seconds`, `end_seconds`). No se exponen `audio_path`, `audio_ref` ni rutas
internas.

**El catálogo es de solo lectura durante el runtime** (`catalog_mutable: false`). El HNSW
se construye antes de arrancar el servidor (ingesta → codebook → índice) y, con el
índice creado, el motor bloquea `INSERT`/`DELETE` en `audio_segments`. Para cambiar el
catálogo se prepara de nuevo con `fma_catalog prepare` (o la demo) en otro directorio. No
hay endpoints para añadir o quitar canciones.

### Audio de referencia

`GET /api/v1/catalog/{song_id}/audio` sirve el archivo original de una canción
catalogada con `FileResponse` de Starlette:

- admite `Range` (`206 Partial Content`, `416` si el rango no es satisfacible) y
  anuncia `Accept-Ranges: bytes`, así que el `<audio>` del navegador puede saltar a
  `source_start` sin descargar el archivo entero;
- `Content-Type` según la extensión (`audio/mpeg`, `audio/wav`, `audio/flac`,
  `audio/ogg`, `audio/basic`) y `Content-Disposition: inline` con el nombre del archivo;
- no se generan recortes: el frontend posiciona el reproductor con `source_start` y
  `source_end` del resultado.

Medidas contra path traversal:

- el endpoint recibe un **entero** (`song_id`, validado entre 1 y 2³¹−1; si no, `422`);
  nunca acepta una ruta del cliente y cualquier parámetro extra se ignora;
- la ruta sale de `songs.audio_path` y se valida con `resolve_dataset_path`, el mismo
  validador de la ingesta: relativa, canónica, sin `..`, sin salir de
  `COPYRIGHT_DATASET_ROOT` al resolver enlaces, archivo existente;
- solo se sirven extensiones de audio conocidas;
- si algo falla (fila manipulada, archivo ausente) responde `404` sin revelar la ruta.

Con CORS configurado se permite la cabecera `Range` y se exponen `Accept-Ranges`,
`Content-Range`, `Content-Length` y `Content-Disposition`.

### Historial de análisis

Cada `POST /api/v1/analyze` correcto se guarda en SoupDB **antes** de responder, en dos
tablas de la misma base, independientes de `songs`/`audio_segments`:

```
analyses (analysis_id TEXT, filename TEXT, created_at TEXT,
          segment_seconds FLOAT, match_count INT)
analysis_matches (analysis_id TEXT, match_order INT, song_id INT,
                  query_start FLOAT, query_end FLOAT, source_start FLOAT, source_end FLOAT,
                  score FLOAT, segment_count INT, min_score FLOAT, max_score FLOAT,
                  alignment_offset_seconds FLOAT, is_best_run BOOL)
```

- El servidor crea las tablas (y un índice B+ sobre `analysis_id` en cada una) al
  arrancar si faltan; si existen con otro schema, no arranca.
- `filename` es solo el nombre mostrado (sin ruta); `created_at` es ISO 8601 en UTC.
- `title` y `artist` no se duplican: se leen de `songs` al reconstruir.
- **No se guarda el audio subido.** El upload sigue la política de siempre: staging
  temporal → análisis → borrado al terminar. Un análisis histórico conserva el
  resultado y la metadata, pero el audio ya no existe en el servidor: para volver a
  escucharlo, el usuario necesita su archivo.
- Atomicidad: SoupDB no tiene `UNIQUE` ni transacciones durables entre tablas. Se
  insertan primero las runs y al final la fila de `analyses`, que actúa como marca de
  commit; las lecturas solo ven análisis con esa fila y comprueban `match_count`. Si la
  escritura falla, se borran las runs escritas y el `POST` responde `500`.
- Durabilidad: el `Catalog` escribe las páginas a disco al cerrarse (y al desalojarlas
  del buffer). Con un apagado normal del servidor el historial persiste; matar el
  proceso de golpe puede perder los análisis más recientes (el motor aún no tiene WAL).

`GET /api/v1/analyses` lista los análisis del más reciente al más antiguo:

```json
[{"analysis_id": "7f3c…", "filename": "remix.mp3", "created_at": "2026-11-15T10:42:07.123456Z",
  "segment_seconds": 5.0, "match_count": 9, "song_count": 4, "best_score": 0.94}]
```

`best_score` es el mayor `score` medio de una run (similitud, no probabilidad) y vale
`null` si el análisis no tiene runs. `GET /api/v1/analyses/{analysis_id}` devuelve el
`AnalysisResult` completo más `created_at`. Un id inexistente o mal formado da `404`.

### Exportación

`GET /api/v1/analyses/{analysis_id}/export?format=json|csv` (por defecto `json`; otro
formato da `422`) responde con `Content-Disposition: attachment;
filename="analysis-<id>.<ext>"`.

- **JSON**: el mismo cuerpo que `GET /analyses/{analysis_id}`.
- **CSV** (UTF-8): una fila por run, en el orden del resultado; sin runs, solo la
  cabecera. Columnas: `analysis_id, created_at, filename, match_order, song_id, title,
  artist, query_start, query_end, source_start, source_end, duration, score,
  segment_count, min_score, max_score, alignment_offset_seconds, is_best_run`.
  `duration` es `query_end - query_start`. Los textos que empiezan por `= + - @` se
  prefijan con `'` para que una hoja de cálculo no los ejecute como fórmula.

No hay exportación a PDF.

### Valores reales del sistema

Para cualquier texto o pantalla del frontend:

| Concepto | Valor real |
|---|---|
| Catálogo académico | 80 canciones de FMA Small (10 por género) y 480 segmentos indexados |
| Segmentos | 5 s **sin solapamiento**; resolución temporal de 5 s |
| Descriptor de entrada | MFCC de **20 coeficientes** por frame (22 050 Hz, mono) |
| Codebook | K-Means con **K=128** |
| Vector final | histograma TF-IDF de **128 dimensiones**, normalizado L2 |
| Índice | **HNSW** |
| Métrica | **euclidean** |
| Score | `1 / (1 + distancia)`: similitud en `(0, 1]`, no probabilidad |
| Umbral de copyright | **no existe** un umbral validado |
| Dictamen | **no existe** dictamen jurídico automático |

**No implementado** (no debe presentarse como funcionalidad): vectores MFCC de 1024
dimensiones, métrica cosine, solapamiento de segmentos del 10/50/75 %, umbrales del
tipo "≥ 80 %" u "≥ 85 %", cumplimiento ISO 27037, cadena de custodia, dictamen pericial,
reparación por FFT, auto-reparación de audio, procesamiento forense y actualización
inmediata del HNSW al subir canciones.

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

## Pipeline multimedia: ingesta y segmentación

`applications/copyright_detection/ingestion.py` registra una canción original del
dataset en el catálogo. Requiere `soupdb-vector` (librosa), incluido en los grupos
`copyright-api` y `dev`; `app.py` no importa este módulo.

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

- **Ubicación del candidato:** `audio_path` es relativo a la raíz
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

## Decisiones técnicas

| Decisión | Valor |
|---|---|
| Segmentación | 5 s, sin solapamiento; el último bloque se conserva solo si dura ≥ 2.5 s |
| Sample rate | 22 050 Hz, mono |
| MFCC | 20 coeficientes |
| Codebook | K=128 (K-Means) |
| Representación | TF-IDF con normalización L2 |
| Índice | HNSW |
| Métrica | euclidean |
| Búsqueda | k vecinos por segmento, configurable con `COPYRIGHT_SEARCH_K` (por defecto 5) |
| Score | `1 / (1 + d)`: similitud, no probabilidad |

HNSW se eligió por el benchmark de audio de la Parte 4 sobre 100 000 segmentos:
Recall@10 ≈ 0.990 y ≈ 12.3 ms por consulta. Son cifras de ese benchmark, no de la demo,
y no significan que HNSW supere siempre a IVF: fue la mejor opción en esas condiciones.

## Limitaciones

- No hay un umbral calibrado: se devuelven todas las runs.
- `score` es una similitud, no una probabilidad, y la respuesta no es un veredicto
  jurídico.
- Como los segmentos no se solapan, la resolución temporal es de 5 s.
- Las búsquedas de un mismo proceso se ejecutan de una en una, protegidas por un `RLock`,
  porque el `Catalog` no está documentado como thread-safe.
- La caché de MFCC conserva las entradas de los uploads ya borrados; no se limpian.
- Con la API actual no se pueden añadir canciones una vez creado el HNSW: haría falta
  integrar la ingesta con transacciones o reconstruir el índice. El catálogo es de solo
  lectura en runtime (`catalog_mutable: false`).
- El historial no guarda el audio subido: un análisis antiguo no se puede volver a
  escuchar desde el servidor.
- El historial es durable tras un apagado normal; si el proceso muere de golpe se
  pueden perder los análisis más recientes (sin WAL en el motor).
- El historial no tiene paginación ni borrado.
- No hay límite de tamaño para los uploads.
- El frontend se desarrolla por separado y solo consume la API REST.
