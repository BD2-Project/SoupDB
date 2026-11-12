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
