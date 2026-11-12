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
