# Consultas SQL textuales y multimedia

> **Alcance actual:** integración del lenguaje SQL, planificación, operadores,
> catálogo, pruebas y protocolo usando proveedores **simulados**. Las consultas
> siguientes **no son ejecutables con un catálogo nuevo sin una factory de
> índices rankeados y los proveedores compatibles**. No se considera validada
> todavía la calidad de los algoritmos de indexación reales.

## Preparación y requisitos

```bash
uv sync --frozen --group multimedia
uv run --frozen --group multimedia pytest -q \
  tests/integration/test_ranked_sql.py \
  tests/unit/transactions/test_ranked_protocol.py
```

Para construir índices con `CREATE INDEX ... USING`, el catálogo debe recibir
una `RankedFactory` compatible. Para multimedia, también debe disponer de un
`query_vectorizer` con `vectorize(reference, *, profile_id)`, que genere un
vector de la **misma dimensión y perfil** con que se construyó el índice.
No se debe inventar ni regenerar el perfil en cada consulta.

La factory de ejemplo `tests/fakes/fake_ranked_factory.py` y los fakes de
búsqueda son únicamente herramientas de prueba: **no calculan TF-IDF, BM25 ni
vecinos aproximados reales**.

## Búsqueda de texto

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

SELECT *, SCORE() AS relevancia FROM articulos
WHERE MATCH(texto, 'machine learning')
USING BM25 ORDER BY relevancia DESC;
```

- `MATCH(columna, 'consulta')` necesita un índice `INVERTED` sobre la columna.
- `USING TF_IDF` y `USING BM25` seleccionan el método que se pasa al índice;
  SQL no recalcula el ranking.
- `SCORE()` devuelve un `FLOAT` recibido del proveedor, de mayor a menor
  relevancia. `SELECT *` solo muestra columnas reales de la tabla.
- Si falta `LIMIT`, o hay filtros residuales, `DISTINCT` o reordenación, el
  proveedor debe declarar `exact`, `exhaustive` y `item_count` conocidos. Si
  no garantiza enumeración completa, el plan **rechaza** la consulta; nunca
  usa un top-k oculto que pueda dar una respuesta incompleta.

## Búsqueda multimedia

```sql
SELECT * FROM imagenes
WHERE SIMILAR_TO('foto_consulta.jpg', k=10)
USING HNSW WITH METRIC=cosine;

SELECT nombre, artista, SIMILARITY_SCORE() AS score
FROM canciones
WHERE SIMILAR_TO('audio_query.mp3', k=5)
USING IVF WITH METRIC=euclidean
ORDER BY score DESC;
```

- El argumento de `SIMILAR_TO` es una **referencia**, no un embedding que se
  entregue directamente al índice. Se vectoriza al **ejecutar** la consulta.
- `SIMILARITY_SCORE()` conserva el `FLOAT` del proveedor; puede ser negativo
  según la métrica. El valor más alto se considera mejor.
- `k` define el conjunto inicial de vecinos; `LIMIT` y `OFFSET`, si aparecen,
  operan después. No se permite prometer los k mejores **filtrados** si el
  índice ANN no puede garantizarlo.
- HNSW/IVF son métodos aproximados cuando así lo declara el proveedor; no se
  debe atribuir exactitud a sus resultados sin medición y validación externas.
- Varios índices compatibles implican error de ambigüedad, no selección
  arbitraria por orden de registro.

## DDL especializado

```sql
CREATE INDEX idx_contenido ON documentos (contenido) USING INVERTED;

CREATE INDEX idx_imagenes ON imagenes (archivo) USING HNSW
WITH METRIC=cosine, DIMENSION=128, PROFILE='imagen-v1';

CREATE INDEX idx_audio ON canciones (nombre) USING IVF
WITH METRIC=euclidean, DIMENSION=64, PROFILE='audio-v1';

DROP INDEX idx_contenido;
```

Los valores `DIMENSION` y `PROFILE` son **ilustrativos**, no perfiles
certificados del equipo. La creación, reapertura y eliminación dependen de la
factory inyectada. La metadata rankeada se mantiene separada de los índices
BTREE/HASH/RTREE y de las tablas de sistema anteriores.

## Planes y protocolo TCP

```sql
EXPLAIN (FORMAT JSON)
SELECT *, SCORE() AS relevancia FROM articulos
WHERE MATCH(texto, 'machine learning') USING BM25 LIMIT 10;

EXPLAIN ANALYZE (FORMAT JSON)
SELECT * FROM imagenes
WHERE SIMILAR_TO('foto_consulta.jpg', k=5)
USING HNSW WITH METRIC=cosine;
```

`EXPLAIN` sin `ANALYZE` no debe llamar al índice ni vectorizar referencias.
`EXPLAIN ANALYZE` sí ejecuta la búsqueda y muestra contadores de filas y
candidatos. El protocolo binario transporta scores como `FLOAT`, sin
convertirlos a texto ni descartar su signo.

Las pruebas de integración usan `ConnectionHandler` sobre un puerto efímero,
`execute_sql`, `FakeCatalog`, `ScriptedRankedIndex` y `FakeQueryVectorizer`.
Probar estas rutas **no implica** haber probado los proveedores reales ni el
frontend `SoupChef`.

## Escrituras y transacciones

Los índices rankeados admiten `INSERT`/`DELETE` solo si todos los proveedores
implicados declaran capacidad mutable **y reversible**, con mutaciones
idempotentes y atomicidad declarada. De lo contrario, el motor debe rechazar
la escritura **antes** de modificar la tabla. El rollback restaura entradas
por RID; tras un `DELETE` revertido puede asignarse un RID nuevo.

Las pruebas cubren compensación dentro del proceso, no recuperación atómica
ante apagado, corte de energía o fallo de disco. El DDL rankeado sigue
bloqueado dentro de transacciones explícitas.

## Pendientes para afirmar integración real

1. **C15 / Breysi:** adaptador de extracción con perfiles de imagen/audio
   definidos y reproducibles; no calcular un codebook nuevo al consultar.
2. **C22 / Daniel:** adaptadores SQL para las implementaciones IVF/HNSW reales,
   con sus métricas y capacidades declaradas y verificadas.
3. **C23 / Yeimi:** adaptador SQL del índice invertido real, con contrato de
   coincidencia, exhaustividad, persistencia y mantenimiento.
4. Validar los cuatro ejemplos con datos reales, resultados esperados y
   comprobaciones independientes de scores, recall y persistencia.

**No describir el proyecto como integración real completa** hasta que se
superen estos puntos. Para detalles del contrato, consultar
[Integración SQL rankeada](integracion_sql_rankeada.md).

## Integración real opcional SIFT/MFCC e IVF/HNSW (C15/C22)

La integración usa los **módulos reales existentes** `engine.multimedia`
(SIFT/MFCC, codebook K-Means, BoW TF-IDF) y los índices físicos
`engine.indexes.vector.ivf.IVFIndex` y `HNSWIndex`. No entrena un codebook
implícitamente: se debe proporcionar un archivo `codebook.npz` generado con
`python -m engine.multimedia.build_vectors` usando el mismo corpus.

```python
from pathlib import Path
from engine.common.catalog import Catalog
from engine.indexes.vector.sql_factory import OnDiskVectorFactory
from engine.multimedia.sql_profiles import MultimediaProfile
from engine.query import execute_sql

root = Path("/ruta/a/dataset/imagenes")
profiles = {
    "imagenes-v1": MultimediaProfile(
        "imagenes-v1", "image", root, Path("/ruta/a/codebook_imagen.npz"),
        cache=Path("/ruta/a/cache_sift"),
    ),
}
factory = OnDiskVectorFactory(Path("/ruta/a/soupdb/indices_ranked"), profiles)
catalog = Catalog(Path("/ruta/a/soupdb/datos"), ranked_factory=factory)

# Las filas deben contener rutas RELATIVAS a root, p.ej. "gatos/001.jpg".
# CREATE TABLE, INSERT iniciales y CREATE INDEX se ejecutan con execute_sql.
execute_sql(
    "CREATE INDEX idx_imagenes ON imagenes (archivo) USING HNSW "
    "WITH METRIC=cosine, DIMENSION=128, PROFILE='imagenes-v1'",
    catalog,
)
result = execute_sql(
    "SELECT archivo, SIMILARITY_SCORE() AS score FROM imagenes "
    "WHERE SIMILAR_TO('gatos/001.jpg', k=5) USING HNSW "
    "WITH METRIC=cosine ORDER BY score DESC",
    catalog,
)
catalog.close()
```

`DIMENSION` corresponde a la **cantidad de palabras del codebook** (K), no
a las dimensiones de SIFT (128) o MFCC (20). Para audio use
`MultimediaProfile(id, "audio_segment", ...)` con referencias
`ruta/audio.wav#segment=000000` (sin sufijo se utiliza el primer segmento de
5 segundos). Imagen y audio deben usar perfiles y codebooks separados.

Restricciones actuales:

- Los índices físicos IVF/HNSW **son aproximados**: `exact=False`,
  `exhaustive=False`; no se garantiza recall perfecto.
- INSERT/DELETE posteriores en tablas indexadas quedan **bloqueados**.
  Requieren un proveedor con mutación atómica y reversible.
- Los archivos multimedia deben permanecer bajo la raíz del perfil; el
  codebook debe mantenerse idéntico entre construcción y reapertura.
- Al reabrir la BD se debe proporcionar de nuevo el factory con el mismo
  `storage_root`, perfiles y parámetros de almacenamiento.
- El espacio de vectores y el codebook no pueden cambiar sin reconstruir el índice.
- El adapter carga y valida los codebooks existentes; **no** equivale a un
  benchmark de recall sobre datasets reales ni integra el índice textual INVERTED.
