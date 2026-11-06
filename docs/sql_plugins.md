# SQL de plugins y búsquedas uniformes

A partir de v0.5.0 las extensiones comparten una única forma de declararse y
consultarse. El SQL del núcleo (tablas, DML, JOIN, agregados, transacciones) no
cambia.

## Activar un plugin

Antes de crear índices de un plugin hay que activarlo (equivalente a
`CREATE EXTENSION` de Postgres). El plugin debe estar **instalado** (entry point
visible); la sentencia persiste su configuración en `SysPlugins` y el motor la
reaplica al abrir la base:

```sql
CREATE EXTENSION inverted WITH (storage_root = '/app/data/plugins/inverted');
DROP EXTENSION inverted;
```

- Sin `WITH` se usa la configuración por defecto del plugin.
- Repetir `CREATE EXTENSION` es idempotente: reemplaza la configuración.
- `DROP EXTENSION` es **RESTRICT**: falla si existen índices de ese plugin.
- Es SQL normal: viaja por el protocolo `OP_QUERY` (no requiere cambios de driver).

## Declarar un índice

```sql
CREATE INDEX idx ON tabla (columna) USING <plugin> [WITH (opcion = valor, ...)];
DROP INDEX idx;
```

Ejemplos:

```sql
CREATE INDEX idx_docs ON docs (contenido) USING INVERTED WITH (method = 'bm25');
CREATE INDEX idx_img ON imagenes (archivo) USING HNSW
  WITH (metric = 'cosine', dimension = 128, profile = 'imagenes-v1');
```

Los nombres de plugin se resuelven contra los plugins instalados. Un plugin no
instalado se rechaza al planificar, nunca se ignora en silencio.

## Consultar

```sql
SELECT <columnas>, SCORE() [AS alias]
FROM <tabla>
WHERE SEARCH(<columna>, <referencia o texto>)
  USING <plugin>
  [WITH (k = 10, method = 'bm25', ...)]
[ORDER BY SCORE() [DESC]]
[LIMIT n [OFFSET m]];
```

- `SCORE()` es único para todos los plugins y devuelve el `FLOAT` del proveedor
  (mayor = mejor). `SIMILARITY_SCORE()` sigue funcionando como alias vectorial.
- `k` es el número inicial de candidatos; `LIMIT/OFFSET` se aplican después.
- Un `SEARCH` requiere exactamente un índice del plugin indicado sobre la
  columna. Varios índices compatibles producen error de ambigüedad.
- Si el plugin declara `exact=false` o `exhaustive=false`, el plan rechaza
  filtros o reordenamientos que prometan resultados completos.

## Compatibilidad

Durante una versión se aceptan los alias anteriores:

```sql
-- equivale a SEARCH(...) USING INVERTED WITH (method='tf_idf', k=10)
SELECT * FROM docs WHERE MATCH(contenido, 'texto') USING TF_IDF LIMIT 10;

-- equivale a SEARCH(...) USING HNSW WITH (metric='cosine', k=10)
SELECT * FROM imagenes WHERE SIMILAR_TO('foto.jpg', k=10) USING HNSW WITH METRIC=cosine;
```

El SQL nuevo es el recomendado para clientes y documentación.

## DML y mantenimiento

Los índices de los plugins del equipo (INVERTED, IVF, HNSW) se mantienen con
`INSERT`/`UPDATE`/`DELETE` dentro de la transacción. Los borrados de ANN usan
tombstones persistentes; las altas se insertan de forma incremental. El rollback
revierte las entradas gracias al journal de la sesión transaccional.
