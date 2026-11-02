# Integración SQL de búsquedas rankeadas — Partes 3 y 4

**Estado:** propuesta de integración; los puntos marcados «pendiente» requieren acuerdo del equipo.
**Base auditada:** `origin/dev` en `86f2530c823f4659559e6786b0067fdda0054494` (01/11/2026).
**Responsable de integración SQL:** Willian.

Este documento **complementa**, sin reemplazarlos ni modificarlos, los contratos
congelados de `docs/contratos.md` y `engine/indexes/base.py`.

## 1. Contratos existentes (verificados en la base auditada)

- `Index` y `RankedIndex` son ABC independientes. No insertar instancias
  `RankedIndex` en `Catalog.indexes()` / `indexes_for()` ni exigirles
  `supports_range`, `insert` o `remove`.
- `RankedIndex.top_k(query, k, **opts) -> list[tuple[RID, float]]`;
  `RankedIndex.close() -> None`. Los hits apuntan a RIDs de una tabla real.
- Los scores siempre se ordenan de mayor a menor, sin recalcularlos en SQL.
- Para texto: consulta `str` y `method="tf_idf" | "bm25"`.
- Para vectores: secuencia numérica de la dimensión del índice;
  `metric="euclidean" | "cosine" | "inner_product"`.
  IVF utiliza `nprobe`; HNSW utiliza `ef`.
- Convención actual de score: Euclidiana `1/(1+d)`; coseno = similitud
  coseno; `inner_product` = producto punto.
- Los operadores Volcano mantienen `open`, `next`, `close` y `explain`.
  `PlanNode` y `ResultSet` ya admiten la representación de plan y FLOAT.
- **Limitaciones existentes:** `top_k` no garantiza por contrato
  enumeración exhaustiva, desempate global, filtro por RID, paginación,
  ni mantenimiento transaccional reversible.

## 2. Superficie SQL requerida por el enunciado

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

SELECT *, SCORE() AS relevancia FROM articulos
WHERE MATCH(texto, 'machine learning')
USING BM25 ORDER BY relevancia DESC;

SELECT * FROM imagenes
WHERE SIMILAR_TO('foto_consulta.jpg', k=10)
USING HNSW WITH METRIC=cosine;

SELECT nombre, artista, SIMILARITY_SCORE() AS score
FROM canciones
WHERE SIMILAR_TO('audio_query.mp3', k=5)
USING IVF WITH METRIC=euclidean
ORDER BY score DESC;
```

- **Propuesta:** representar `MATCH` y `SIMILAR_TO` mediante nodos AST
  especializados, no evaluarlos por cada fila como funciones escalares.
- **Propuesta:** `SCORE()` y `SIMILARITY_SCORE()` leen el score del hit
  original; el operador agrega un campo FLOAT **temporal y no visible**,
  enlazado por posición. `SELECT *` expande solo columnas base.
- **Propuesta:** mantener `LIMIT`/`OFFSET`, `ORDER BY`, `ResultSet`,
  `Record`, `RID` y el formato exterior de `PlanNode` sin modificaciones.
- **Propuesta:** si falta una capacidad necesaria, rechazar la consulta
  de forma explícita en lugar de devolver resultados incompletos.

## 3. Fronteras propuestas (NO son APIs implementadas)

- `RankedBinding`: nombre de índice, tabla, columna, familia/algoritmo,
  métrica, dimensión, perfil de extracción, instancia `RankedIndex`,
  cardinalidad conocida y capacidades (`exact`, `exhaustive`, `mutable`).
- `Catalog.ranked_indexes_for(table, column=None)`: registro separado
  de los índices escalares, pendiente de decidir implementación/persistencia.
- `QueryVectorizer.vectorize(reference, *, profile_id)`: convierte una
  referencia multimedia en vector compatible con su índice; no se invoca
  al parsear ni en `EXPLAIN` sin `ANALYZE`.
- `RankedMaintenance`: preparación y operaciones reversibles de alta/baja
  por RID. No equivale al ABC `RankedIndex`.
- Factory/lifecycle de índices: crear, abrir, cerrar, eliminar y recuperar
  recursos y metadata por índice, sin reconstrucción accidental.
- Operadores requeridos en el plan: `InvertedIndexScan`, `IVFSearch`,
  `HNSWSearch` (con esos nombres en `PlanNode.op`).

Todas las firmas exactas, tipos, ubicación y persistencia de estas fronteras
deberán fijarse antes de codificar su versión de producción.

## 4. Reglas provisionales de seguridad y correctitud

Estas reglas son **recomendaciones de integración, no acuerdos del equipo**:

1. `MATCH` devuelve coincidencias definidas por el analizador textual;
   nunca inferir coincidencia a partir de `score > 0` sin contrato.
2. Una búsqueda textual **sin `LIMIT`** necesita enumeración exhaustiva
   de coincidencias. No introducir un `k` oculto; si no está garantizada,
   devolver un error de capacidad.
3. En una búsqueda textual con `LIMIT`, pedir solo `LIMIT + OFFSET`
   es correcto únicamente si no hay filtros residuales, deduplicación,
   cambios de orden ni empates que exijan más candidatos.
4. `SIMILAR_TO(..., k=N)` define el conjunto inicial de vecinos;
   `LIMIT/OFFSET` opera después sobre ese conjunto, salvo acuerdo distinto.
5. Con filtros adicionales en ANN, no usar sobremuestreo fijo como
   garantía de top-k filtrado. Rechazar hasta definir y soportar semántica.
6. Si hay más de un índice multimedia compatible, informar ambigüedad;
   no elegir por orden de registro.
7. Validar `k`, dimensión, métricas, perfil, tipo y finitud de vectores
   antes de llamar al índice. Conservar scores negativos válidos.
8. Usar **RID real** del archivo de tabla: IDs de `vectores.bin`,
   corpus y benchmarks no son RIDs automáticamente.
9. Sin `remove` y undo reversible, bloquear DML afectado **antes** de
   tocar almacenamiento. No asumir que `insert` implica mutabilidad segura.
10. Convertir errores de índice/vectorizador a errores propios de SoupDB
    con contexto; no exponer excepciones inesperadas por el protocolo TCP.

## 5. Decisiones pendientes por responsable

| Responsable | Pendiente de confirmar | Bloquea |
|---|---|---|
| Yeimi | Coincidencias/no coincidencias; texto vacío; estrategia `tf_idf`/`bm25`; enumeración exhaustiva sin LIMIT; RIDs y mantenimiento | Integración textual real y algunas optimizaciones |
| Daniel | Desempate; compatibilidad de métricas; construcción/reapertura; `remove` y mantenimiento reversible; límites de IVF/HNSW | Garantías ANN, DML y rollback reales |
| Breysi | Perfil estable (medio, dimensión, codebook/IDF, normalización); audio completo vs segmento; referencias de archivo accesibles al servidor | Vectorización multimedia SQL real |
| Suso | Identificadores y tamaño de documentos del corpus; contrato con UI/TCP; exposición de planes/score | Ingesta y demo integrada |
| Willian + equipo | Selección de índice ambiguo; `k` omitido y `k=0`; filtros + top-k; ties; DDL `TYPE`/`USING`; metadata persistente; aislamiento y lifecycle | Congelar semántica SQL final |

La tabla **no implica aprobación** de ninguna decisión. Los acuerdos nuevos se
registrarán aquí con fecha, responsables y evidencia de revisión.

## 6. Casos mínimos que debe cubrir la implementación

- Cuatro consultas oficiales completas, primero con fakes y después
  con índices reales cuando sus contratos estén disponibles.
- Texto sin `LIMIT` y sin matches; RIDs duplicados y obsoletos;
  puntuación cero o negativa; empate en el corte de top-k.
- `SELECT *`, `SELECT *, SCORE()`, alias en `ORDER BY`, `LIMIT/OFFSET`.
- Imagen y audio con perfil correcto; k cero/negativo/no entero;
  métrica o dimensión incompatibles; archivo faltante.
- Filtros relacionales + ANN sin garantía; dos índices compatibles.
- Índice inexistente; cierre/reapertura; `EXPLAIN (FORMAT JSON)`;
  errores propagados por TCP y control de acceso concurrente.
- Regresión obligatoria: consultas relacionales, BTREE/HASH, R-Tree,
  transacciones y pruebas existentes.

**Condición de aceptación del C01:** este documento separa hechos actuales,
propuestas y pendientes; no altera los ABC congelados ni habilita cambios
irreversibles. La implementación seguirá el principio de fallo explícito cuando
una capacidad necesaria aún no esté disponible.

## SQL textual ejecutable con índices rankeados (C07–C10)

Estas capacidades están verificadas **contra el catálogo de prueba y el
`ScriptedRankedIndex`**, no constituyen todavía integración con el índice
invertido persistente de Yeimi. Se conserva el `RankedIndex.top_k` original.

```sql
SELECT * FROM documentos
WHERE MATCH(contenido, 'base de datos vectorial')
USING TF_IDF LIMIT 10;

SELECT *, SCORE() AS relevancia FROM articulos
WHERE MATCH(texto, 'machine learning') USING BM25
ORDER BY relevancia DESC;
```

Reglas de la implementación actual:

- Un único `MATCH` por `SELECT`, como predicado principal o como parte de una
  conjunción `AND`. `OR MATCH`, `NOT MATCH`, JOIN y GROUP BY se rechazan.
- El registro debe contener exactamente un binding de familia `INVERTED`
  compatible con la columna de texto; no se elige uno arbitrariamente.
- Sin filtro residual, DISTINCT ni reordenamiento, `LIMIT + OFFSET` determina
  los candidatos. Sin `LIMIT`, o cuando hay filtros, DISTINCT o un orden
  distinto de `SCORE() DESC`, se solicita `item_count` y se requieren las tres
  garantías explícitas: `exact=True`, `exhaustive=True`, `item_count` conocido.
  No se utiliza sobremuestreo fijo ni un `k` por defecto.
- `SCORE()` lee el FLOAT recibido del proveedor, sin recalcularlo, y es visible
  solo cuando se proyecta explícitamente.
- `EXPLAIN (FORMAT JSON)` no consulta el índice; `EXPLAIN ANALYZE` sí. La
  hoja del plan se llama `InvertedIndexScan` e incluye `requested_k`,
  `candidate_policy`, `provider_hits`, `rows_fetched`, `exact`, `exhaustive`
  y las opciones de búsqueda.
- El SQL todavía no crea ni persiste estos bindings; faltan C16–C20.
  La semántica de exhaustividad del índice textual real debe confirmarla Yeimi.

## SQL multimedia con índices rankeados (C11–C14)

Se admiten consultas `SIMILAR_TO('referencia', k=N) USING IVF|HNSW WITH METRIC=cosine|euclidean|inner_product` y la proyección `SIMILARITY_SCORE()`. El parámetro `k` es obligatorio y se interpreta como el tamaño del conjunto de vecinos del índice; `LIMIT/OFFSET` se aplica **después**, y no amplía `k`. `k=0` devuelve cero filas sin vectorización ni búsqueda. El score es el valor original del proveedor (`mayor = mejor`), sin nuevas conversiones.

La consulta selecciona un índice vectorial cuando hay **exactamente un** binding de la tabla cuyo algoritmo y métrica coincidan. Si no existe ninguno o hay varios candidatos compatibles, la consulta falla explícitamente. Cada índice debe anunciar `dimension`, `profile_id` y `metric` y estar asociado a una columna de referencia TEXT/VARCHAR.

El servidor recibe un `query_vectorizer` inyectado en el catálogo con método `vectorize(reference, *, profile_id) -> Sequence[float]`. Se invoca **durante la ejecución**, nunca al parsear ni en `EXPLAIN` sin ANALYZE. El vector debe coincidir en dimensión, tener números finitos y no contener booleanos. El perfil debe ser el mismo que se usó para construir el índice, no uno elegido por extensión de archivo. El fake sirve para integración SQL; **la extracción real imagen/audio depende de C15 y del perfil de Breysi**.

Por seguridad, se rechazan de momento los filtros residuales (incluidos AND, OR y NOT con `SIMILAR_TO`), `DISTINCT`, agregados, GROUP BY, HAVING y JOIN. El ANN actual no ofrece una búsqueda filtrada con garantías; aplicar un filtro relacional después del top-k sin declararlo podría devolver menos vecinos y alterar la semántica. `ORDER BY` opera sobre los `k` resultados recuperados; si ordena por un campo distinto al score, reordena ese conjunto sin ampliar el universo. Los empates en el límite y el recall ANN dependen de las garantías del proveedor.

Ejemplos de consulta:

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

Estas consultas funcionan con `RankedIndex` y `QueryVectorizer` inyectados (fakes); **no implica disponibilidad del registro de perfiles ni del índice multimedia real**.


## Estado de implementación y pruebas (C21)

Los contratos iniciales de este documento eran una propuesta. En la rama
`feat/sql-texto-multimedia` se han implementado parser, planificador,
proyección/orden de scores, registro de metadata, factory inyectada,
operadores y guardas transaccionales. `tests/integration/test_ranked_sql.py`
prueba las cuatro consultas por API SQL y TCP **con proveedores simulados**.
Las integraciones reales de extracción, IVF/HNSW e índice invertido
continúan pendientes de verificación. Véase
[SQL textual y multimedia](sql_texto_multimedia.md).
