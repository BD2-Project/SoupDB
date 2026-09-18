# Decisiones de diseño

El proyecto tiene presentación final con preguntas. Cada decisión debe poder defenderse. Se registran aquí a medida que se toman.

- **Página de 4096 bytes:** coincide con el tamaño de bloque típico del sistema de archivos.
- **RID = (page_id, slot)** en vez de offset absoluto: permite compactar dentro de la página.
- **Clave compuesta** para índices sobre campos con duplicados.
- **Borrado lazy y RIDs estables en el archivo secuencial:** los borrados generan tombstones y las inserciones no mueven registros vivos; no se realiza reorganización global automática.
- **Modelo Volcano en el ejecutor:** permite componer operadores sin materializar resultados.
- **Layout de registros:** codec con longitud variable en `common/`. `INT` = 4 bytes, `FLOAT` = 8 bytes, `BOOL` = 1 byte, `VARCHAR(n)`/`TEXT` = prefijo `uint32` + bytes UTF-8. Little-endian, sin alineación. `ColumnType` y `ColumnDef` viven en `common/schema.py` (capa base) y `engine/query/ast.py` los re-exporta.
- **Overflow de registros grandes:** en *stand by*. Hasta que el dominio de storage lo defina, un registro que exceda el tamaño de página no tiene soporte.
- **Operadores Volcano concretos en `engine/query/operators.py`** (`TableScan`, `Filter`, `Project`, `Sort`, `Aggregate`, `Distinct`): en memoria, operando sobre `Record` con el codec de `common/record.py`. El `PlanNode` de `explain()` reporta filas emitidas, `elapsed_ms` (reloj de pared) y el delta de lecturas/escrituras de un `DiskManager` inyectado.
- **Propagación del `DiskManager` desde el catálogo:** el planner obtiene `catalog.disk_manager` por *duck typing* y lo inyecta en todos los operadores del plan. Si el contrato del catálogo no lo expone, los planes creados desde SQL reportan cero lecturas/escrituras en `explain()` (limitación conocida, registrada en los comentarios del planner).
- **Planner y executor consumen el catálogo por *duck typing*** con métodos esperados `schema(name)`, `file_org(name)`, `indexes(name)` y `create_table(name, columns)`. El contrato formal de `engine/common/catalog.py` se pacta con el equipo antes de la integración real; los tests usan un catálogo fake en memoria.
- **ResultSet interno provisional** en `engine/query/resultset.py` (columnas + filas tipadas + contador de filas afectadas). El formato de wire hacia `rsoup`/frontend se define en la integración con el dominio de Transacciones.
- **Selección de índice en el planner:** `catalog.indexes(name)` devuelve un mapeo `columna → Index`. Si un `WHERE` permite acceso por igualdad o rango inclusivo sobre una columna indexada, se usa `IndexLookup`/`IndexRangeScan` como hoja y se mantiene el `Filter` encima para conservar la semántica del predicado.
- **Modo *spill* en `Sort` y `Aggregate`:** opcional, activado por `memory_limit_bytes`. Usa `external_sort` (k-way merge) y `external_hash_group_by` (Particionamiento hash) del dominio de Indexación. Claves compatibles: `int`, `str` y compuestas (`tuple`). Por defecto permanecen en memoria.
- **SemVer por hitos del curso:** releases `v0.1.0`..`v1.0.0` alineadas a los Milestones (ver `estrategia-release.md`).
- **Publicación del contenedor en GHCR** junto a cada release con tag `v*`.