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
- **SemVer por hitos del curso:** releases `v0.1.0`..`v1.0.0` alineadas a los Milestones (ver `estrategia-release.md`).
- **Publicación del contenedor en GHCR** junto a cada release con tag `v*`.