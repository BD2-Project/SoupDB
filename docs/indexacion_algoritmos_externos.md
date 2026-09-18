# Indexación y algoritmos externos

## Estructuras implementadas

### B+ Tree no agrupado

Se implementó un árbol B+ persistente sobre páginas de disco.

Características principales:

- búsqueda por igualdad;
- búsqueda por rango inclusivo;
- claves duplicadas;
- inserción con split de hojas;
- split de nodos internos;
- crecimiento del árbol a múltiples niveles;
- eliminación de entradas;
- persistencia mediante `DiskManager` y `BufferManager`;
- RIDs como referencias a los registros.

Archivos principales:

- `engine/indexes/bplus_tree.py`
- `engine/indexes/_bplus_page.py`
- `engine/indexes/_key_codec.py`

### B+ Tree agrupado

El B+ agrupado combina:

- un B+ Tree para localizar registros mediante RIDs;
- un `SequentialFile` como organización física ordenada por clave.

La inserción se realiza primero en el archivo secuencial y posteriormente se
registra en el B+ el RID obtenido.

Esto permite diferenciar entre:

- `BPlusTree`: índice no agrupado;
- `ClusteredBPlusTree`: índice asociado a almacenamiento ordenado físicamente.

Archivo principal:

- `engine/indexes/clustered_bplus.py`

### Extendible Hash

El proyecto dispone de un índice Hash Extensible persistente integrado en
`dev`.

Soporta:

- búsqueda por igualdad;
- inserción;
- eliminación;
- duplicación dinámica del directorio;
- split de buckets;
- páginas de overflow al alcanzar la profundidad máxima.

No soporta consultas por rango.

Archivos principales:

- `engine/indexes/extendible_hash.py`
- `engine/indexes/_hash_page.py`

## Algoritmos externos

### External Sort

`engine/algorithms/external_sort.py` implementa ordenamiento externo mediante:

1. generación de runs ordenados limitados por memoria;
2. almacenamiento temporal de los runs;
3. combinación k-way;
4. múltiples pasadas cuando la cantidad de runs supera el fan-in.

El algoritmo acepta un comparador inyectable para representar diferentes
criterios de `ORDER BY`.

### External Hashing

`engine/algorithms/external_hash.py` implementa particionamiento hash externo
para:

- `GROUP BY`;
- equality `JOIN`.

Cuando una partición excede el límite de memoria puede ser reparticionada.
En el JOIN, si luego de alcanzar la profundidad máxima una partición todavía
es grande, se procesa mediante bloques limitados por memoria.

## Comparación experimental

Se incluyeron tres scripts reproducibles.

### Comparación de estructuras de índices


uv run python -m benchmarks.scripts.compare_indexes \
  --records 10000 \
  --queries 1000 \
  --mutations 500


Compara:

* B+ Tree;
* Extendible Hash.

Mide:

* construcción;
* búsqueda por igualdad;
* búsqueda por rango;
* inserción;
* eliminación;
* lecturas y escrituras de disco;
* tamaño del índice.

### Comparación con recuperación de registros


uv run python -m benchmarks.scripts.compare_physical_indexes \
  --records 10000 \
  --queries 1000


Compara:

* B+ no agrupado + HeapFile;
* B+ agrupado + SequentialFile;
* Extendible Hash + HeapFile.

Las consultas incluyen la recuperación real del registro a partir del RID.

### Comparación de algoritmos externos


uv run python -m benchmarks.scripts.compare_external_algorithms \
  --records 10000 \
  --memory 4096 16384 65536


Evalúa el efecto del límite de memoria sobre:

* `ORDER BY` mediante External Sort;
* `GROUP BY` mediante External Hashing;
* equality `JOIN` mediante External Hashing.

## Integración con consultas SQL

Los algoritmos están implementados de forma independiente de la capa SQL.

El stack de consultas (lexer, parser, planner, executor y operadores volcano)
consume estas estructuras a través del catálogo real (`engine/common/catalog.py`):
el planner selecciona entre sequential scan e index lookup/range scan y el modo
spill de Sort/Aggregate se activa cuando el plan incluye un `memory_limit_bytes`.

El catálogo es persistente: registra cada índice en `SysIndexes`, hace
*backfill* de las filas existentes al crearlos y el executor los mantiene en
`INSERT`/`DELETE` vía `indexes_for`. Los planes de `explain()` reportan el
delta de lecturas/escrituras de los `DiskManager`s que el catálogo expone por
*duck typing*.
