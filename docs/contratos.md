# Contratos congelados

Los contratos son la **Verdad Absoluta** y viven únicamente en el código fuente. No se modifican sin acuerdo de todos los desarrolladores.

| Contrato | Ruta canónica |
|---|---|
| `RID(page_id, slot)` | `engine/common/rid.py` |
| Excepciones propias | `engine/common/errors.py` |
| `Index` (ABC) | `engine/indexes/base.py` |
| `FileOrganization` (ABC) | `engine/storage/base.py` |
| `ConcurrencyStrategy` (ABC) | `engine/transactions/base.py` |
| Operadores Volcano | `engine/query/operators.py` |
| `PlanNode` (JSON del plan) | `engine/query/operators.py` |

Reglas del contrato:

- Todos los índices apuntan a RIDs. Todas las organizaciones de archivo devuelven RIDs.
- `ExtendibleHash.supports_range` es `False` y su `range_search` lanza `UnsupportedOperation`. No se simula con un scan completo.
- `op` en el plan de ejecución es string libre: en la Parte 3 aparecerá `InvertedIndexScan` y en la Parte 4 `HNSWSearch` sin tocar el renderizador.
- El esquema soporta `VARCHAR(n)` y `TEXT` desde el día uno (longitud variable).
- El control de concurrencia se consume a través de `ConcurrencyStrategy` (nunca de una implementación concreta); las estrategias son intercambiables. Ver [Transacciones](transacciones.md).
- El **R-Tree** (`engine/indexes/rtree/`) es una implementación de `Index` con `supports_range=True`: `search` por punto y `range_search` por caja espacial o rango 1D (las claves escalares se mapean al eje X para cumplir la conformance suite). Ver [R-Tree](rtree.md).
## Convenciones de consultas espaciales

Fijadas por el enunciado (§2.2.1 y §2.2.3) y verificadas por los tests de
`tests/unit/query/test_parser_spatial.py` y `test_spatial_haversine.py`.

| Elemento | Convención |
|---|---|
| Nombre de la función | `distancia(...)`, con `distance(...)` como alias |
| Literal de punto | `POINT(latitud, longitud)` |
| Vértices de polígono | `POLYGON((latitud, longitud), ...)`, el mismo orden |
| Representación interna | `Point(x=longitud, y=latitud)` |
| Métrica por defecto | euclidiana, en grados de coordenada |
| Métrica geodésica | `'haversine'`, en **metros** |

El intercambio entre el orden del literal y el interno ocurre en un único lugar,
`_parse_point_expr` del parser. Hacia adentro el motor trabaja siempre con
`Point(x, y)`.

La unidad importa: el ejemplo del enunciado es
`distancia(ubicacion, POINT(-12.0464, -77.0428)) < 5000` para un radio de 5 km,
así que haversine devuelve metros. Si devolviera kilómetros ese mismo predicado
seleccionaría un radio de 5000 km y devolvería la tabla entera **sin fallar**,
que es el peor tipo de error: silencioso.
