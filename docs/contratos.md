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