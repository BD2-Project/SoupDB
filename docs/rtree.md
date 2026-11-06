# R-Tree (núcleo espacial)

Implementación del **R-Tree** como estructura de índice espacial para la **Parte 2** (almacenamiento de coordenadas y mapas). Este es el **núcleo en memoria**: estructura fija, inserción con división **cuadrática**, cálculo y actualización de **MBRs**, y una interfaz para pasar de memoria a almacenamiento persistente y viceversa. Vive en `engine/indexes/rtree/`.

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| `point.py` | `Point(x, y)`: la clave espacial del R-Tree |
| `mbr.py` | `MBR(min_x, min_y, max_x, max_y)`: `union`, `area`, `enlargement`, `contains`, `intersects` |
| `node.py` | `RTreeNode`: estructura fija con capacidad `order` (hoja: `(Point, RID)`; interna: `(MBR, RTreeNode)`); cada nodo mantiene su propio MBR |
| `rtree.py` | `RTree(Index)`: inserción, split cuadrático, búsqueda, rango, borrado y persistencia |
| `_codec.py` | Serialización binaria (memoria ↔ disco) |

### Estructura fija

Cada nodo tiene capacidad máxima `order` (parámetro del constructor, default `4`) y un mínimo de `max(1, order // 2)`. Un nodo se divide al superar la capacidad; la división se propaga hacia la raíz y, si la raíz se divide, crece la altura del árbol.

## Algoritmos

### Inserción (choose-subtree + Quadratic Split)

1. Desde la raíz se baja eligiendo en cada nodo interno el hijo cuyo MBR requiere el **menor enlargement** (empate → menor área).
2. Se inserta en la hoja y se **actualizan los MBRs** de todo el camino hacia arriba.
3. Si una hoja excede `order`, se aplica **Quadratic Split**:
   - Semillas: el par de entradas que maximiza el área desperdiciada (área de la unión − suma de áreas).
   - El resto de entradas se asigna al grupo que requiere menor enlargement (empate → menor área).
   - Se respeta el llenado mínimo: si un grupo está a punto de quedarse sin candidatos, el resto se asigna a él.
4. El hermano derecho se propaga al padre; si el padre se llena, se repite.

### Búsqueda

- `search(point)`: poda por MBRs que **contienen** el punto.
- `range_search(lo, hi)`: poda por MBRs que **intersectan** la caja `[lo, hi]`; en hojas devuelve los puntos contenidos.

### Borrado (simple)

Localiza las hojas cuyo MBR contiene la clave, elimina la(s) entrada(s) que matchean y **poda nodos vacíos** hacia arriba. No implementa *condense tree* ni *reinsert* (limitación del núcleo, documentada).

## Persistencia (memoria ↔ disco)

```python
tree = RTree(order=4)
tree.insert((10, 10), RID(0, 1))
tree.save("rtree.bin")          # serializa el árbol completo (codec binario)

loaded = RTree.load("rtree.bin")
# o abrir/crear:
tree2 = RTree.open("rtree.bin", order=3)
```

Formato binario del codec (big-endian): `magic "RT" + version + order u16 + has_root` y, por nodo, `is_leaf + count u32` con puntos+RIDs (hoja) o MBRs+hijos (internos). La versión paginada (un nodo por página) la construye el dominio de indexación sobre este núcleo.

## Convenciones del gestor

- Subclase de `Index` (`engine/indexes/base.py`): `insert`, `search`, `range_search`, `remove`, `supports_range`, `close`.
- `supports_range` es `True` (soporta rango espacial y 1D).
- Para pasar la **conformance suite** de `Index`, las claves escalares 1D se mapean a puntos sobre el eje X (`Point.from_key`); las claves string se mapean a un punto estable por hash.

## Uso

```python
from engine.common.rid import RID
from engine.indexes.rtree import RTree

tree = RTree(order=2)
for point in [(1, 1), (2, 5), (5, 2), (8, 8)]:
    tree.insert(point, RID(0, 1))

tree.search((2, 5))                        # [RID(0, 1)]
tree.range_search((0, 0), (6, 6))          # puntos en la caja
tree.range_search(20, 30)                  # rango 1D (claves escalares)
```

## Fuera de alcance (lo integran los demás)

- Tipo `POINT` en schema/parser y predicados espaciales (`INTERSECTS`/`CONTAINS`).
- Integración en planner/executor y `CREATE INDEX ... USING RTREE`.
- Comparación experimental con PostgreSQL GIST.
- Versión paginada en disco (un nodo por página con `DiskManager`/`BufferManager`).

Ver API reference en [api/rtree](api/rtree.md) y la sección de [indexación](indexacion_algoritmos_externos.md).