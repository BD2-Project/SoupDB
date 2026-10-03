# Consultas espaciales

SoupDB implementa consultas espaciales avanzadas sobre el R-Tree existente,
manteniendo separados el núcleo del índice, los algoritmos de consulta y la
capa SQL.

La fachada principal es:

```python
from engine.indexes.rtree import SpatialQueries
```

Las métricas, geometrías y estadísticas se exponen desde:

```python
from engine.algorithms.spatial import (
    Polygon2D,
    SpatialQueryStats,
    euclidean_metric,
    haversine_metric,
    point_from_latlon,
)
```

## Arquitectura

El flujo básico de una consulta espacial es:

```text
RTree
  ↓
SpatialQueries
  ↓
algoritmos espaciales
  ↓
RID
```

La capa espacial devuelve principalmente `RID`.

La recuperación de la fila asociada corresponde posteriormente a la
organización física de la tabla mediante `fetch(rid)`.

Los algoritmos espaciales no conocen el parser SQL, el catálogo ni el formato
de los registros.

## Coordenadas

SoupDB conserva la representación existente:

```text
Point(x, y)
```

Para coordenadas cartesianas, `x` e `y` representan las unidades propias del
sistema utilizado.

Para coordenadas geográficas se adopta:

```text
Point(x=longitud, y=latitud)
```

Las coordenadas geográficas se expresan en grados.

Cuando una entrada externa se encuentra en orden:

```text
(latitud, longitud)
```

debe convertirse explícitamente mediante:

```python
point_from_latlon(latitude, longitude)
```

Por ejemplo:

```python
point = point_from_latlon(
    -12.0464,
    -77.0428,
)
```

produce internamente un punto equivalente a:

```python
Point(-77.0428, -12.0464)
```

## Métricas de distancia

### Euclidiana

```python
metric = euclidean_metric()
```

La distancia Euclidiana se calcula mediante:

```text
sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

y se expresa en las mismas unidades de las coordenadas.

La métrica incluye una cota mínima punto-MBR compatible con la distancia
Euclidiana, utilizada para poda en radio y k-NN.

### Haversine

```python
metric = haversine_metric()
```

Haversine interpreta los puntos como:

```text
x = longitud en grados
y = latitud en grados
```

y devuelve metros.

El radio terrestre utilizado por defecto es:

```text
6 371 000 metros
```

Puede proporcionarse otro radio explícitamente al construir la métrica.

La implementación normaliza la diferencia de longitud para considerar
correctamente cruces de ±180 grados.

La cota inferior utilizada durante la poda es conservadora: se basa en la
distancia mínima a la banda de latitudes del MBR.

Esta cota garantiza corrección, aunque puede podar menos nodos que una cota
geográfica más ajustada.

## Rango rectangular

API:

```python
rids = SpatialQueries(tree).range_search(box)
```

Ejemplo:

```python
from engine.indexes.rtree import MBR, SpatialQueries

box = MBR(
    min_x=-77.10,
    min_y=-12.10,
    max_x=-77.00,
    max_y=-12.00,
)

rids = SpatialQueries(tree).range_search(box)
```

Propiedades:

- los límites son inclusivos;
- se permiten MBR degenerados que representen un punto o una línea;
- las coordenadas deben ser finitas;
- se preservan entradas duplicadas;
- no existe garantía de orden global.

El recorrido utiliza siempre el MBR actual del nodo hijo para decidir la poda.

## Consulta por radio

API:

```python
rids = SpatialQueries(tree).radius_search(
    center,
    radius,
    metric,
)
```

Ejemplo geográfico:

```python
queries = SpatialQueries(tree)

center = point_from_latlon(
    -12.0464,
    -77.0428,
)

rids = queries.radius_search(
    center,
    5000.0,
    haversine_metric(),
)
```

La semántica es inclusiva:

```text
distance <= radius
```

El radio:

- debe ser numérico;
- debe ser finito;
- no puede ser negativo;
- puede ser cero.

La consulta recorre únicamente nodos cuya cota inferior todavía permite
encontrar un resultado dentro del radio.

En las hojas se calcula siempre la distancia exacta antes de aceptar una
entrada.

## k vecinos más cercanos

API:

```python
rids = SpatialQueries(tree).knn(
    center,
    k,
    metric,
)
```

Existe también:

```python
hits = SpatialQueries(tree).knn_hits(
    center,
    k,
    metric,
)
```

`knn_hits` permite conservar la información calculada durante la búsqueda,
incluyendo la distancia del resultado.

La implementación utiliza:

```text
best-first search
+
branch and bound
```

sobre el R-Tree.

La frontera de nodos se organiza por la cota inferior de distancia.

Cuando ya existen `k` candidatos, se mantiene como umbral la distancia del peor
candidato actual.

Un nodo solo puede descartarse cuando su cota es estrictamente mayor que ese
umbral.

La comparación estricta es necesaria porque un nodo con la misma distancia
todavía podría contener un candidato con mejor criterio de desempate.

### Semántica de k

- `k` debe ser un entero positivo;
- `bool` no se considera un valor válido de `k`;
- si el árbol está vacío se devuelve una lista vacía;
- si `k > N`, se devuelven los `N` elementos;
- se preservan entradas duplicadas;
- se devuelven exactamente `min(k, N)` resultados.

El orden final es determinista y sigue:

```text
(distancia, x, y, page_id, slot)
```

## Polígonos

La geometría de consulta se representa mediante:

```python
Polygon2D
```

Ejemplo:

```python
polygon = Polygon2D(
    (
        Point(0, 0),
        Point(4, 0),
        Point(4, 4),
        Point(0, 4),
    )
)
```

Consulta:

```python
rids = SpatialQueries(tree).polygon_search(polygon)
```

El algoritmo realiza:

```text
MBR del polígono
      ↓
prefiltro con R-Tree
      ↓
point-in-polygon exacto
```

La frontera del polígono se considera incluida.

`Polygon2D` valida:

- al menos tres vértices;
- vértices distintos;
- coordenadas válidas;
- área distinta de cero;
- ausencia de auto-intersecciones.

Si el primer vértice aparece repetido al final, el cierre se normaliza
internamente.

## Alcance actual de polígonos

La implementación actual soporta polígonos simples de un solo anillo.

No se soportan:

- agujeros;
- `MultiPolygon`;
- polígonos auto-intersectados;
- geometrías tridimensionales;
- polígonos geodésicos sobre la esfera;
- polígonos que requieran tratamiento especial del antimeridiano.

El predicado punto-en-polígono trabaja en el plano de las coordenadas
proporcionadas.

Para zonas geográficas locales esto debe interpretarse como una operación
planar sobre longitud/latitud, no como intersección geodésica esférica.

## Estadísticas de consulta

Las consultas pueden instrumentarse mediante:

```python
stats = SpatialQueryStats()
```

Ejemplo:

```python
stats = SpatialQueryStats()

rids = SpatialQueries(tree).knn(
    center,
    10,
    haversine_metric(),
    stats=stats,
)
```

Campos disponibles:

```text
nodes_visited
internal_nodes_visited
leaf_nodes_visited
mbr_tests
bound_evaluations
nodes_pruned
entries_examined
distance_evaluations
polygon_tests
heap_pushes
heap_pops
frontier_peak
candidates_peak
result_count
```

El objeto se reinicia automáticamente al iniciar cada consulta pública de
`SpatialQueries`.

Las estadísticas no modifican el resultado.

### Interpretación

`nodes_visited` representa nodos realmente inspeccionados por el algoritmo.

`distance_evaluations` cuenta distancias exactas punto-punto y no evaluaciones
de cotas.

`heap_pushes` y `heap_pops` corresponden a la frontera de nodos de k-NN.

`frontier_peak` representa el mayor número de nodos pendientes durante k-NN.

`candidates_peak` representa el máximo número de mejores candidatos mantenidos.

`result_count` conserva multiplicidad y coincide con el número de entradas
devueltas.

## Entradas del árbol

La fachada también permite obtener:

```python
entries = SpatialQueries(tree).entries()
```

Cada entrada contiene:

```text
(Point, RID)
```

Esta función es útil principalmente para validación, pruebas e integración.

No debe utilizarse como sustituto de una consulta indexada cuando el objetivo
sea medir rendimiento del R-Tree.

## Persistencia

Las consultas espaciales funcionan sobre árboles:

- construidos directamente en memoria;
- restaurados mediante la persistencia monolítica del R-Tree;
- restaurados mediante `RTreePageStore`.

Las pruebas de integración verifican que rango, radio, k-NN y polígonos
mantienen sus resultados después de persistir y recargar el árbol.

## Limitación de persistencia paginada

`RTreePageStore.load_tree()` reconstruye actualmente todo el árbol en memoria.

Por tanto, una consulta posterior mediante `SpatialQueries` recorre nodos
residentes en RAM.

Consecuentemente:

```text
nodes_visited != disk_reads
```

y:

```text
nodes_pruned != páginas físicas evitadas necesariamente
```

El tamaño del `BufferManager` afecta el proceso de guardar/cargar el árbol, pero
no convierte las consultas actuales en recorridos de páginas bajo demanda.

Esta distinción debe conservarse al interpretar benchmarks.

## Correctitud y poda

La poda depende de una propiedad fundamental:

```text
lower_bound(query, node_mbr)
    <=
distancia(query, cualquier punto descendiente)
```

Por esta razón una distancia y una cota no deben combinarse arbitrariamente.

`SpatialMetric` agrupa ambas operaciones para impedir que el algoritmo utilice
una cota incompatible con la métrica exacta.

La métrica Euclidiana utiliza su distancia mínima exacta al MBR.

Haversine utiliza actualmente una cota conservadora basada en latitud.

Una cota más débil puede visitar más nodos, pero sigue produciendo resultados
correctos.

Una cota que sobrestime la distancia podría producir falsos negativos y no debe
utilizarse.

## Duplicados

Las consultas espaciales preservan multiplicidad.

Dos entradas con:

```text
mismo Point
distinto RID
```

son resultados distintos.

También puede existir más de una entrada idéntica en datasets sintéticos o en
el baseline de comparación.

Por ello las validaciones de rango y radio deben comparar multiconjuntos y no
convertir resultados a `set`.

k-NN conserva además el orden definido por su clave total.

## Árbol cerrado

`SpatialQueries` respeta el ciclo de vida de `RTree`.

Una consulta sobre un árbol cerrado es rechazada de la misma manera que las
operaciones públicas del R-Tree.

## Comparación con baseline

El baseline secuencial se encuentra en:

```text
benchmarks/spatial_baseline.py
```

El workload reproducible se encuentra en:

```text
benchmarks/spatial_workload.py
```

Los escenarios preparados incluyen:

```text
datasets:
1 000
10 000
100 000 puntos

radios:
1 km
5 km
10 km

k:
10
50
100
```

La validación se ejecuta antes de cualquier medición de rendimiento.

Para rango y radio se compara preservando multiplicidad.

Para k-NN se exige igualdad exacta del orden producido por el baseline.

## Complejidad

El R-Tree reduce trabajo cuando sus MBR permiten descartar regiones de manera
selectiva.

No existe una garantía de que toda consulta sea sublineal.

En casos adversariales:

- MBR muy solapados;
- puntos repetidos;
- consultas que cubren casi todo el dataset;
- cotas geográficas poco selectivas;

el recorrido puede aproximarse a examinar todo el árbol.

Por ello las métricas de recorrido son más informativas que afirmar que el
R-Tree siempre es más rápido.

## Integración SQL

La integración SQL se documenta por separado en:

```text
docs/integracion_sql_espacial.md
```

La separación deliberada es:

```text
SQL / parser / planner
        ↓
SpatialQueries
        ↓
algoritmos espaciales
        ↓
RTree
```

Los algoritmos no reciben strings SQL.

La capa SQL consume las APIs públicas y conserva la semántica SQL de filtros,
ordenamiento y `LIMIT`.
