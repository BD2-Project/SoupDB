# Integración SQL con consultas espaciales

Este documento define el contrato entre los algoritmos espaciales y la futura
extensión SQL de SoupDB.

La implementación de parsing, AST, planificación y ejecución SQL espacial no
pertenece a este módulo. La capa SQL debe consumir las APIs documentadas aquí
sin duplicar los algoritmos de búsqueda.

## 1. Componentes disponibles

La fachada principal para consultar un R-Tree es:

```python
from engine.indexes.rtree import SpatialQueries
```

Las métricas y utilidades espaciales se encuentran en:

```python
from engine.algorithms.spatial import (
    Polygon2D,
    SpatialQueryStats,
    euclidean_metric,
    haversine_metric,
    point_from_latlon,
)
```

Las estructuras base se importan desde:

```python
from engine.indexes.rtree import MBR, Point, RTree
```

Uso básico:

```python
queries = SpatialQueries(tree)
```

`SpatialQueries` trabaja sobre un `RTree` existente y devuelve RIDs asociados
a los registros almacenados en la tabla.

## 2. Convención de coordenadas

La representación interna es:

```text
Point(x=longitud, y=latitud)
```

para datos geográficos.

Por ejemplo, Lima:

```python
Point(-77.0428, -12.0464)
```

El ejemplo SQL del proyecto utiliza:

```sql
POINT(-12.0464, -77.0428)
```

es decir:

```text
POINT(latitud, longitud)
```

Por tanto, la frontera SQL no debe construir directamente:

```python
Point(latitude, longitude)
```

Debe convertir explícitamente mediante:

```python
point_from_latlon(latitude, longitude)
```

Ejemplo:

```python
center = point_from_latlon(
    -12.0464,
    -77.0428,
)
```

## 3. Métricas disponibles

### Euclidiana

```python
metric = euclidean_metric()
```

La distancia usa las mismas unidades de las coordenadas.

Debe emplearse cuando las coordenadas pertenecen a un espacio cartesiano o
proyectado.

### Haversine

```python
metric = haversine_metric()
```

Interpreta:

```text
Point.x = longitud en grados
Point.y = latitud en grados
```

y devuelve distancias en metros.

Esta es la métrica que debe utilizarse para ejemplos SQL geográficos como:

```sql
distancia(ubicacion, POINT(...)) < 5000
```

donde `5000` representa 5000 metros.

## 4. Consulta rectangular

API:

```python
rids = queries.range_search(box)
```

Ejemplo:

```python
box = MBR(
    min_x=-77.10,
    min_y=-12.10,
    max_x=-77.00,
    max_y=-12.00,
)

rids = queries.range_search(box)
```

La frontera del MBR es inclusiva.

No existe garantía de orden global para los resultados.

Los duplicados se preservan.

## 5. Consulta por radio

API:

```python
rids = queries.radius_search(
    center,
    radius,
    metric,
)
```

Ejemplo geográfico:

```python
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

La semántica de la API es:

```text
distance <= radius
```

El radio debe ser finito y no negativo.

### Integración con WHERE

Para:

```sql
SELECT *
FROM tiendas
WHERE distancia(
    ubicacion,
    POINT(-12.0464, -77.0428)
) < 5000;
```

la capa SQL puede usar `radius_search(..., 5000, ...)` para obtener candidatos,
pero debe conservar el comparador SQL original.

En particular:

```text
radius_search    -> distancia <= 5000
consulta SQL     -> distancia < 5000
```

Por tanto, para `<` debe mantenerse un filtro residual exacto que elimine un
resultado cuya distancia sea exactamente 5000.

La capa de algoritmos no debe cambiar su semántica inclusiva para imitar cada
operador SQL.

## 6. k-NN

API de RIDs:

```python
rids = queries.knn(
    center,
    k,
    metric,
)
```

Ejemplo:

```python
rids = queries.knn(
    center,
    10,
    haversine_metric(),
)
```

También existe:

```python
hits = queries.knn_hits(
    center,
    k,
    metric,
)
```

`knn_hits` permite reutilizar las distancias calculadas durante la búsqueda.

Cada hit contiene la información espacial necesaria para representar el
resultado de la consulta, incluyendo RID, punto y distancia.

Propiedades:

- `k` debe ser un entero positivo.
- Si `k > N`, se devuelven los `N` elementos existentes.
- Se preservan entradas duplicadas.
- Se devuelven exactamente `min(k, N)` resultados.
- Los resultados tienen un orden determinista.
- La búsqueda utiliza best-first search y branch and bound sobre el R-Tree.

## 7. Integración con ORDER BY ... LIMIT

Para:

```sql
SELECT *
FROM restaurantes
ORDER BY distancia(
    ubicacion,
    POINT(-12.0464, -77.0428)
)
LIMIT 10;
```

la traducción conceptual es:

```python
center = point_from_latlon(
    -12.0464,
    -77.0428,
)

hits = SpatialQueries(tree).knn_hits(
    center,
    10,
    haversine_metric(),
)
```

Esta optimización corresponde a:

```text
ORDER BY distancia(...) ASC
LIMIT k
```

No debe utilizarse directamente para:

```sql
ORDER BY distancia(...) DESC
```

porque k-NN busca los vecinos más cercanos.

## 8. WHERE adicional y k-NN

Una consulta como:

```sql
SELECT *
FROM tiendas
WHERE categoria = 'gasolinera'
ORDER BY distancia(ubicacion, POINT(...))
LIMIT 10;
```

no puede resolverse correctamente haciendo:

```text
knn(k=10)
→ filtrar categoria
```

Ese procedimiento podría devolver menos de diez resultados y omitir
gasolineras válidas más lejanas.

Mientras no exista búsqueda k-NN con predicado de aceptación o un iterador
incremental de vecinos, la capa SQL debe utilizar una estrategia correcta como:

```text
scan
→ filtro
→ cálculo de distancia
→ sort
→ limit
```

La optimización directa con `SpatialQueries.knn()` debe reservarse inicialmente
para consultas compatibles sin filtros adicionales que alteren el conjunto
elegible.

## 9. Polígonos

API:

```python
rids = queries.polygon_search(polygon)
```

Ejemplo:

```python
polygon = Polygon2D(
    (
        Point(-77.10, -12.10),
        Point(-77.00, -12.10),
        Point(-77.00, -12.00),
        Point(-77.10, -12.00),
    )
)

rids = queries.polygon_search(polygon)
```

La implementación actual soporta polígonos simples.

La frontera del polígono se considera incluida.

La consulta utiliza el MBR del polígono como prefiltro del R-Tree y después
aplica el predicado punto-en-polígono exacto.

## 10. Resolución de RIDs

Las consultas espaciales retornan RIDs, no filas completas.

La capa SQL debe resolver cada RID contra la organización física de la tabla:

```python
record = file_org.fetch(rid)
```

y posteriormente decodificar el registro usando el schema correspondiente.

La capa espacial no debe depender del formato de las filas ni del parser SQL.

## 11. Estadísticas opcionales

Las consultas pueden recibir:

```python
stats = SpatialQueryStats()
```

Ejemplo:

```python
rids = queries.radius_search(
    center,
    5000.0,
    haversine_metric(),
    stats=stats,
)
```

Las estadísticas permiten observar, entre otros:

- nodos visitados;
- nodos internos y hojas visitadas;
- pruebas de MBR;
- evaluaciones de cotas;
- nodos podados;
- entradas examinadas;
- evaluaciones de distancia;
- operaciones de la frontera k-NN;
- máximo tamaño de frontera;
- cantidad de resultados.

Las estadísticas son diagnósticas y no cambian el resultado de la consulta.

El objeto se reinicia al comenzar cada llamada pública de `SpatialQueries`.

## 12. Responsabilidad de la capa SQL

La integración SQL debe encargarse de:

1. Reconocer `POINT(...)`.
2. Reconocer `distancia(...)`.
3. Soportar funciones escalares con múltiples argumentos.
4. Incorporar `LIMIT`.
5. Evaluar distancia dentro de expresiones escalares.
6. Detectar consultas compatibles con búsqueda por radio.
7. Detectar `ORDER BY distancia(...) ASC LIMIT k` compatible con k-NN.
8. Mantener filtros residuales necesarios.
9. Resolver RIDs a registros.
10. Convertir los errores de la API espacial al modelo de errores de consulta.
11. Integrar el R-Tree con catálogo y ciclo de vida del índice.

Estas responsabilidades no requieren modificar los algoritmos espaciales.

## 13. Responsabilidad de la capa espacial

La capa espacial proporciona:

- distancia Euclidiana;
- distancia Haversine;
- cotas inferiores compatibles con cada métrica;
- búsqueda rectangular;
- búsqueda por radio;
- k-NN best-first / branch and bound;
- consulta punto-en-polígono;
- preservación de duplicados;
- orden determinista en k-NN;
- estadísticas de recorrido;
- fachada `SpatialQueries`.

No procesa strings SQL ni conoce el AST, planner, executor o catálogo.

## 14. Estado actual de integración

La API espacial está disponible para ser consumida por la extensión SQL.

La conexión completa:

```text
SQL
→ Parser / AST
→ Planner
→ operador espacial
→ SpatialQueries
→ RTree
→ RID
→ Record
```

debe realizarse en la capa de procesamiento SQL.

El objetivo de este contrato es permitir que ambas partes evolucionen de forma
independiente sin duplicar lógica espacial ni acoplar los algoritmos al parser.