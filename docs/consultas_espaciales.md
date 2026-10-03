# Consultas espaciales

Este documento define las convenciones utilizadas por los algoritmos de consulta espacial de SoupDB.

## Coordenadas

La estructura existente `Point(x, y)` se conserva como representación interna.

Para coordenadas geográficas:

- `x` representa longitud.
- `y` representa latitud.
- Las coordenadas se expresan en grados.

Una entrada externa expresada como `(latitud, longitud)` debe convertirse explícitamente al orden interno `(longitud, latitud)`.

## Métricas de distancia

Se soportarán dos métricas:

- Distancia Euclidiana para coordenadas cartesianas.
- Distancia Haversine para coordenadas geográficas.

La distancia Haversine se expresará en metros.

El radio terrestre utilizado inicialmente será:

```text
6 371 000 metros
```

## Consultas por radio

Las consultas por radio serán inclusivas:

```text
distancia <= radio
```

El radio debe ser finito y no negativo.

## k-NN

`k` debe ser un entero positivo.

Si `k` es mayor que la cantidad de elementos disponibles, se devolverán todos los elementos.

Los resultados se ordenarán utilizando la siguiente clave:

```text
(distancia, x, y, page_id, slot)
```

Esto permite obtener resultados deterministas cuando existen empates de distancia.

Las entradas duplicadas se conservarán.

## Consultas por rango

Las consultas rectangulares utilizarán `MBR`.

Los límites del rectángulo serán inclusivos.

Se permitirán MBR degenerados que representen un punto o una línea, siempre que sus coordenadas sean válidas.

## Consultas con polígonos

La primera implementación soportará polígonos simples de un único anillo.

La frontera del polígono será considerada parte del polígono.

Inicialmente no se soportarán:

- agujeros;
- `MultiPolygon`;
- polígonos que crucen el antimeridiano;
- polígonos auto-intersectados.

## R-Tree

Los algoritmos de consulta reutilizarán el `RTree`, `RTreeNode`, `Point` y `MBR` existentes.

No se implementará un segundo árbol espacial ni una segunda representación de puntos.

Durante los recorridos de nodos internos se utilizará el `mbr` actual del nodo hijo para realizar las decisiones de poda.

Los algoritmos de consulta no modificarán la estructura del árbol.

## Resultados

Las consultas espaciales devolverán principalmente `RID`.

Cuando una operación lo necesite, podrán utilizarse resultados auxiliares que incluyan el punto y la distancia calculada.

Las métricas y estadísticas de ejecución se mantendrán fuera del contrato general `Index`.