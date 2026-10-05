# Experimentación: búsqueda espacial

Comparación experimental 2.2.4 del enunciado. Se miden tres técnicas sobre los
mismos datos y las mismas consultas: búsqueda secuencial, R-Tree y PostgreSQL con
índice GiST.

## 1. Metodología

El workload lo genera `benchmarks/spatial_workload.py` a partir de una semilla
fija, así que las tres técnicas reciben exactamente los mismos puntos y los mismos
centros de consulta. Los puntos se distribuyen de forma uniforme sobre el área de
Lima metropolitana (longitud −77.20 a −76.90, latitud −12.20 a −11.90).

| Parámetro | Valores |
|---|---|
| Tamaño del dataset | 1 000 / 10 000 / 100 000 |
| Radios | 1 km / 5 km / 10 km |
| k para k-NN | 10 / 50 / 100 |
| Consultas por parámetro | 20 |
| Corridas por combinación | 5, se reporta la mediana |

Se reporta la mediana y no el promedio porque la primera corrida suele venir
contaminada por el cache del sistema operativo y arrastraría el promedio.

La métrica de distancia es haversine en metros para las tres técnicas.

### Reproducción

```bash
uv run python -m benchmarks.scripts.compare_spatial_indexes

POSTGRES_PORT=55433 docker compose up -d postgres
POSTGRES_PORT=55433 uv run python -m benchmarks.scripts.compare_postgres_gist
```

Los CSV quedan en `benchmarks/results/spatial_indexes_*.csv`. El panel de
*Benchmarks* de SoupChef los grafica sin configuración adicional.

## 2. Verificación de correctitud

Antes de comparar rendimiento hay que comprobar que las tres técnicas devuelven
lo mismo: un índice que poda de más sería arbitrariamente rápido y los números no
significarían nada.

La búsqueda secuencial actúa de oráculo. Los conteos de resultados coinciden en
las 21 combinaciones entre el R-Tree y el scan, y en 20 de 21 contra PostgreSQL.

La excepción es el radio de 10 km con 100 000 registros: 398 964 resultados contra
398 965. La diferencia es de **un punto sobre casi cuatrocientos mil** y se explica
por el radio terrestre: PostGIS usa 6 371 008,77 m para su esfera y el motor usa
6 371 000 m. A 10 km eso son unos 14 mm, suficiente para que un punto que cae justo
sobre el borde quede de un lado o del otro.

## 3. Cómo se cuentan los accesos a disco

El costo en accesos a disco es la métrica que evalúa el curso, porque no depende
de la máquina. Conviene ser explícito sobre cómo se obtuvo cada número.

- **Construcción del R-Tree**: medida directa con `DiskManager`, que cuenta las
  páginas realmente escritas al persistir el árbol.
- **Consultas del R-Tree**: una lectura por nodo visitado. Las consultas corren
  sobre el árbol en memoria, porque `RTreePageStore` persiste y carga el árbol
  entero y no página por página; un `DiskManager` no registraría ninguna lectura
  durante una búsqueda y reportar cero sería engañoso. Como un nodo ocupa
  exactamente una página, contar nodos visitados equivale a contar páginas.
- **Búsqueda secuencial**: todas las páginas de su archivo de entradas, porque no
  tiene índice y debe recorrer todo.
- **PostgreSQL**: `shared_hit + shared_read` del plan de `EXPLAIN (ANALYZE,
  BUFFERS)`.

El R-Tree se configura con `order = 102`, el máximo que entra en una página de
4096 bytes. Con el orden por defecto (8) cada nodo gastaría una página entera para
ocho entradas y la medición reflejaría ese desperdicio en vez de la poda del
índice: con `order = 8` el R-Tree hacía 1084 lecturas contra 120 del scan
secuencial, invirtiendo la conclusión.

## 4. Resultados

Tiempo en milisegundos y páginas leídas, para 20 consultas por fila.

### 1 000 registros

| Operación | Secuencial | R-Tree | PostgreSQL GiST |
|---|---|---|---|
| Radio 1 km | 26,4 ms · 120 p | **6,6 ms · 90 p** | 910,1 ms · 2 235 p |
| Radio 5 km | 26,6 ms · 120 p | **11,6 ms** · 144 p | 901,1 ms · 2 405 p |
| Radio 10 km | 26,3 ms · **120 p** | **17,3 ms** · 203 p | 905,6 ms · 2 441 p |
| k-NN, k=10 | 26,9 ms · 120 p | **12,2 ms · 105 p** | 920,4 ms · 3 858 p |
| k-NN, k=100 | 28,1 ms · 120 p | **21,7 ms** · 167 p | 911,7 ms · 5 524 p |

### 10 000 registros

| Operación | Secuencial | R-Tree | PostgreSQL GiST |
|---|---|---|---|
| Radio 1 km | 267,0 ms · 1 180 p | **40,6 ms · 419 p** | 887,9 ms · 2 818 p |
| Radio 5 km | 271,0 ms · 1 180 p | **100,7 ms · 1 001 p** | 885,5 ms · 4 463 p |
| Radio 10 km | 271,2 ms · **1 180 p** | **165,3 ms** · 1 625 p | 893,8 ms · 4 862 p |
| k-NN, k=10 | 282,6 ms · 1 180 p | **52,6 ms · 354 p** | 895,0 ms · 3 890 p |
| k-NN, k=100 | 285,9 ms · 1 180 p | **85,1 ms · 561 p** | 904,1 ms · 5 726 p |

### 100 000 registros

| Operación | Secuencial | R-Tree | PostgreSQL GiST |
|---|---|---|---|
| Radio 1 km | 2 896,6 ms · 11 780 p | **316,1 ms · 2 797 p** | 892,0 ms · 8 802 p |
| Radio 5 km | 2 914,8 ms · 11 780 p | **1 039,0 ms · 8 968 p** | 1 796,7 ms · 32 959 p |
| Radio 10 km | 2 948,0 ms · **11 780 p** | **1 808,4 ms** · 15 531 p | 7 664,3 ms · 36 436 p |
| k-NN, k=10 | 3 319,9 ms · 11 780 p | **241,9 ms · 1 436 p** | 911,9 ms · 3 950 p |
| k-NN, k=50 | 3 347,9 ms · 11 780 p | **295,0 ms · 1 829 p** | 914,2 ms · 4 788 p |
| k-NN, k=100 | 3 541,8 ms · 11 780 p | **358,5 ms · 2 102 p** | 916,3 ms · 5 819 p |

### Construcción y espacio en disco, 100 000 registros

| Técnica | Construcción | Tamaño |
|---|---|---|
| Secuencial | 0,3 ms | 2,4 MB (solo datos) |
| R-Tree | 47 689,7 ms | 5,9 MB |
| PostgreSQL GiST | 508,3 ms | 7,4 MB |

## 5. Análisis

### El scan secuencial escala linealmente y el R-Tree no

El tiempo del scan se multiplica por diez cada vez que el dataset se multiplica
por diez: 26 ms → 271 ms → 2 897 ms. Es el comportamiento esperado de un
recorrido completo, y además su costo es insensible a la consulta: lee las mismas
11 780 páginas tanto para un radio de 1 km como para uno de 10 km, y tanto para
k=10 como para k=100.

El R-Tree, en cambio, paga por lo que realmente necesita mirar. Con 100 000
registros y k=10 visita 1 436 páginas contra 11 780 del scan: **8,2 veces menos**.

### Existe un punto de cruce por selectividad

La ventaja del índice depende de cuánto devuelve la consulta, no del tamaño del
dataset.

Con 100 000 registros, el radio de 10 km devuelve 398 965 resultados en 20
consultas, es decir unos 19 948 por consulta: cerca del **20 % del dataset**. A esa
selectividad el R-Tree tiene que visitar casi todo el árbol y además paga el
recorrido de los nodos internos, así que termina leyendo 15 531 páginas contra
11 780 del scan completo: **lee un 32 % más**.

En tiempo sigue ganando (1 808 ms contra 2 948 ms) porque evita evaluar la
distancia en todos los puntos, pero en accesos a disco pierde. Esta es la
conclusión central del experimento: **el índice conviene mientras la consulta sea
selectiva; cuando devuelve una fracción grande de la tabla, el recorrido completo
es preferible**. Es el mismo razonamiento por el que un planificador descarta un
índice cuando estima baja selectividad.

El cruce se ve progresar con el tamaño: con 1 000 y 10 000 registros el R-Tree ya
leía más páginas que el scan en el radio de 10 km, pero la diferencia era chica;
con 100 000 se vuelve clara.

### La construcción es la debilidad del R-Tree

Construir el árbol con 100 000 puntos toma 47,7 segundos, contra 0,5 de
PostgreSQL. La causa es que se inserta punto por punto, con división cuadrática en
cada desborde, sin carga masiva. Un algoritmo de *bulk loading* como STR ordenaría
los puntos y construiría el árbol de abajo hacia arriba, evitando las divisiones.
Es la mejora más rentable si se quisiera seguir trabajando sobre esta estructura.

El scan secuencial no construye nada, que es justamente su única ventaja real:
**si la carga de datos domina y las consultas son pocas, el índice no se amortiza**.

### Sobre la comparación con PostgreSQL

Nuestro R-Tree queda por delante de GiST en las tres operaciones, pero afirmarlo
sin matices sería incorrecto.

El tiempo de PostgreSQL se mantiene casi constante alrededor de 900 ms en las
consultas chicas, independientemente del tamaño del dataset. Eso indica que está
dominado por costo fijo —conexión, planificación, instrumentación de `EXPLAIN
ANALYZE`— y no por el trabajo de la consulta. En otras palabras, en los tamaños
chicos no estamos midiendo su índice sino su overhead.

Además PostgreSQL paga cosas que nuestro índice no paga: control de concurrencia
multiversión, durabilidad, un planificador que decide la estrategia, y un cómputo
geodésico más caro. Nuestro R-Tree opera en memoria sobre un árbol ya cargado.

Lo que la comparación sí permite afirmar es que **el R-Tree implementado está en
el mismo orden de magnitud que un índice espacial de producción**, lo cual era el
objetivo. No permite afirmar que sea mejor.

Se verificó con `EXPLAIN` que PostgreSQL efectivamente usa el índice (`Bitmap Heap
Scan` sobre `Bitmap Index Scan`) y no lo descarta por un recorrido secuencial, lo
que habría invalidado la comparación.

## 6. Conclusión

| Situación | Técnica preferible |
|---|---|
| Consultas selectivas sobre datos grandes (k-NN, radios chicos) | R-Tree |
| Consultas que devuelven más del 20 % de la tabla | Búsqueda secuencial |
| Pocas consultas sobre datos que se cargan una vez | Búsqueda secuencial |
| Muchas consultas sobre datos estables | R-Tree |

El R-Tree justifica su costo de construcción cuando el volumen de consultas
selectivas lo amortiza. Para consultas de baja selectividad, o cuando la carga
domina sobre la consulta, el recorrido completo sigue siendo la opción correcta.

## 7. Limitaciones

- Las consultas del R-Tree no pasan por el buffer pool: la persistencia paginada
  existe pero carga el árbol completo. Los accesos durante las consultas se
  derivan de los nodos visitados, no se miden con `DiskManager`.
- Los puntos se distribuyen de forma uniforme. Datos reales se agrupan en zonas
  densas, lo que favorecería más al índice, porque la poda por MBR descarta
  regiones vacías con mayor efectividad.
- Las mediciones de PostgreSQL incluyen el costo fijo del proceso, que domina en
  los tamaños chicos.
