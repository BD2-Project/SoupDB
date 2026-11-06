# Informe de sesión: SPIMI, ranking e integración SQL

Fecha: 3 de noviembre de 2026. Entorno: Windows, Python 3.12, repositorio SoupDB.

## Resultado y alcance

Se implementó un índice invertido persistente construido con SPIMI, diccionario
de términos sobre ExtendibleHash, postings binarios, TF-IDF con similitud coseno
y BM25. El índice cumple RankedIndex y dispone de un provider SQL real para
`CREATE INDEX ... USING INVERTED` y `MATCH ... USING TF_IDF/BM25`.

La ejecución fue incremental, con revisión independiente por microfeature y
commits posteriores a los gates `AUDIT: PASSED - NO ACTIVE BUGS`. El registro
detallado de cada revisión se conserva en `spimi_contract_audit.md`. Este informe
describe el resultado de la sesión y distingue las comprobaciones anteriores
de la nueva regresión completa solicitada al cierre.

No se modificaron las implementaciones IVF/HNSW/vectoriales. La composición de
factories permite reutilizarlas en el mismo catálogo. No se publicaron cambios
en servicios externos ni se incluyeron números de issues en los commits nuevos.

## Decisiones confirmadas antes de implementar

1. El corpus procede de la misma fuente que `scripts/corpus.py`: TREC-COVID,
   campos `_id`, `title` y `text`. Se indexa la concatenación de título y texto.
2. La muestra contiene 1.000 documentos seleccionados con semilla fija; conserva
   los identificadores originales. Las pruebas pequeñas cubren español e inglés.
3. El analizador utiliza stopwords ES/EN y stemming heurístico documentado,
   suficiente para el nivel intermedio solicitado, sin dependencias lingüísticas.
4. ExtendibleHash almacena únicamente RID. Por ello, término -> RID auxiliar ->
   sidecar de metadatos es la representación aprobada. Estos RID auxiliares no
   son identificadores de documentos ni se entregan al operador SQL.
5. Se amplió la API pública de external_sort para fusionar runs ya ordenados;
   SPIMI no importa sus funciones privadas ni vuelve a ordenar todos los bloques.
6. `top_k` conserva el universo completo que exige la suite canónica; SQL MATCH
   usa un modo explícito de solo coincidencias del analizador. El score cero no
   se interpreta como ausencia de coincidencia.
7. La factory se inyecta expresamente en Catalog y se delega por algoritmo.
   Importar `provider.py` no registra automáticamente ningún proveedor.
8. SQL NULL se omite del universo indexado; un texto vacío sí cuenta para N,
   tiene longitud cero y no produce coincidencias MATCH.

## Implementación por fases

| Fase | Resultado entregado |
|---|---|
| 0: descubrimiento | Contrato RankedIndex, checklist de siete tests, mapa SQL y APIs reales documentados; dudas resueltas antes de Fase 1. |
| 1: estructura | Paquete inverted importable y scaffold abstracto; colección de tests sin errores. |
| 2: analizador | NFKC/casefold, separación de puntuación, stopwords configurables, stemming reproducible y configuración inmutable. |
| 3: SPIMI | Reader JSONL, conteos TF por bloque, runs exclusivos, merge público limitado por fan_in y agrupación por término/documento. |
| 4: persistencia | Postings binarios, diccionario hash/sidecar, estadísticas y manifest versionado; staging y validación de reapertura. |
| 5: ranking | Fórmulas verificadas, normas documentales completas, top_k persistente y MATCH separado; casos reales añadidos a suite canónica. |
| 6: SQL | Provider durable, delegación de factories, hook MATCH compatible con fakes y pruebas con Catalog/RID reales. |
| 7: validación | Sampler reproducible, benchmark de 1k, smoke SQL de 1k y documentación de uso. |

El nuevo paquete contiene tokenizer, corpus_reader, spimi, postings, dictionary,
merge, scoring, index, provider y utils, además de `store.py`, que administra el
ciclo de vida de los artefactos persistentes. Los exports del paquete se mantienen
mínimos. Los scripts nuevos son `scripts/sample_inverted_corpus.py` y
`scripts/bench_inverted.py`.

## Arquitectura y persistencia

El flujo de construcción es: documentos y RID -> tokenización -> Counter por
documento -> bloques término/documento/TF -> runs ordenados -> merge multipaso ->
postings y metadatos -> validación de reapertura -> publicación mediante rename.

Los documentos reciben IDs internos densos; el manifiesto conserva el mapa a
RID reales. Los identificadores `_id` del corpus se conservan para trazabilidad
en el sampler y el benchmark, sin reinterpretarlos como RID de tablas SQL.

| Artefacto | Contenido |
|---|---|
| manifest.json | Versión, page_size, configuración/version del analizador, RID y longitud por documento, N y avgdl. |
| postings.bin | Magic/version; cada lista contiene count uint64 y pares ordenados `(doc_id, tf)` uint64. |
| terms.meta | Magic/version; término UTF-8 y df/offset/longitud uint64; offsets de registros para enumeración. |
| terms.hash | ExtendibleHash persistente, término -> RID auxiliar del sidecar. |
| sql_index.json | En el directorio del provider SQL: versión, identidad SQL, algoritmo y cardinalidad, junto al subdirectorio index. |

Los nombres físicos SQL se derivan con SHA-256 para impedir que un nombre de
índice escape de storage_root. build rechaza sobrescribir índices existentes y
limpia sus propios temporales ante fallo. load exige los artefactos existentes:
no crea un hash vacío ni reconstruye silenciosamente desde la tabla o corpus.

La reapertura valida coherencia entre hash y sidecar, versiones, límites de
postings, df, documentos referenciados, sumas TF por documento y estadísticas.
Los archivos se cierran también al detectar corrupción. Las listas de postings
deben tener IDs únicos ascendentes y TF positivos.

## Contrato y fórmulas

`top_k(query, k, **opts)` devuelve `list[tuple[RID, float]]`, con scores finitos
descendentes y desempate por RID ascendente. k negativo se rechaza; k cero
devuelve []; opciones desconocidas se rechazan. k mayor que N devuelve todos
los documentos indexados, incluidos scores cero. La reapertura conserva orden,
RID y scores aproximadamente iguales.

Para TF-IDF se utiliza TF crudo y `idf(t) = log(N / df(t))`. Tanto consulta como
documento usan pesos TF por IDF; el coseno normaliza por sus normas completas.
Las normas de documentos se derivan de todos sus términos persistidos, no solo
de los términos de la consulta. Los vectores de norma cero tienen score cero;
los términos OOV no aportan peso.

BM25 utiliza `log(1 + (N - df + 0.5) / (df + 0.5))` y la saturación
`tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))`, multiplicada por
la frecuencia del término en la consulta. Los defaults son k1=1.2 y b=0.75.
La implementación reordena algebraicamente la saturación para evitar overflow
con k1 grande. Rechaza parámetros no finitos, k1 <= 0, b fuera de [0,1] y enteros
que exceden el rango float; se verificó también k1 positivo subnormal.

`match_top_k` limita candidatos a la unión de postings de los términos
analizados. Consulta vacía, solo stopwords o completamente OOV devuelve [].
Una coincidencia TF-IDF con df=N permanece en resultados aunque su score sea
cero. Este comportamiento evita mezclar el contrato general de RankedIndex con
la semántica de coincidencia textual SQL.

## Integración SQL y capacidades

La configuración es `Catalog(db_path, ranked_factory=factory)`. Para texto se
usa `OnDiskInvertedFactory(storage_root)`. Si se combinan algoritmos:

```python
factory = DelegatingRankedFactory({
    "INVERTED": text_factory,
    "IVF": vector_factory,
    "HNSW": vector_factory,
})
```

La delegación enruta build/load/drop y conserva el query_vectorizer compartido
sin importar multimedia. Distintos resolvers vectoriales requieren un resolver
compartido explícito; no se selecciona uno arbitrariamente.

Catalog construye el provider antes de registrar metadata, lo recarga al abrir
la base y cierra los índices al cerrar. El provider toma RID reales de scan,
decodifica la columna elegida y omite NULL. load no vuelve a escanear la tabla.
DROP INDEX elimina los artefactos tras cerrar el índice y permite reconstruir.

InvertedIndexScan llama match_top_k cuando el provider lo expone; conserva
top_k como fallback para proveedores existentes y fakes. RankedScan continúa
validando tipo float finito, RID válidos y únicos, orden y cantidad de hits;
recupera las filas con FileOrganization.fetch y no recalcula puntuaciones.

Los bindings invertidos declaran exact=True, exhaustive=True e item_count
conocido. Sin LIMIT, o con filtros residuales, DISTINCT o reordenamiento, SQL
solicita el universo indexado para aplicar esos pasos correctamente. Con LIMIT
simple solicita LIMIT+OFFSET. Las capacidades mutable/reversible son False:
INSERT/DELETE afectados se bloquean después de crear el índice. La carga inicial
de filas se realiza antes del CREATE INDEX.

## Tests y auditorías durante la implementación

Se añadieron suites específicas para tokenizer, corpus_reader, spimi, postings,
dictionary, store, scoring, index, provider y scripts de benchmark. También se
añadieron tests de runs públicos y delegación de factories, integración SQL real
invertida y dos casos reales TF-IDF/BM25 a la suite RankedIndex existente, sin
debilitar sus asserts originales.

Las comprobaciones incluyen fórmulas manuales, tokenización ES/EN, aliases JSONL,
IDs duplicados, corrupción y truncamiento, límites de páginas, múltiples spills,
cleanup ante excepción/cierre temprano, reapertura sin corpus/scan, RID reales,
scores cero, OOV, vacíos, stopwords, NULL, filtros, DISTINCT, orden, offsets,
DROP/rebuild y bloqueo DML. Las pruebas de fakes SQL se conservaron como regresión
y se distinguieron de las pruebas del provider físico real.

La selección ampliada previa aprobó 563 tests. Inicialmente, la suite SQL
vectorial real no pudo recolectarse por falta de scikit-learn; ese bloqueo del
entorno se resolvió posteriormente. Tras instalar dependencias, sus cuatro
tests aprobaron. Estos resultados son antecedentes; no se presentan como la
nueva regresión completa solicitada.

Los hallazgos corregidos antes de sus gates fueron el directorio de SPIMI vacío
sin runs, la aceptación de versión bool/float por igualdad Python en el descriptor
SQL y OverflowError al validar parámetros BM25 enteros gigantes. La ampliación
de external_sort también garantiza cierre de archivos ya abiertos si falla una
apertura posterior, usando ExitStack.

## Entorno y dependencias locales

Se instaló pytest para habilitar el baseline y, al ampliar la regresión,
scikit-learn y los grupos locales dev/multimedia. Esto resolvió la limitación de
colección multimedia anterior; no se modificaron las dependencias declaradas
del proyecto para implementar el índice. No se afirma que falten ahora esas
dependencias opcionales.

Versiones consultadas al preparar este informe: pytest 9.2.0, pytest-cov 7.1.0,
Hypothesis 6.170.14, Ruff 0.17.3, coverage 7.17.1, scikit-learn 1.9.1, NumPy 2.5.4,
librosa 1.0.0, opencv-python-headless 5.0.0.93, soundfile 0.14.0 y
threadpoolctl 3.7.0.

## Corpus y mediciones reales de 1.000 documentos

El sampler procesa por streaming la fuente TREC-COVID sin importar
scripts/corpus.py y sin ejecutar sus lecturas adicionales. Reservoir sampling
retiene O(tamaño de muestra) documentos. La reproducción exige mismo contenido,
orden de origen, tamaño y semilla; el hash comprueba la muestra producida.

Se seleccionaron 1.000 de 171.332 documentos, semilla 7. SHA-256:
`3e997607225a722ee8ff2e4d2d91bdefa24b064e19f87fbb624d2693d8558212`.

| Medición | Resultado observado |
|---|---:|
| Presupuesto aproximado del bloque | 262.144 bytes |
| Build, validaciones y normas | 13,0007201 s |
| Load, validaciones y normas | 3,4068064 s |
| Artefactos persistidos | 2.055.198 bytes |
| Consultas TF-IDF | 0,490–0,674 ms |
| Consultas BM25 | 0,467–0,659 ms |
| Build por recorrido SQL real | 16,1560435 s |

Se consultó `coronavirus infection`, `clinical trial` y `transmisión virus` con
ambos métodos y k=10. El benchmark verificó los mismos RID y scores tras
reabrir. Utiliza RID sintéticos exclusivamente para medir el índice aislado;
el smoke SQL adicional utilizó heap y RID reales de 1.000 filas.

Ese smoke creó el índice por factory delegada, obtuvo top-10 TF-IDF/BM25
idénticos tras reabrir Catalog y confirmó cero filas para MATCH vacío. Usó
heap page_size=8192 para admitir la muestra, cuyo texto concatenado máximo es
5.827 bytes UTF-8. Los artefactos temporales se eliminaron al terminar.

Datos e informes locales ignorados por Git:

- benchmarks/data/trec_covid_1k.jsonl: muestra reproducible.
- benchmarks/data/inverted_bench.json: tiempos, IDs fuente, RID y scores.
- benchmarks/data/inverted_sql_smoke.json: resumen del recorrido SQL real.
- benchmarks/data/regression_full.xml: salida JUnit de la nueva regresión.
- benchmarks/data/regression_coverage.json: cobertura medida de la nueva regresión.
- benchmarks/data/regression_post_review.xml: regresión focalizada tras ajustes de estilo.
- benchmarks/data/regression_post_review.xml: regresión focalizada posterior al ajuste de estilo.

Los tiempos describen una corrida local, sin comparación de rendimiento ni
promesa de latencia. La consulta española no demuestra que el corpus real sea
bilingüe ni sustituye las pruebas unitarias ES/EN. No se midió calidad de
relevancia con qrels, recall, MAP o nDCG.

## Nueva regresión completa solicitada al cierre

Resultado: **2.330 tests aprobados, cero fallos, errores u omisiones**, sobre
el snapshot de implementación c0c920a. Duración comunicada por pytest:
217,93 s; el JUnit registra 217,536 s para su testsuite. La diferencia corresponde
a las mediciones de cada salida. Los resultados se verificaron leyendo el XML
y el JSON generados al finalizar esta ejecución.

Comando de la ejecución instrumentada:

```powershell
python -m pytest -o addopts= -q --hypothesis-seed=20261103 --junitxml=benchmarks/data/regression_full.xml --cov=engine --cov-report=json:benchmarks/data/regression_coverage.json
```

Semilla Hypothesis fijada en 20261103. La cobertura de líneas medida para
`engine` fue **95,00 %**: 8.856 líneas cubiertas de 9.322, 466 sin cubrir y tres
excluidas. En `engine/indexes/inverted` fue **93,52 %**: 693 de 741 líneas,
48 sin cubrir y cero excluidas. No se midió cobertura de ramas; los porcentajes
de líneas no prueban por sí solos ausencia de bugs.

| Área de la suite completa | Tests aprobados |
|---|---:|
| Unitarios de algoritmos | 146 |
| Unitarios de benchmarks | 118 |
| Unitarios common | 117 |
| Unitarios de índices | 447 |
| Unitarios multimedia | 89 |
| Unitarios de query | 1.031 |
| Unitarios de almacenamiento | 202 |
| Unitarios de transacciones | 139 |
| Integración gestor E2E | 9 |
| Integración SQL invertida real | 11 |
| Integración SQL vectorial real | 4 |
| Integración SQL con fakes | 13 |
| Integración espacial | 4 |
| Total | 2.330 |

| Módulo inverted | Líneas cubiertas / medibles | Cobertura |
|---|---:|---:|
| corpus_reader | 34 / 34 | 100,00 % |
| dictionary | 130 / 141 | 92,20 % |
| index | 82 / 85 | 96,47 % |
| merge | 29 / 30 | 96,67 % |
| postings | 64 / 69 | 92,75 % |
| provider | 90 / 98 | 91,84 % |
| scoring | 30 / 30 | 100,00 % |
| spimi | 57 / 60 | 95,00 % |
| store | 115 / 131 | 87,79 % |
| tokenizer | 62 / 63 | 98,41 % |

`__init__.py` y utils no contienen sentencias medibles. Las áreas con líneas
sin cubrir se conservan explícitas; este informe no afirma cobertura total.
`pip check` terminó sin requisitos rotos.

Después de esta corrida se identificaron siete incidencias de estilo Ruff en
el alcance modificado. Se corrigieron los defaults mediante DEFAULT_TOKENIZER,
una instancia inmutable compartida, y detalles de formato. La revisión
independiente confirmó que no cambian el comportamiento. Ruff pasó en todos
los archivos Python de la sesión. El auditor ejecutó 196 tests de inverted,
contrato e integración SQL invertida, todos aprobados. La selección ampliada
posterior del coordinador aprobó 567 tests en 7,45 s; su XML registra 567 tests,
cero fallos, errores u omisiones y 7,439 s. Incluye ranked, inverted, sort/merge,
SQL con fakes y SQL real invertida/vectorial.

No se atribuye la cobertura del snapshot c0c920a a una nueva corrida completa
sobre esos ajustes. qa-auditor revisó este informe y contrastó directamente
los XML de ambas corridas, el JSON de cobertura y los reportes de benchmark.
**AUDIT: PASSED - NO ACTIVE BUGS** para el informe y los ajustes revisados.

## Commits de la sesión

| Commit | Cambio |
|---|---|
| b50d86d | Scaffold y contrato auditado. |
| 816383f | Tokenizador bilingüe configurable. |
| 9382a38 | API pública de runs y merge limitado. |
| 107d59e | Lector JSONL validado. |
| 2cf8b8f | Bloques SPIMI y merge de postings por término. |
| f111478 | Postings binarios versionados. |
| 7ac6db0 | Diccionario sobre ExtendibleHash. |
| 8bae0fd | Artefactos SPIMI completos y reapertura. |
| 316781a | Fórmulas TF-IDF/coseno y BM25. |
| 1d75c6f | Índice textual ranked persistente. |
| fb82773 | Delegación de factories por algoritmo. |
| 55dbe15 | Provider durable y SQL MATCH real. |
| a023d71 | Rechazo consistente de parámetros BM25 fuera del rango float. |
| c0c920a | Muestra reproducible, benchmark y documentación. |
| fcdcaf2 | Default inmutable compartido y ajustes de estilo validados. |

El commit documental que contiene este informe se consulta mediante `git log`;
no se incluye a sí mismo en la tabla. Los archivos de usuario ajenos al trabajo
no se incorporaron a los commits.

## Límites conocidos y trabajo futuro

El presupuesto SPIMI es aproximado y controla el bloque, no el RSS completo:
la tokenización/Counter de un documento y una lista de postings individual
pueden superarlo. El mapa de documentos, longitudes y normas ocupa O(N); el
diccionario mantiene offsets de registros para enumerar términos. No se guarda
un índice completo de términos en memoria para consultar.

Las normas TF-IDF se recalculan desde postings al abrir. Esto facilita la
persistencia coherente, pero añade trabajo a load. El stemming es heurístico,
no Porter, y conserva acentos; no hay detección automática de idioma.

El índice es inmutable desde SQL; no implementa actualizaciones incrementales,
undo reversible ni recuperación atómica frente a corte de energía. El rename y
cleanup comprobados cubren publicación y errores del proceso. Los términos
deben caber en una página de ExtendibleHash: los demasiado largos fallan con
error explícito. Para SQL, el tamaño de página de la tabla debe admitir cada
registro del corpus.

El benchmark básico no mide relevancia ni escalabilidad por encima de 1k.
Los cuatro tests multimedia reales usan extracción de descriptores simulada
con índices físicos reales; aprobarlos no constituye un benchmark sobre un
dataset multimedia real ni certifica perfiles de extracción.
