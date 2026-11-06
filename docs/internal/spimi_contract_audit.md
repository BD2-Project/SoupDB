# SPIMI: contrato y auditoría de descubrimiento

Fecha: 2026-11-03. Fase 0; no contiene implementación del índice.

## Acuerdos del usuario

- Fuente: `scripts/corpus.py`, función `corpus()`, dataset TREC-COVID, campos `_id`, `title`, `text`.
- Exportar muestra reproducible de 1.000, 10.000 y 100.000 documentos con semilla fija y conservar esos campos. Indexar título y texto concatenados. Añadir casos pequeños ES/EN.
- Stopwords español e inglés; stemming de nivel intermedio. No referencias numéricas de issues en commits.
- Leer APIs completas está autorizado. Aclarar dudas descubiertas antes de Fase 1.
- El script tiene llamadas de descarga al nivel del módulo: importarlo ejecuta las tres lecturas. No se ejecutó ni descargó el corpus durante este análisis.

## Contrato canónico

Fuentes: `engine/indexes/base.py`, `engine/common/ranked.py`, `engine/common/ranked_catalog.py`, `engine/indexes/ranked_factory.py`.

`RankedIndex` es un ABC independiente de `Index`: exige solamente `top_k(query, k, **opts) -> list[tuple[RID, float]]` y `close() -> None`. No exige insert/remove/range_search. Consulta textual `str`; opciones `method="tf_idf" | "bm25"`. Scores de mayor a menor; opciones desconocidas producen ValueError según docstring (los tests admiten también TypeError).

`RID` es `NamedTuple(page_id: int, slot: int)`. El identificador `_id` del corpus no es un RID de almacenamiento: SQL debe conservar el RID real de `file_org.scan()`.

`RankedBinding`: name/table/column/algorithm/index obligatorios; metric/dimension/profile_id/item_count opcionales; exact/exhaustive/mutable/reversible booleanos con default False. Validación: nombres no vacíos, algoritmo INVERTED/IVF/HNSW, instancia RankedIndex, columna existente TEXT/VARCHAR. Para INVERTED metric, dimension y profile_id deben ser None. item_count debe ser int no negativo, nunca bool. Capacidades deben ser bool exacto.

`RankedMetadata` persiste name/table/column/algorithm/metric/dimension/profile_id/item_count/exact/exhaustive/mutable en `SysRankedIndexes`; no persiste reversible. No es un índice ni proporciona búsquedas por sí mismo.

## Checklist 1:1 de test_ranked_index_contract.py

La suite actual parametriza fake_vector, fake_text, brute_force, ivf y hnsw; todavía no contiene implementación invertida real. Pasarla sin agregar caso invertido no demostraría conformidad del nuevo índice.

| Test | Exigencia concreta |
|---|---|
| test_results_are_sorted_by_descending_score | Para dataset de 60 documentos, k=10 devuelve exactamente 10 hits; scores descendentes. |
| test_results_are_inserted_rids | Todo RID devuelto pertenece al conjunto construido. |
| test_k_larger_than_total_returns_everything | k=180 devuelve los 60 RID, incluidos documentos sin términos compartidos. No basta devolver únicamente postings coincidentes. |
| test_k_zero_returns_empty | k=0 devuelve []. |
| test_negative_k_is_rejected | k=-1 produce ValueError. |
| test_unknown_option_is_rejected | bogus_option produce ValueError o TypeError, sin method explícito. |
| test_persistence_after_close_and_reopen | Mismos RID en igual orden y scores aproximadamente iguales tras close/reopen. |

La suite no comprueba fórmulas TF-IDF/BM25, consulta vacía/OOV, float finito, duplicados, tipo de consulta, k booleano/no entero o errores de corrupción. Se requieren pruebas específicas independientes para esas propiedades. `fake_text` cuenta ocurrencias compartidas: no es un oráculo numérico de scoring real. El fake desempata por RID; la suite no exige directamente ese criterio, pero persistencia exige orden reproducible.

## Flujo SQL y lifecycle

No existe registro automático por presencia de `provider.py`. La frontera real es `Catalog(db_path, ranked_factory=factory)` en `engine/common/catalog.py`. Factory protocol:

```python
build(metadata: RankedMetadata, file_org: FileOrganization,
      schema: tuple[ColumnDef, ...]) -> RankedBinding
load(metadata: RankedMetadata, file_org: FileOrganization,
     schema: tuple[ColumnDef, ...]) -> RankedBinding
```

CREATE INDEX INVERTED llama `Catalog.create_ranked_index`: valida metadata, factory.build, coherencia de nombre/tabla/columna/algoritmo/metadata vectorial y register_ranked_index. Publica metadata después de build exitoso. Al reabrir, Catalog llama factory.load para cada metadata y valida binding; no debe reconstruir silenciosamente datos faltantes. Catalog.close cierra los índices. drop opcional de factory se consulta mediante getattr; el protocolo mínimo no lo declara.

SELECT MATCH -> `engine/query/ranked_planning.py:plan_text_search` -> binding único INVERTED de la columna -> choose_candidates -> InvertedIndexScan -> RankedScan.open -> index.top_k(query, request, method=estrategia.lower()) -> fetch de cada RID real -> score temporal FLOAT. SQL no calcula ranking. EXPLAIN sin ANALYZE no busca.

`choose_candidates`: LIMIT sin filtros, DISTINCT ni reordenamiento pide LIMIT+OFFSET. Sin LIMIT o con cualquiera de esos pasos requiere exact=True, exhaustive=True, item_count conocido y pide item_count. LIMIT=0 evita búsqueda. Declarar capacidades exige demostrar comportamiento, no solo añadir flags.

`RankedScan` exige lista, máximo k hits, tuples (RID,float), RID con componentes int exactos no negativos, float exacto finito, RID únicos y scores descendentes. RID obsoleto al fetch genera error. El operador no cierra el índice compartido al terminar una consulta.

`tests/integration/test_ranked_sql.py` prueba cuatro escenarios por API/TCP, scores proyectados FLOAT, EXPLAIN sin búsqueda y supervivencia del worker tras fallo. Usa ScriptedRankedIndex y FakeCatalog: pasarla no valida provider real ni sus fórmulas. Debe añadirse integración con Catalog real/build/load.

`docs/integracion_sql_rankeada.md` conserva secciones históricas de propuesta y dudas de coincidencias; usar código actual como evidencia de APIs, sin tratar propuestas antiguas como acuerdos nuevos. DML posterior debe quedar bloqueado con mutable=False/reversible=False; no anunciar mantenimiento reversible no implementado.

## APIs de almacenamiento y merge verificadas

ExtendibleHash: open(path, page_size=4096, buffer_capacity=64, max_global_depth=16) crea o reabre; insert(key,rid), search(key)->list[RID], remove(key,rid=None)->int, close idempotente. Claves int o str (bool inválido); hash string blake2b estable entre procesos. Solo almacena RID, no dicts ni metadatos arbitrarios. No API pública de iteración. `_hash_page.py` serializa RID como `<IH`: uint32 page_id y uint16 slot. Claves UTF-8 tienen límite por página; términos largos requieren política explícita. Reabrir requiere mismo page_size. `open` crea si falta: load del wrapper deberá verificar existencia primero.

external_sort(records: Iterable[Record], comparator, *, memory_limit_bytes, fan_in=8, temp_dir=None)->Iterator[Record]. API pública crea y fusiona sus propios runs; no admite paths de runs preordenados. Payload `Record.data` arbitrario; framing interno uint32 little-endian de longitud seguido de bytes. memory_limit_bytes>0, fan_in>=2; presupuesto limita payload serializado, no overhead Python; un registro individual puede superar el presupuesto. Temporales en TemporaryDirectory se eliminan al terminar/cerrar el generador, también ante excepción. `_merge_runs` y codecs con prefijo _ son privados.

## Decisiones confirmadas tras Fase 0

1. Diccionario: término -> RID auxiliar -> archivo sidecar con df/offset/longitud. Los RID auxiliares nunca se usan para recuperar documentos de tabla.
2. Merge: ampliar la API pública de external_sort para fusionar runs ya ordenados; no importar funciones privadas ni reordenar los bloques innecesariamente.
3. Coincidencias: top_k por defecto conserva la enumeración de todos los documentos requerida por la suite. MATCH usa un modo explícito de solo coincidencias del analizador, incluidas coincidencias cuyo score sea cero. Consulta vacía, OOV o formada solo por stopwords devuelve [] en ese modo; no inferir coincidencia a partir de score > 0.
4. Inyección: factory delegada por algoritmo, con provider INVERTED y delegación a providers vectoriales existentes. Mantener la inyección explícita Catalog(..., ranked_factory=...) y no alterar implementaciones vectoriales.

Estas decisiones fueron confirmadas por el usuario. No quedan pendientes las cuatro preguntas iniciales de esta fase.

## Estado de auditoría

Documento preparado por contract-analyst y revisado independientemente por qa-auditor. La revisión estática confirma los siete requisitos de la suite, la separación entre RID auxiliares y reales, las limitaciones de ExtendibleHash y la ausencia inicial de merge público de runs preexistentes.

El intento inicial de baseline falló por falta de pytest. Posteriormente se instaló pytest 9.2.0 localmente y el coordinador confirmó 79 tests aprobados en el conjunto de contrato ranked, ExtendibleHash, external_sort e integración SQL. No se descargaron datasets durante este descubrimiento.

Decisiones de diseño resueltas y baseline aprobado. La Fase 1 agrega únicamente módulos importables y un scaffold abstracto de InvertedIndex; su auditoría independiente debe registrarse antes del commit.

### Fase 1: revision independiente

AUDIT: PASSED - NO ACTIVE BUGS

qa-auditor verifico los 11 modulos nuevos: docstrings sin efectos laterales e InvertedIndex como subclase abstracta de RankedIndex, sin implementar funcionalidad prematuramente. La importacion explicita de todos los modulos paso. `python -m pytest tests/unit/indexes/test_ranked_index_contract.py --collect-only -q` recolecto 35 tests sin errores. No se modificaron implementaciones vectoriales ni archivos de usuario ajenos al alcance. Este gate cubre solamente el esqueleto de Fase 1; no acredita scoring, persistencia o integracion todavia.

### Fase 2: tokenizador

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de qa-auditor: normalizacion NFKC/casefold conserva acentos y ene; puntuacion y guiones bajos separan tokens; stopwords ES/EN se filtran antes de aplicar reglas explicitas de stemming. Stoplist y stemming configurables, configuracion inmutable y TOKENIZER_VERSION disponibles para persistencia posterior. Las reglas son heuristicas bilingues documentadas, sin prometer equivalencia linguistica completa.

Validacion: `python -m pytest tests/unit/indexes/inverted/test_tokenizer.py tests/unit/indexes/test_ranked_index_contract.py -q`: 68 tests aprobados (33 del tokenizador y 35 del contrato existente). Incluye normalizacion Unicode, vocabulario valido, stopwords personalizadas, plurales ES/EN, verbos/adverbios, entradas vacias e invalidas y determinismo. Persistencia del analizador se verificara cuando exista build/load; no se modificaron modulos vectoriales.

### Fase 3: API publica de runs ordenados

AUDIT: PASSED - NO ACTIVE BUGS

qa-auditor reviso write_sorted_run/read_sorted_run/merge_sorted_runs: framing reutilizado, validacion del orden al escribir, rechazo de sobrescritura, merge con fan_in limitado, preservacion de entradas y cleanup ante fallo o cierre temprano. ExitStack corrige liberacion de archivos ya abiertos si falla abrir otro run. Validacion independiente: `python -m pytest tests/unit/algorithms/test_external_sort.py tests/unit/algorithms/test_sorted_runs.py -q`: 19 tests aprobados. Esta microfeature no cambia APIs vectoriales ni implementa todavia SPIMI.

### Fase 3.1: lector de corpus

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente: conserva IDs fuente separados de RID, concatena title/text, admite aliases documentados, mantiene documentos vacios, rechaza IDs duplicados normalizados y registra ruta/linea al omitir JSON o campos invalidos. Lectura sin crear archivos faltantes. Las pruebas incluyen Unicode, los 12 pares de aliases y recuperacion tras lineas invalidas.

### Fase 3.2-3.4: bloques SPIMI y merge por termino

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente: TF por documento, IDs densos con mapa a RID reales, preservacion de longitudes cero, spills en directorios exclusivos y uso de la API publica de external_sort. Merge agrega TF de pares repartidos y produce terminos/postings ordenados. El presupuesto es aproximado y sus excepciones por documento/Counter estan documentadas. Fallos eliminan solo directorios propios, el merge cierra temporales y el caso sin runs elimina su directorio vacio. Hallazgo inicial del coordinador sobre ese directorio huerfano corregido antes del gate.

Validacion independiente de lector y SPIMI: `python -m pytest tests/unit/indexes/inverted/test_corpus_reader.py tests/unit/indexes/inverted/test_spimi.py -q`: 28 tests aprobados (17 lector, 11 SPIMI). La muestra externa 1k corresponde a validacion posterior, no se descargo aqui.

### Fase 4.1: postings binarios

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de PostingsFile: magic/version, conteo y pares uint64, IDs ordenados unicos, TF positivo, validacion antes de escritura y comprobacion de offset/longitud/df/bounds al leer. load no crea archivos faltantes, rechaza headers desconocidos y cierra el handle ante error; close es idempotente. `python -m pytest tests/unit/indexes/inverted/test_postings.py -q`: 9 tests aprobados, incluyendo round-trip, listas multiples/vacias y corrupcion/truncamiento. La limpieza de builds parciales se auditara en el componente que administra el directorio completo.

### Fase 4.2: diccionario sobre ExtendibleHash

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente: find consulta realmente ExtendibleHash.search y resuelve RID auxiliares al sidecar versionado, sin usar RID de documentos. Valida capacidad uint32/uint16 de esos punteros, longitud UTF-8 de terminos, metadata y coherencia hash/sidecar al reabrir. Los offsets separados permiten iterar e intercalar consultas. load exige artefactos existentes y cierra recursos ante corrupcion; managers propios permiten controlar fallos de apertura. `python -m pytest tests/unit/indexes/inverted/test_dictionary.py -q`: 7 tests aprobados, incluyendo 160 terminos con paginas pequenas, reapertura, duplicados, limites y corrupciones con liberacion de handles verificada en Windows.

### Fase 4.3-4.4: estadisticas y layout persistente

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de InvertedStore: manifest versionado persiste RID reales, doc_len, N, avgdl, page_size y configuracion/version del analizador. load exige los cuatro artefactos, verifica estadisticas, df, IDs de postings y suma TF por documento antes de devolver recursos. Los documentos vacios cuentan para N. build publica por rename desde staging tras validar reapertura, limpia runs/staging en fallo y rechaza sobrescribir destinos existentes. El analizador se reconstruye sin corpus ni descargas.

Validacion independiente: `python -m pytest tests/unit/indexes/inverted/test_store.py -q`: 11 tests aprobados. Dos comprobaciones adicionales con TemporaryDirectory confirmaron limpieza completa si un termino excede la pagina hash y rechazo de un doc_id corrupto en postings; tras fallar load se pudieron eliminar todos los archivos en Windows (sin handles filtrados). No se modificaron proveedores vectoriales.

### Fase 5.1-5.3: formulas de scoring

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de scoring.py: TF-IDF usa TF crudo y log(N/df); coseno define vectores cero como score cero y espera normas de vectores completos. BM25 usa IDF positiva log(1+(N-df+0.5)/(df+0.5)), k1=1.2 y b=0.75 por defecto; algebra de saturacion equivalente evita overflow para k1 grande. Parametros BM25 finitos y rangos validados explicitamente. Coincidencia permanece independiente del valor del score. `python -m pytest tests/unit/indexes/inverted/test_scoring.py -q`: 13 tests aprobados con calculo manual, normalizacion por longitud, ceros y parametros invalidos. Aplicacion de las formulas al indice completo pendiente en el siguiente gate.

### Fase 5.4: RankedIndex real

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de InvertedIndex: normas TF-IDF calculadas con todo el vector documental persistido; consulta usa TF crudo y omite OOV. Ambos metodos acumulan sobre postings y devuelven floats ordenados por score descendente/RID ascendente. top_k conserva universo completo con ceros; match_top_k usa union de documentos coincidentes, conserva ceros y devuelve [] para consulta vacia/stopwords/OOV. Valida k, query y opciones antes de buscar. close idempotente, apertura reconstruye normas sin corpus y cierra store si falla su inicializacion.

Validacion: `python -m pytest tests/unit/indexes/inverted/test_index.py tests/unit/indexes/test_ranked_index_contract.py -q`: 67 tests aprobados. Se agregaron casos reales TF-IDF y BM25 a la suite canonica conservando sus asserts originales, ademas de calculos manuales, frecuencia de consulta, reapertura y semantica MATCH separada. Integracion SQL del hook pendiente Fase 6.

### Fase 6: delegacion de factories y hook MATCH

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de DelegatingRankedFactory: copia la configuracion, enruta build/load/drop por algoritmo, rechaza algoritmos no configurados y propaga query_vectorizer compartido sin importar implementaciones multimedia. Hook InvertedIndexScan usa match_top_k solo cuando es callable; RankedScan y fakes conservan top_k y validacion de resultados. SQL real confirma exclusiones OOV/vacio/stopwords y conserva coincidencias TF-IDF con score cero. Validacion independiente combinada de factory, SQL real/fake y ejecucion textual: 39 tests aprobados. El gate separado del provider durable espera completar pruebas de fallo y validacion de descriptor.

### Fase 6.1-6.4: provider durable e integracion SQL real

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente: factory construye desde file_org.scan con RID reales, omite NULL y preserva textos vacios; declara exact/exhaustive y bloquea DML mediante capacidades inmutables. Descriptor vincula identidad SQL, algoritmo y cardinalidad con artefactos; load no accede al corpus. Staging y cleanup cubren error de entrada y fallo de reapertura posterior a publicacion. Drop/rebuild y Catalog reopen conservan resultados en ambos metodos. Hallazgo de version booleana/flotante aceptada por igualdad Python corregido con tipo int exacto y pruebas de regresion antes del gate.

Validacion independiente: `python -m pytest tests/unit/indexes/inverted/test_provider.py tests/unit/indexes/test_ranked_factory.py tests/integration/test_ranked_real_inverted_sql.py -q`: 20 tests aprobados. Ademas del conjunto previo de 39 tests, cubre identidad/cardinalidad, falta de hash sin recrearlo, load sin scan y cleanup de builds fallidos. Integracion real cubre filtros, DISTINCT, orden, offsets, score cero, consultas sin coincidencias y reapertura/drop/rebuild.

### Correccion de validacion BM25 tras revision final

AUDIT: PASSED - NO ACTIVE BUGS

La revision final adicional detecto OverflowError al pasar enteros gigantes a math.isfinite. El validador ahora transforma ese caso en ValueError consistente con opciones invalidas. Revision independiente del fix y `python -m pytest tests/unit/indexes/inverted/test_scoring.py tests/unit/indexes/inverted/test_index.py -q`: 33 tests aprobados. Regresiones cubren enteros gigantes para ambos parametros y k1 positivo subnormal sin score invalido. El hallazgo queda resuelto antes del cierre.

### Fase 7: muestra reproducible, benchmark y cierre

AUDIT: PASSED - NO ACTIVE BUGS

Revision independiente de sampler/benchmark/docs: reservoir sampling uniforme con semilla, schema confirmado, exportacion UTF-8 atomica, sin importar scripts/corpus.py ni ejecutar sus efectos laterales. Benchmark concatena titulo/texto, conserva IDs fuente, verifica ranking y reapertura de ambos metodos y limpia indices temporales incluso ante fallos. `python -m pytest tests/unit/indexes/inverted/test_bench_scripts.py -q`: 9 tests aprobados.

Corrida real del coordinador: muestra 1000 de 171332 documentos TREC-COVID, semilla 7, SHA-256 `3e997607225a722ee8ff2e4d2d91bdefa24b064e19f87fbb624d2693d8558212`. Presupuesto 262144 bytes, build 13.0007201 s, load 3.4068064 s, artefactos 2055198 bytes; seis consultas TF-IDF/BM25 con reapertura verificada. qa-auditor verifico independientemente hash del corpus, cardinalidad, pertenencia RID/ID fuente, unicidad y scores finitos descendentes del reporte local `benchmarks/data/inverted_bench.json`. Datos/reportes quedan excluidos de Git.

Regresion ampliada reportada por coordinador: 552 tests aprobados (unitarios ranked/inverted/sort e integracion SQL fake/real invertida). La integracion SQL vectorial real opcional no pudo recolectarse por dependencia multimedia preexistente ausente: sklearn. Los tests unitarios vectoriales ranked y routing de factories si pasaron; este limite no equivale a validacion multimedia completa. No se instalaron dependencias multimedia adicionales.

## Validación final del coordinador

563 tests aprobados tras agregar scripts de benchmark y regresión de parámetros
BM25 fuera del rango float. Comando: selección de tests/unit/*ranked*, suite
tests/unit/indexes/inverted, test_external_sort, test_sorted_runs e integración
SQL fake/real invertida. La suite SQL vectorial real no se ejecutó por falta
de scikit-learn; no se considera validación multimedia completa.

Smoke SQL real adicional: 1.000 documentos, heap page_size=8192, provider delegado,
build 16,156 s, MATCH TF-IDF/BM25 top-10 idénticos tras cerrar/reabrir Catalog,
MATCH vacío sin filas y temporales eliminados. Reporte local ignorado por Git:
benchmarks/data/inverted_sql_smoke.json.

## Regresion completa y revision de estilo posterior

AUDIT: PASSED - NO ACTIVE BUGS

El usuario solicito instalar las dependencias pendientes y ejecutar la regresion completa. La limitacion anterior de scikit-learn quedo resuelta; la integracion SQL vectorial real aprobo sus cuatro tests. qa-auditor verifico directamente `benchmarks/data/regression_full.xml`: 2330 tests, cero fallos, errores y skips, tiempo registrado 217.536 s. `regression_coverage.json` registra 8856/9322 lineas ejecutables cubiertas (95.0011%). Esta evidencia corresponde a la corrida completa previa al ajuste de estilo siguiente.

Ruff detecto siete incumplimientos de estilo en archivos de la sesion. Correccion revisada: DEFAULT_TOKENIZER compartido es seguro porque Tokenizer es frozen y su stoplist es frozenset; los otros cambios solo separan una linea larga y normalizan imports. Validacion independiente posterior: Ruff sobre todos los archivos Python de la sesion sin errores y 196 tests de inverted/contrato/SQL real invertida aprobados. No se detectaron bugs activos en esa correccion. El informe de sesion consolida resultados y limites de cobertura.

Revision independiente del informe de sesion completada: resultados y limites cotejados con XML/JSON y reportes locales. El XML posterior a los ajustes registra 567 tests, cero fallos/errores/skips y 7.439 s (7.45 s comunicado por pytest). Esta seleccion posterior incluye SQL vectorial real. El informe distingue la cobertura completa del snapshot anterior de la validacion focalizada posterior. AUDIT: PASSED - NO ACTIVE BUGS para el informe final.
