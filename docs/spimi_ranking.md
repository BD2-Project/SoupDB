# Índice invertido SPIMI y ranking

El módulo `engine/indexes/inverted/` construye bloques SPIMI con presupuesto de
memoria configurable. Los runs se fusionan mediante la API pública
`merge_sorted_runs` de `engine.algorithms.external_sort`; los archivos de entrada
pertenecen al llamador y los temporales de la fusión se eliminan al terminar.
El presupuesto estima las estructuras del bloque, no el RSS total de Python:
un documento o una posting list individual puede superar ese presupuesto.

## Persistencia y scoring

Cada directorio de índice contiene `manifest.json`, `postings.bin`, `terms.meta`
y `terms.hash`. El manifiesto conserva RID reales, longitudes, N, avgdl y la
configuración versionada del tokenizador. ExtendibleHash resuelve cada término
a un RID auxiliar del sidecar; este localiza df, offset y longitud de postings.
Los RID auxiliares no se devuelven como resultados de documentos.

La construcción usa un directorio temporal y publica mediante rename después
de validar una reapertura completa. `load` necesita únicamente los artefactos
persistidos, falla si faltan archivos y no reconstruye silenciosamente.

El tokenizador conserva acentos y ñ, aplica NFKC/casefold y stopwords ES/EN.
Las reglas de stemming heurísticas están documentadas en `tokenizer.py`; no
son Porter ni detectan automáticamente el idioma. Configuración y versión se
conservan para usar el mismo análisis en documentos y consultas.

TF-IDF usa frecuencia cruda, `idf = log(N/df)` y coseno sobre el vector completo
de cada documento. Las normas se recalculan desde postings al abrir el índice.
BM25 usa IDF positivo de Okapi y parámetros predeterminados k1=1.2, b=0.75.
Los empates se ordenan por RID ascendente.

`top_k` satisface RankedIndex y completa resultados con documentos de score
cero. `match_top_k`, utilizado por SQL MATCH, devuelve solo documentos que
comparten términos analizados. Las consultas vacías, de stopwords o desconocidas
devuelven `[]` en MATCH; una coincidencia con TF-IDF cero sigue siendo válida.

## Integración con SQL

La factory se inyecta explícitamente en `Catalog`; no se registra por importar
`provider.py`. Para texto, configurar `OnDiskInvertedFactory(storage_root)`.
`DelegatingRankedFactory({"INVERTED": text_factory, "IVF": vector_factory,
"HNSW": vector_factory})` permite compartir catálogo con los proveedores
vectoriales existentes. Esa instancia se pasa como `ranked_factory` al crear
y reabrir el catálogo. Los bindings invertidos son exactos y exhaustivos,
pero inmutables: las escrituras SQL posteriores al índice quedan bloqueadas.
Las filas SQL NULL se omiten del universo indexado; los textos vacíos se
conservan con longitud cero. Cargar los documentos antes de crear el índice.

## Muestra y benchmark reproducibles

Desde la raíz del repositorio, con el Python local:

```powershell
python -m scripts.sample_inverted_corpus --size 1000 --seed 7
python -m scripts.bench_inverted benchmarks/data/trec_covid_1k.jsonl --output benchmarks/data/inverted_bench.json
```

El sampler descarga por streaming el mismo corpus TREC-COVID que utiliza
`scripts/corpus.py`; no importa ese script ni ejecuta sus descargas adicionales.
Aplica reservoir sampling sobre todo el corpus, conserva `_id`, `title`, `text`
y comunica número de documentos de origen y SHA-256 de la muestra. También
admite `--input ruta/corpus.jsonl` para trabajar sin red. La reproducción exige
el mismo orden/contenido de origen y la misma semilla; el hash permite comprobarlo.
Los datos generados quedan bajo `benchmarks/data/`, excluido de Git.

El benchmark concatena título y texto, mide construcción, reapertura y consultas
TF-IDF/BM25, conserva el `_id` de origen en los resultados y compara RID y scores
antes/después de cerrar y abrir. Usa RID sintéticos únicamente para medir el
índice; las pruebas de integración SQL comprueban RID reales de almacenamiento.
Los índices temporales se eliminan también ante errores. Los tiempos de build
y load incluyen las validaciones y el cálculo de normas de la implementación.
No se mide precisión de relevancia con qrels en este benchmark básico.

## Corrida de validación: 2026-11-03

Windows, Python 3.12; una corrida con 1.000 documentos de los 171.332 del
corpus TREC-COVID, semilla 7, k=10 y presupuesto de bloque de 262.144 bytes.
Los tiempos son observaciones de este equipo, sin promesas de rendimiento.
Comando del benchmark: agregar `--memory-limit-bytes 262144` al ejemplo anterior.

SHA-256 de la muestra:
`3e997607225a722ee8ff2e4d2d91bdefa24b064e19f87fbb624d2693d8558212`.

| Medición | Resultado |
|---|---:|
| Construcción y validación | 13,001 s |
| Reapertura y normas | 3,407 s |
| Artefactos persistidos | 2.055.198 bytes |
| Consultas TF-IDF, tres consultas | 0,490–0,674 ms |
| Consultas BM25, tres consultas | 0,467–0,659 ms |

Las consultas fueron `coronavirus infection`, `clinical trial` y
`transmisión virus`. Cada método devolvió diez resultados por consulta y se
verificaron los mismos RID y scores después de reabrir. El informe completo
está en `benchmarks/data/inverted_bench.json`; muestra e informe permanecen
fuera de Git y se pueden regenerar con los comandos indicados.

También se verificó el recorrido SQL completo con esa muestra: carga de 1.000
filas, `CREATE INDEX ... USING INVERTED` mediante la factory delegada, consultas
MATCH TF-IDF/BM25 y reapertura de Catalog con los mismos resultados. La consulta
vacía devolvió cero filas. La construcción SQL tomó 16,156 s; se usaron páginas
heap de 8.192 bytes para admitir los documentos de esta muestra, cuyo texto
concatenado más largo ocupa 5.827 bytes UTF-8. Los temporales se eliminaron.
Resumen local: `benchmarks/data/inverted_sql_smoke.json`.

La validación final aprobó 563 tests de ranked, índice invertido, sort/merge y
SQL. La suite SQL vectorial real no se recolectó por ausencia de scikit-learn
en el entorno; las pruebas unitarias ranked vectoriales y de delegación pasaron.
