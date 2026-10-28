# Benchmark vectorial real (#129): resultados finales

Generado por `benchmarks/scripts/summarize_real_vector_results.py` desde `vector_indexes_real_final.csv`. Ningún número de este documento se escribió a mano, y no se volvió a ejecutar ningún índice. Técnicas: brute_force, ivf, hnsw; métricas: euclidean, cosine; k = 10; 50 consultas por dataset (objetos del propio dataset, sin contarse a sí mismos); 5 corridas de consulta por combinación; dimensión 128.

## Origen de los datos

| CSV de origen | datasets | filas | galería |
|---|---|---:|---|
| `vector_indexes_real_20261028_104529.csv` | audio 1k, image 1k | 72 | `vector_indexes_real_20261028_104529_gallery.html` |
| `vector_indexes_real_20261027_224129.csv` | audio 10k, image 10k | 72 | `vector_indexes_real_20261027_224129_gallery.html` |
| `vector_indexes_real_20261028_000200.csv` | image 100k | 36 | `vector_indexes_real_20261028_000200_gallery.html` |
| `vector_indexes_real_20261028_072721.csv` | audio 100k | 36 | `vector_indexes_real_20261028_072721_gallery.html` |

Consolidado: `vector_indexes_real_final.csv`, **216 filas** = 2 medios × 3 tamaños × 3 técnicas × 2 métricas × (1 build + 5 knn).

Validado: sin mediciones duplicadas; medios, tamaños, técnicas y métricas exactamente los esperados; k = 10; queries = 50; dim = 128; 1 build y 5 knn por combinación, con los mismos `params`.

Parámetros: IVF 1k: `nlist=32;nprobe=8` · 10k: `nlist=100;nprobe=8` · 100k: `nlist=316;nprobe=8`; HNSW 1k: `M=16;ef_construction=100;ef_search=64` · 10k: `M=16;ef_construction=100;ef_search=64` · 100k: `M=16;ef_construction=100;ef_search=64`.

## Datos

### Recall@10

Mediana de las 5 corridas (en este experimento todas dan el mismo valor). Contra la fuerza bruta de la misma métrica; la fuerza bruta es 1 por definición.

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 1.000 | 1.000 | 1.000 |
| image | brute_force | cosine | 1.000 | 1.000 | 1.000 |
| image | ivf | euclidean | 0.984 | 0.946 | 0.862 |
| image | ivf | cosine | 0.982 | 0.944 | 0.866 |
| image | hnsw | euclidean | 1.000 | 1.000 | 0.966 |
| image | hnsw | cosine | 1.000 | 1.000 | 0.966 |
| audio | brute_force | euclidean | 1.000 | 1.000 | 1.000 |
| audio | brute_force | cosine | 1.000 | 1.000 | 1.000 |
| audio | ivf | euclidean | 0.992 | 1.000 | 0.978 |
| audio | ivf | cosine | 0.994 | 1.000 | 0.980 |
| audio | hnsw | euclidean | 1.000 | 0.978 | 0.990 |
| audio | hnsw | cosine | 1.000 | 0.960 | 0.970 |

### Tiempo de construcción

Una construcción por combinación (`operation = build`).

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 0.05 s | 0.26 s | 1.79 s |
| image | brute_force | cosine | 0.08 s | 0.70 s | 1.53 s |
| image | ivf | euclidean | 3.06 s | 18.41 s | 77.6 s (1.3 min) |
| image | ivf | cosine | 2.61 s | 61.9 s (1.0 min) | 72.8 s (1.2 min) |
| image | hnsw | euclidean | 26.13 s | 276.0 s (4.6 min) | 3,508.0 s (58.5 min) |
| image | hnsw | cosine | 50.39 s | 1,100.5 s (18.3 min) | 5,111.2 s (85.2 min) |
| audio | brute_force | euclidean | 0.10 s | 0.21 s | 1.80 s |
| audio | brute_force | cosine | 0.09 s | 0.19 s | 2.99 s |
| audio | ivf | euclidean | 2.44 s | 19.77 s | 113.7 s (1.9 min) |
| audio | ivf | cosine | 2.66 s | 16.22 s | 118.1 s (2.0 min) |
| audio | hnsw | euclidean | 23.47 s | 92.1 s (1.5 min) | 2,328.6 s (38.8 min) |
| audio | hnsw | cosine | 42.45 s | 159.7 s (2.7 min) | 4,200.4 s (70.0 min) |

### Tiempo de consulta

ms por consulta: mediana de las 5 corridas (mínimo–máximo entre corridas).

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 8.180 (6.45–21.67) | 64.666 (53.83–82.00) | 497.635 (479.67–621.25) |
| image | brute_force | cosine | 20.495 (19.97–21.89) | 230.736 (222.96–235.51) | 621.834 (579.51–631.55) |
| image | ivf | euclidean | 5.872 (5.47–6.36) | 9.987 (7.95–10.53) | 29.571 (26.53–30.69) |
| image | ivf | cosine | 8.278 (7.64–8.73) | 32.450 (31.16–33.43) | 36.655 (32.62–38.25) |
| image | hnsw | euclidean | 8.807 (8.70–9.64) | 45.000 (43.32–48.25) | 17.885 (17.34–18.72) |
| image | hnsw | cosine | 11.289 (11.01–11.43) | 15.848 (15.22–16.59) | 19.485 (18.70–21.33) |
| audio | brute_force | euclidean | 16.341 (15.30–17.37) | 59.600 (57.21–61.74) | 776.649 (628.27–801.27) |
| audio | brute_force | cosine | 20.666 (19.79–22.20) | 57.746 (54.51–62.75) | 998.847 (948.01–1052.85) |
| audio | ivf | euclidean | 5.398 (5.28–5.49) | 4.859 (4.20–6.67) | 20.880 (19.89–22.21) |
| audio | ivf | cosine | 6.534 (5.98–6.82) | 4.794 (4.57–5.71) | 27.099 (25.53–29.23) |
| audio | hnsw | euclidean | 8.116 (7.57–8.49) | 3.142 (2.94–3.17) | 12.257 (11.43–12.96) |
| audio | hnsw | cosine | 9.961 (9.85–10.52) | 5.069 (5.00–6.04) | 31.524 (11.33–34.03) |

### Almacenamiento

`bytes_on_disk` del índice en MiB (2^20 bytes); el valor exacto en bytes está en el CSV.

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 0.56 | 5.59 | 55.81 |
| image | brute_force | cosine | 0.56 | 5.59 | 55.81 |
| image | ivf | euclidean | 0.65 | 5.84 | 56.55 |
| image | ivf | cosine | 0.65 | 5.84 | 56.55 |
| image | hnsw | euclidean | 0.84 | 8.29 | 82.68 |
| image | hnsw | cosine | 0.84 | 8.29 | 82.68 |
| audio | brute_force | euclidean | 0.56 | 5.59 | 55.81 |
| audio | brute_force | cosine | 0.56 | 5.59 | 55.81 |
| audio | ivf | euclidean | 0.64 | 5.82 | 56.55 |
| audio | ivf | cosine | 0.64 | 5.82 | 56.55 |
| audio | hnsw | euclidean | 0.84 | 8.29 | 82.68 |
| audio | hnsw | cosine | 0.84 | 8.29 | 82.68 |

### I/O de construcción

Páginas de 4 KiB: lecturas / escrituras durante la construcción.

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 144 / 144 | 1,430 / 2,604 | 14,287 / 28,318 |
| image | brute_force | cosine | 144 / 144 | 1,430 / 2,604 | 14,287 / 28,318 |
| image | ivf | euclidean | 166 / 166 | 1,516 / 2,754 | 27,765 / 41,987 |
| image | ivf | cosine | 166 / 166 | 1,516 / 2,754 | 27,765 / 41,987 |
| image | hnsw | euclidean | 216 / 216 | 5,755,150 / 137,280 | 156,405,228 / 3,092,237 |
| image | hnsw | cosine | 216 / 216 | 5,755,151 / 137,280 | 156,404,675 / 3,092,233 |
| audio | brute_force | euclidean | 144 / 144 | 1,430 / 2,604 | 14,287 / 28,318 |
| audio | brute_force | cosine | 144 / 144 | 1,430 / 2,604 | 14,287 / 28,318 |
| audio | ivf | euclidean | 163 / 163 | 1,567 / 2,800 | 31,608 / 45,828 |
| audio | ivf | cosine | 163 / 163 | 1,567 / 2,800 | 31,608 / 45,828 |
| audio | hnsw | euclidean | 216 / 216 | 965,071 / 27,242 | 40,211,420 / 1,227,902 |
| audio | hnsw | cosine | 216 / 216 | 965,203 / 27,294 | 40,248,268 / 1,228,384 |

### I/O de consulta

Lecturas de página por consulta (mediana de las corridas; cada corrida reabre el índice con el buffer pool vacío, 256 páginas).

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 2.86 | 1429 | 14286 |
| image | brute_force | cosine | 2.86 | 1429 | 14286 |
| image | ivf | euclidean | 3.2 | 154.18 | 714.6 |
| image | ivf | cosine | 3.2 | 156.02 | 719.28 |
| image | hnsw | euclidean | 4.26 | 823.94 | 1390.9 |
| image | hnsw | cosine | 4.26 | 823.94 | 1390.88 |
| audio | brute_force | euclidean | 2.86 | 1429 | 14286 |
| audio | brute_force | cosine | 2.86 | 1429 | 14286 |
| audio | ivf | euclidean | 3.06 | 98.74 | 350.56 |
| audio | ivf | cosine | 3.14 | 100.3 | 358.74 |
| audio | hnsw | euclidean | 4.26 | 176.52 | 447.7 |
| audio | hnsw | cosine | 4.26 | 177.26 | 453.94 |

### Memoria

KiB: retenida al reabrir el índice (build) / pico de Python durante las consultas (knn).

| medio | técnica | métrica | 1k | 10k | 100k |
|---|---|---|---:|---:|---:|
| image | brute_force | euclidean | 15.5 / 638.9 | 15.5 / 1148.5 | 15.5 / 1161.6 |
| image | brute_force | cosine | 14.7 / 638.9 | 14.7 / 1148.4 | 14.7 / 1161.5 |
| image | ivf | euclidean | 163.7 / 708.2 | 482.4 / 1147.0 | 1492.8 / 1161.6 |
| image | ivf | cosine | 163.3 / 708.2 | 482.1 / 1146.2 | 1492.5 / 1161.5 |
| image | hnsw | euclidean | 43.5 / 972.9 | 43.6 / 2378.1 | 44.8 / 2870.8 |
| image | hnsw | cosine | 42.7 / 972.9 | 42.9 / 2378.1 | 44.0 / 2870.8 |
| audio | brute_force | euclidean | 14.5 / 638.9 | 14.6 / 1148.4 | 15.5 / 1161.4 |
| audio | brute_force | cosine | 14.5 / 638.9 | 14.5 / 1148.4 | 14.7 / 1161.4 |
| audio | ivf | euclidean | 163.3 / 680.6 | 482.1 / 1145.0 | 1492.8 / 1161.2 |
| audio | ivf | cosine | 163.3 / 695.9 | 482.1 / 1144.2 | 1492.5 / 1161.2 |
| audio | hnsw | euclidean | 42.7 / 975.8 | 42.8 / 2344.0 | 44.8 / 2741.6 |
| audio | hnsw | cosine | 42.7 / 975.8 | 42.8 / 2352.0 | 44.0 / 2749.8 |

### Vectores duplicados

Vectores de cada `vectores.bin` idénticos a otro anterior del mismo archivo.

| medio | 1k | 10k | 100k |
|---|---:|---:|---:|
| image | 0 | 0 | 12 |
| audio | 0 | 28 | 446 |

### Anomalías observadas

Hechos que no siguen la tendencia general, detectados con reglas fijas del script (corridas con máximo/mínimo > 1.5; razón entre métricas fuera de [1/2, 2]; consulta más lenta con menos datos; recall que sube con N). Se dejan como están, sin corregir. Los tamaños vienen de ejecuciones distintas, así que las comparaciones de tiempo entre tamaños mezclan ejecuciones.

- Corridas dispersas: image 1k brute_force euclidean, 6.45–21.67 ms por consulta.
- Corridas dispersas: image 10k brute_force euclidean, 53.83–82.00 ms por consulta.
- Corridas dispersas: audio 10k ivf euclidean, 4.20–6.67 ms por consulta.
- Corridas dispersas: audio 100k hnsw cosine, 11.33–34.03 ms por consulta.
- Consulta cosine/euclidean = 2.5× en image 1k brute_force (lecturas iguales).
- Consulta cosine/euclidean = 3.6× en image 10k brute_force (lecturas iguales).
- Construcción cosine/euclidean = 2.7× en image 10k brute_force (lecturas iguales).
- Consulta cosine/euclidean = 3.2× en image 10k ivf (lecturas 156.02 / 154.18).
- Construcción cosine/euclidean = 3.4× en image 10k ivf (lecturas iguales).
- Consulta cosine/euclidean = 0.4× en image 10k hnsw (lecturas iguales).
- Construcción cosine/euclidean = 4.0× en image 10k hnsw (lecturas 5,755,151 / 5,755,150).
- Consulta cosine/euclidean = 2.6× en audio 100k hnsw (lecturas 453.94 / 447.70).
- Consulta más lenta con menos datos: image hnsw euclidean, 45.000 ms a 10k frente a 17.885 ms a 100k.
- Consulta más lenta con menos datos: audio ivf euclidean, 5.398 ms a 1k frente a 4.859 ms a 10k; el índice de 1k (163 páginas) cabe en el buffer de 256.
- Recall que sube con N: audio ivf euclidean, 0.992 a 1k → 1.000 a 10k.
- Consulta más lenta con menos datos: audio ivf cosine, 6.534 ms a 1k frente a 4.794 ms a 10k; el índice de 1k (163 páginas) cabe en el buffer de 256.
- Recall que sube con N: audio ivf cosine, 0.994 a 1k → 1.000 a 10k.
- Consulta más lenta con menos datos: audio hnsw euclidean, 8.116 ms a 1k frente a 3.142 ms a 10k; el índice de 1k (216 páginas) cabe en el buffer de 256.
- Recall que sube con N: audio hnsw euclidean, 0.978 a 10k → 0.990 a 100k.
- Consulta más lenta con menos datos: audio hnsw cosine, 9.961 ms a 1k frente a 5.069 ms a 10k; el índice de 1k (216 páginas) cabe en el buffer de 256.
- Recall que sube con N: audio hnsw cosine, 0.960 a 10k → 0.970 a 100k.

## Gráficas

Generadas desde el CSV maestro con `write_charts` (log-log; color = técnica, trazo continuo = euclidiana, discontinuo = coseno):

- `vector_indexes_real_final_audio_knn.svg`
- `vector_indexes_real_final_audio_build.svg`
- `vector_indexes_real_final_image_knn.svg`
- `vector_indexes_real_final_image_build.svg`

## Galerías (evidencia de las ejecuciones originales)

No se regeneran. Enlazan por `file://` a los datasets fuente, así que las imágenes y audios solo se ven en la máquina que los tiene. No se copió ningún archivo multimedia.

| galería | datasets | consultas | HIT | MISS | top-10 exacto euclidiana = coseno |
|---|---|---:|---:|---:|---:|
| `vector_indexes_real_20261028_104529_gallery.html` | audio 1k, image 1k | 6 | 227 | 13 | 6/6 |
| `vector_indexes_real_20261027_224129_gallery.html` | audio 10k, image 10k | 6 | 190 | 50 | 6/6 |
| `vector_indexes_real_20261028_000200_gallery.html` | image 100k | 3 | 70 | 50 | 3/3 |
| `vector_indexes_real_20261028_072721_gallery.html` | audio 100k | 3 | 96 | 24 | 3/3 |

---

## Interpretación

Esta sección resume tendencias visibles en las tablas anteriores. Donde se ofrece una explicación, es una hipótesis compatible con los datos, no algo que este experimento demuestre.

**Escalabilidad IVF frente a HNSW.** En image, construir HNSW (euclidean) a 100k costó 3,508.0 s (58.5 min) frente a 77.6 s (1.3 min) de IVF (45.2×); en audio, construir HNSW (euclidean) a 100k costó 2,328.6 s (38.8 min) frente a 113.7 s (1.9 min) de IVF (20.5×). La fuerza bruta en image lee 2.86 → 1429 → 14286 páginas por consulta y su tiempo crece 7.7× de 10k a 100k. La fuerza bruta en audio lee 2.86 → 1429 → 14286 páginas por consulta y su tiempo crece 13.0× de 10k a 100k. A 100k (euclidean, image) una consulta tardó 497.6 / 29.6 / 17.9 ms (brute_force / ivf / hnsw). A 100k (euclidean, audio) una consulta tardó 776.6 / 20.9 / 12.3 ms (brute_force / ivf / hnsw).

**Trade-off construcción / consulta / recall.** A 100k: construcción más cara: hnsw en todos los casos. Consulta más rápida: hnsw en image·euclidean, image·cosine, audio·euclidean; ivf en audio·cosine. Mayor recall entre las aproximadas: hnsw en image·euclidean, image·cosine, audio·euclidean; ivf en audio·cosine. IVF en image: fracción de listas visitadas (nprobe/nlist) 25.0% → 8.0% → 2.5%, recall 0.984 → 0.946 → 0.862; la caída es compatible con visitar una fracción menor, aunque el experimento no varía nprobe para confirmarlo. IVF en audio: fracción de listas visitadas (nprobe/nlist) 25.0% → 8.0% → 2.5%, recall 0.992 → 1.000 → 0.978; el recall no cae de forma monótona con N.

**Image frente a audio.** Recall a 100k (euclidean, image / audio) con los mismos parámetros: ivf: 0.862 / 0.978; hnsw: 0.966 / 0.990. Lecturas al construir HNSW a 100k: 156,405,228 / 40,211,420 páginas. Los datos muestran estas diferencias, no su causa.

**Vectores L2-normalizados.** Los vectores TF-IDF del pipeline tienen norma 1 (`tfidf_vector`), y para vectores unitarios `||a − b||² = 2 − 2·cos(a, b)`. La distancia euclidiana es entonces una función monótona decreciente del coseno, y la fuerza bruta debe ordenar igual con las dos métricas salvo empates y redondeo float32. En las galerías, el top-10 exacto euclidiano coincide con el coseno en 18 de 18 consultas mostradas. La fuerza bruta lee las mismas páginas con las dos métricas, así que su diferencia de tiempo entre métricas no viene del I/O. Estos datos no permiten separar el costo de cada fórmula del ruido de medición.

**IVF y la métrica.** Los centroides de IVF son promedios de vectores unitarios y no tienen norma 1. Ordenarlos por euclidiana o por coseno puede elegir listas distintas. Las lecturas por consulta de IVF difieren entre métricas en 5 de 6 datasets, y la mayor diferencia de recall entre métricas es 0.004.

**HNSW y la métrica.** HNSW guarda la métrica y se construyó un grafo por métrica. El recall difiere entre métricas en: audio 10k (0.978 / 0.960); audio 100k (0.990 / 0.970). La construcción con cosine tardó entre 1.46× y 3.99× lo de euclidean, con lecturas que difieren como mucho un 0.09%. Los datos no explican esa diferencia de tiempo.

**Limitación del Recall@10 con empates.** Entre vectores empatados (duplicados), el orden depende de la técnica, y una técnica aproximada puede devolver un empatado distinto del que eligió la fuerza bruta. Esa respuesta es igual de cercana pero cuenta como MISS, así que el Recall@10 puede quedar algo subestimado; el experimento no cuantifica cuánto. Datasets con duplicados: audio 10k: 28, audio 100k: 446, image 100k: 12.
