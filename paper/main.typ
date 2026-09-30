#let titleblock(title, subtitle) = [
  #align(center)[
    #text(size: 24pt, weight: "bold")[#title]
    #v(4pt)
    #text(size: 13pt)[#subtitle]
  ]
]

#titleblock(
  "SoupDB: Minigestor de Base de Datos Multimodal",
  "Informe técnico · Base de Datos 2 · UTEC · ciclo 2026-2",
)

= Resumen

#lorem(60)

= Introducción

#lorem(80)

= Arquitectura

#lorem(80)

= Almacenamiento y organización de archivos

#lorem(80)

= Indexación y optimización

La capa de indexación implementa índices persistentes sobre páginas de disco:
un árbol B+ no agrupado y agrupado con búsqueda por igualdad y rango inclusivo,
y un hash extensible con directorio y buckets en disco. Los algoritmos externos
(k-way merge y hashing por particionamiento) sostienen el modo spill de Sort y
Aggregate.

Para la Parte 2 (almacenamiento espacial de coordenadas y mapas) se desarrolló
el núcleo de un R-Tree en memoria (`engine/indexes/rtree/`). El árbol indexa
puntos 2D agrupándolos en nodos de capacidad fija, cada uno con su rectángulo
mínimo envolvente (MBR). La inserción desciende eligiendo el hijo de menor
enlargement del MBR y, al desbordar un nodo, aplica una división cuadrática
(semillas que maximizan el área desperdiciada y asignación del resto por menor
crecimiento). Los MBRs se actualizan en cada mutación, lo que permite podar
búsquedas puntuales (por contención) y de rango (por intersección). La
estructura implementa el contrato Index del gestor y expone una interfaz de
persistencia (save, load y open) mediante un codec binario que serializa el
árbol completo, dejando que la versión paginada en disco se construya sobre
este núcleo. El borrado es simple (sin condense tree ni reinsert), limitación
documentada y acotada al alcance del núcleo.

= Procesamiento de consultas SQL

#lorem(80)

= Transacciones y concurrencia

#lorem(80)

= Benchmarks

#lorem(80)

= Conclusiones

#lorem(60)

#bibliography("refs.bib")