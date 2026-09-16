# Decisiones de diseño

El proyecto tiene presentación final con preguntas. Cada decisión debe poder defenderse. Se registran aquí a medida que se toman.

- **Página de 4096 bytes:** coincide con el tamaño de bloque típico del sistema de archivos.
- **RID = (page_id, slot)** en vez de offset absoluto: permite compactar dentro de la página.
- **Clave compuesta** para índices sobre campos con duplicados.
- **Borrado lazy + reorganización al 30%** en el archivo secuencial: el 30% se mide de forma GLOBAL (espacio reclamable / espacio ocupado en todo el archivo, no por página) y, al superarse, compacta todas las páginas con espacio reclamable. La compactación solo reordena bytes dentro de cada página; nunca mueve registros entre páginas, así que el RID `(page_id, slot)` no cambia.
- **Modelo Volcano en el ejecutor:** permite componer operadores sin materializar resultados.
- **SemVer por hitos del curso:** releases `v0.1.0`..`v1.0.0` alineadas a los Milestones (ver `estrategia-release.md`).
- **Publicación del contenedor en GHCR** junto a cada release con tag `v*`.