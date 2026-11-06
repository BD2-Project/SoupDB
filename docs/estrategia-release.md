# Estrategia de release

## Versionado

**SemVer** (`MAJOR.MINOR.PATCH`, tags `vX.Y.Z`) alineado a los Milestones del repositorio:

| Milestone | Fecha | Versión (tag) | Contenido |
|---|---|---|---|
| Avance 1 | 17/09/2026 | `v0.1.0` | Parte 1: almacenamiento, contratos, fakes |
| Entrega Parcial | 01/10/2026 | `v0.2.0` | Partes 2–3: índices espaciales/vectoriales |
| Avance 3 | 29/10/2026 | `v0.3.0` | Parte 4: extracción de características |
| Entrega Final | 19/11/2026 | `v1.0.0` | Parte 5: RAG completo |

- **PATCH** para fixes entre hitos (ej. `v0.1.1`).
- **MINOR** para features nuevas dentro de un avance.

## Contenido de cada release

- Changelog con los cambios del hito.
- **PDF del informe técnico** (`paper/`) — generado por el workflow `paper.yml`.
- **Wheel** del motor (`uv build`) — generado por el workflow de release.
- **Imagen Docker** publicada en GHCR (`ghcr.io/BD2-Project/soupdb`) — workflow `container.yml`.

## Flujo

1. Completar el hito y cerrar su Milestone.
2. Crear el tag `v*` (release draft).
3. Los workflows generan y adjuntan los artefactos automáticamente.
4. Publicar la release y el changelog.