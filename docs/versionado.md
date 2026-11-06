# Versionado y entregas

Convención formalizada el 2026-11-02 tras la auditoría de ramas, tags y milestones (eran los primeros tags del repositorio).

## Esquema

- SemVer simplificado `vX.Y.Z` mediante **tags anotados** de Git.
- `X.YZ`: bump de **minor** para cada entrega de milestone o feature fusionada a `main`; **patch** para fixes publicados entre entregas.
- El tag se crea **siempre** en el commit de merge `dev -> main` del PR de publicación (`chore: publicar en main ...`).
- `pyproject.toml` debe reflejar la versión del último tag.

## Flujo de trabajo

1. Issues ligadas a un milestone del GitHub Project **Entrega 1 (SoupDB)**.
2. Rama `feat|fix|bench|docs/<tema>` → PR a `dev` (CI en verde).
3. Publicación: PR `dev -> main` y tag inmediato en ese merge.

## Mapa de tags

| Tag | Fecha | Entrega | PR de anclaje | Cambio clave |
|---|---|---|---|---|
| `v0.1.0` | 2026-09-27 | **Avance 1** (BD relacional) | #67 | DiskManager, BufferManager, HeapFile, SequentialFile, ExtendibleHash, parser SQL, transacciones 2PL, protocolo v1 + driver rsoup |
| `v0.2.0` | 2026-10-04 | **Entrega Parcial** (extensión espacial) | #104 | R-Tree persistente, consultas espaciales, gestor ejecutable, plan EXPLAIN |
| `v0.3.0` | 2026-10-25 | **Avance 3** (índices vectoriales) | #136 | RankedIndex, IVF/HNSW persistentes, multimedia SIFT/MFCC, benches con vectores reales |
| `v0.4.0` | 2026-11-02 | **SQL de texto y multimedia** | #147 / #149 | MATCH/SIMILAR_TO, SCORE/SIMILARITY_SCORE, índices rankeados transaccionales y revertibles, benchmarks consolidados |
| `v0.5.0` | 2026-11-06 | **Índice invertido + DML en índices** | #152 | SPIMI/TF-IDF/BM25 reales, mantenimiento por INSERT/DELETE con tombstones, catálogo de plugins y autocarga |
| `v0.6.0` | 2026-11-06 | **Plugins y protocolo v2** | #152 | `SEARCH(...) USING <plugin>`, `SCORE()` único, DDL genérico, protocolo v2 con tipos por nombre y prepared statements, Docker con instalación en caliente |
| `v0.7.0` | 2026-11-06 | **`CREATE EXTENSION`** | pendiente | Activación de plugins por SQL (persistente en `SysPlugins`), `DROP EXTENSION` con RESTRICT |

Histórico de rescates:

- El benchmark huérfano `benchmarks/scripts/compare_storage.py` (PR #38, cerrado sin fusionar) fue rescatado vía PR #148 y vive en `dev` desde entonces.

## Próxima versión

- **v0.8.0**: cierre de *Avance 3* — benchmarks de texto contra PostgreSQL GIN (10k/100k) y validación independiente de recall/scores con datos reales (issues #125–#130).

## Cómo publicar una entrega

1. `uv run --group multimedia pytest` y `uv run ruff check .` en verde.
2. PR `dev -> main` de publicación.
3. `git tag -a vX.Y.Z <merge> -m "..."` y `git push origin vX.Y.Z`.

## Frontend asociado

El frontend **SoupChef** mantiene tags paralelos `vX.Y.0` por cada entrega con UI asociada (ver su `docs/versionado.md`). La entrega de un milestone debe taggear ambos repos.
