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

Histórico de rescates:

- El benchmark huérfano `benchmarks/scripts/compare_storage.py` (PR #38, cerrado sin fusionar) fue rescatado vía PR #148 y vive en `dev` desde entonces.

## Próxima versión

- **v0.5.0**: cierre de *Avance 3* — índice invertido (tokenizador + SPIMI, TF-IDF y BM25, postings y diccionario en disco), corpus de texto de 10k/100k y benchmarks contra PostgreSQL GIN (issues #110–#114, #125–#130). Al fusionar esa publicación se cierra el milestone y se crea el tag.

## Frontend asociado

El frontend **SoupChef** mantiene tags paralelos `vX.Y.0` por cada entrega con UI asociada (ver su `docs/versionado.md`). La entrega de un milestone debe taggear ambos repos.
