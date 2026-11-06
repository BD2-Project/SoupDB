# Plugins

SoupDB separa las extensiones del núcleo. La imagen base del gestor solo
incluye el core; full-text y vectorial se instalan como paquetes Python.

## Paquetes

| Paquete | Contenido | Dependencias |
|---|---|---|
| `soupdb` | core: tipos base, storage, B+/hash/R-tree, SQL, transacciones | ninguna de terceros |
| `soupdb-inverted` | índice invertido SPIMI y búsqueda TF-IDF/BM25 | `soupdb` |
| `soupdb-vector` | índices IVF/HNSW y features SIFT/MFCC | `soupdb`, numpy, opencv, librosa, scikit-learn |

Cada plugin declara su entrada en el grupo `soupdb.plugins`:

```toml
[project.entry-points."soupdb.plugins"]
inverted = "soupdb_inverted.plugin:InvertedPlugin"
```

El core los descubre con `PluginRegistry.discover()` sin importar su
implementación.

## Activación

El plugin debe estar instalado y activado:

```sql
CREATE EXTENSION inverted WITH (storage_root = '/app/data/plugins/inverted');
```

La configuración queda en `SysPlugins` y el motor la reaplica al abrir la base.
`DROP EXTENSION inverted` lo desactiva (falla si hay índices dependientes).

Para el plugin vectorial, `profiles` se puede pasar como **JSON** en el `WITH`
(desde `soupdb-vector 0.1.1`):

```sql
CREATE EXTENSION ivf WITH (
  storage_root = '/app/data/plugins/ivf',
  profiles = '{"img-v1": {"kind": "image", "root": "/datasets/img", "codebook": "/codebooks/img.npz"}}'
);
```

Cada plugin vectorial configurado con perfiles distintos implica vectorizadores
distintos; el motor solo lo rechaza si realmente hay ambigüedad (varios plugins
configurados con configuraciones incompatibles).

## Instalación local

```bash
uv sync --group multimedia        # entorno de desarrollo con ambos plugins
uv pip install -e plugins/soupdb-inverted
uv pip install -e plugins/soupdb-vector
```

## Docker sin reconstruir la imagen

La imagen base arranca solo con el core. Para añadir un plugin:

```bash
# 1. Construir el wheel del plugin
uv build --package soupdb-inverted --out-dir dist/plugins

# 2. Montarlo en el contenedor y activar la instalación en caliente
docker run -p 55432:55432 \
  -v "$PWD/dist/plugins:/opt/soupdb/plugins:ro" \
  -e SOUPDB_PLUGINS=1 \
  soupdb:core
```

El entrypoint instala los `.whl` en `/opt/soupdb/plugin-site` y añade ese
directorio a `PYTHONPATH` antes de arrancar `python -m engine`. No se
reconstruye la imagen; basta reiniciar el contenedor con el wheel nuevo.

## Compatibilidad

Durante una versión se mantienen los módulos antiguos
(`engine.indexes.inverted.*`, `engine.indexes.vector.*`,
`engine.multimedia.*`) como shims que reexportan el plugin instalado. Si el
plugin no está instalado, importar esas rutas falla con `ModuleNotFoundError`.
