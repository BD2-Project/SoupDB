"""Caché en disco de descriptores: un ``.npy`` por objeto multimedia.

objeto -> clave -> ¿existe el archivo? -> sí: cargarlo / no: extraer, guardar y devolver.

La clave es el SHA-256 de (tipo, ruta, parámetros de extracción): nombres largos o
con caracteres raros no dan problemas y cambiar un parámetro da otra entrada.
Reanudar una ejecución interrumpida sale gratis: lo ya guardado se lee de aquí.
"""

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np


def cache_path(
    cache_dir: str | Path, kind: str, source: str | Path, params: Mapping[str, object]
) -> Path:
    # Ruta absoluta y con "/" para que la misma imagen dé la misma clave desde cualquier cwd.
    identity = {"kind": kind, "source": Path(source).resolve().as_posix(), "params": dict(params)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    # Subcarpeta por los dos primeros caracteres: con 100k objetos no hay una carpeta enorme.
    return Path(cache_dir) / kind / digest[:2] / f"{digest}.npy"


def save_array(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Se escribe a un temporal y se renombra: si se corta a mitad, no queda un .npy
    # a medias que una ejecución posterior tomaría por bueno.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def load_array(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def cached(path: str | Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    array = load_array(path)
    if array is None:
        array = compute()
        save_array(path, array)
    return array
