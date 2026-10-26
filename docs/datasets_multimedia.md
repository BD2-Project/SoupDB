# Datasets multimedia reales

Datasets de 1.000, 10.000 y 100.000 objetos para los índices vectoriales (#118, #119, #120).
Ningún archivo multimedia ni generado se versiona: todo vive fuera del repo o en
`benchmarks/data/`, que está en `.gitignore`.

| Medio  | Fuente           | Objeto                 | Disponible                     |
|--------|------------------|------------------------|--------------------------------|
| Imagen | COCO 2017 train  | una imagen             | 118.287 imágenes (~18 GB zip)  |
| Audio  | FMA Medium       | un segmento de 5 s     | 25.000 pistas de 30 s (~22 GiB zip), ~150.000 segmentos |

## Parámetros fijos

| Parámetro         | Valor      | Dónde se fija                                    |
|-------------------|------------|--------------------------------------------------|
| K (palabras)      | 128        | `benchmarks/scripts/build_multimedia_vectors.py` |
| `max_descriptors` | 50.000     | ídem                                             |
| seed              | 20261024   | ídem                                             |
| lado máx. imagen  | 1024 px    | `MAX_IMAGE_SIDE` en `image_features.py`          |
| segmento de audio | 5 s, sin solapamiento | `SEGMENT_SECONDS` en `audio_features.py` |
| audio             | mono, 22050 Hz, 20 MFCC | `audio_features.py`                 |

Cada salida guarda además un `params.json` con estos valores.

## 1. Descargar y extraer (fuera del repo)

Los ejemplos usan `D:/datasets`; cualquier carpeta sirve.

**COCO 2017 train**

```powershell
curl.exe -L -o D:/datasets/coco/train2017.zip http://images.cocodataset.org/zips/train2017.zip
tar -xf D:/datasets/coco/train2017.zip -C D:/datasets/coco    # crea D:/datasets/coco/train2017/
```

**FMA Medium** (https://github.com/mdeff/fma)

```powershell
curl.exe -L -o D:/datasets/fma_medium.zip https://os.unil.cloud.switch.ch/fma/fma_medium.zip
(Get-FileHash D:/datasets/fma_medium.zip -Algorithm SHA1).Hash
# debe ser C67B69EA232021025FCA9231FC1C7C1A063AB50B (README oficial de FMA)
7z x D:/datasets/fma_medium.zip -oD:/datasets                  # crea D:/datasets/fma_medium/000/...
```

En bash: `sha1sum fma_medium.zip` y `unzip fma_medium.zip`. El README de FMA recomienda
7-Zip porque algunos `unzip` fallan con archivos de más de 4 GB.

## 2. Manifests anidados

```powershell
uv run --group multimedia python -m benchmarks.scripts.prepare_multimedia_datasets `
    --coco D:/datasets/coco/train2017 --fma D:/datasets/fma_medium
```

Genera en `benchmarks/data/multimedia/manifests/`:
`images_1k.jsonl`, `images_10k.jsonl`, `images_100k.jsonl`,
`audio_1k.jsonl`, `audio_10k.jsonl`, `audio_100k.jsonl`.

- Los archivos se ordenan por ruta y se numeran con `engine.multimedia.manifest`.
- Se escribe el de 100k y los otros dos se cortan de él: `1k ⊂ 10k ⊂ 100k` línea a línea.
- COCO se cuenta antes de empezar. FMA solo se puede contar decodificando, así que se
  decodifican pistas hasta reunir 100.000 segmentos (~16.700 pistas).
- Los MP3 que no se pueden decodificar se saltan con un aviso. Mismos archivos, mismas
  pistas saltadas, mismos IDs.

## 3. Vectores

```powershell
uv run --group multimedia python -m benchmarks.scripts.build_multimedia_vectors `
    image --root D:/datasets/coco/train2017 --sizes 1000 10000 100000
uv run --group multimedia python -m benchmarks.scripts.build_multimedia_vectors `
    audio --root D:/datasets/fma_medium --sizes 1000 10000 100000
```

```
benchmarks/data/multimedia/
├── manifests/            images_1k.jsonl … audio_100k.jsonl
├── cache/                image/…/*.npy, audio_segment/…/*.npy
├── images/{1000,10000,100000}/codebook.npz, vectores.bin, params.json
└── audio/{1000,10000,100000}/codebook.npz, vectores.bin, params.json
```

- Cada tamaño entrena su propio codebook y escribe su propio `vectores.bin`.
- La caché es compartida por los tres tamaños. Al pasar de 1k a 10k solo se extraen
  los 9.000 objetos nuevos.
- **Reanudar:** si se corta, se repite el mismo comando. Los SIFT/MFCC ya guardados se
  leen de la caché. Un `vectores.bin` a medias se borra solo.
- La caché se indexa por la ruta absoluta del archivo fuente. Si se mueve el dataset,
  se recalcula todo.
- Tras escribir cada `vectores.bin` se valida contra su manifest: número de registros,
  IDs únicos y esperados, paths, dimensión K y ausencia de NaN/infinito.

## Almacenamiento aproximado (sin contar los datasets fuente)

| Artefacto                       | 1k      | 10k    | 100k   |
|---------------------------------|---------|--------|--------|
| caché SIFT (float32, 128 × 4 B) | ~0,5 GB | ~5 GB  | ~50 GB |
| caché MFCC (216 × 20 × 4 B)     | 17 MB   | 170 MB | 1,7 GB |
| `vectores.bin` (cada medio)     | 0,55 MB | 5,5 MB | 55 MB  |
| codebooks, manifests            | < 1 MB  | ~2 MB  | ~17 MB |

La caché SIFT supone unos 1.000 keypoints por imagen de COCO y es la parte dominante.
La caché es acumulativa: 100k incluye a 10k y a 1k.
