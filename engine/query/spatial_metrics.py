"""Métricas de distancia espacial: euclidiana y haversine.

Módulo puro y sin estado: sin E/S, sin dependencias del motor y con la misma
convención que la capa de consulta, de modo que el baseline secuencial de
``benchmarks/spatial_baseline.py`` pueda reutilizar estas funciones como oráculo
y ambos caminos coincidan exactamente.

Convención de coordenadas
-------------------------
Un punto es un par ``(x, y)`` interpreted en grados geográficos:

* ``x`` = longitud (lon) en el rango ``[-180, 180]``;
* ``y`` = latitud (lat) en el rango ``[-90, 90]``.

La convención se mantiene en las dos métricas: ``x`` es siempre el eje
horizontal (longitud) y ``y`` el vertical (latitud). Lo único que cambia es la
unidad del resultado, porque son métricas de Naturaleza distinta:

* ``euclidean``: distancia en el plano, en las unidades de la coordenada (grados
  si el punto se lee como par lon/lat). Es el plano cartesiano tal cual, con
  ``hypot``, sin ninguna conversión.
* ``haversine``: distancia geodésica sobre la esfera en metros, calculada con
  la fórmula del haversine y el radio terrestre medio :data:`EARTH_RADIUS_M`.

Coherencia del radio terrestre: :data:`EARTH_RADIUS_M` es el único factor de
escala y se aplica una sola vez, en :func:`haversine_distance`, tanto si el
primer o el segundo operando es el punto de referencia (la distancia es
simétrica). :func:`euclidean_distance` no lo multiplica porque su unidad es la
del plano: escalar la métrica euclidiana por el radio terrestre rompería los
radios ya expresados en grados y no describe ninguna distancia real. Por eso los
radios de ``WHERE distance(...) < r`` se interpretan siempre en la unidad de la
métrica elegida (grados para ``euclidean``, metros para ``haversine``).

Coordenadas inválidas
---------------------
El módulo **rechaza** en lugar de devolver ``nan`` o un valor sin sentido:

* coordenadas no finitos (``nan``, ``inf``) se rechazan en ambas métricas;
* fuera de rango geográfico (lon fuera de ``[-180, 180]`` o lat fuera de
  ``[-90, 90]``) se rechaza solo en ``haversine``, donde esos valores no
  describen un punto de la superficie terrestre y el resultado sería arbitrario.
  ``euclidean`` es una métrica de plano y por eso acepta cualquier par finito.

No se normalizan las coordenadas: ``lon = 190`` es un error y no ``-170``. Es lo
mismo que hace el resto del motor ante un valor inválido (fallar temprano y con
un mensaje claro) en vez de devolver silenciosamente un número engañoso.

Errores
-------
El módulo es puro y no importa la jerarquía de errores del motor: viola el
contrato lanzando :class:`ValueError` con un mensaje que incluye el valor
rechazado. Quien lo usa decide el tipo final: ``engine.query.evaluator`` lo
convierte en :class:`~engine.common.errors.QueryExecutionError` y
``engine.query.parser`` en :class:`~engine.common.errors.QueryParseError`, de modo
que un mismo fallo se reporte según la capa donde se detecta.
"""

import math
from collections.abc import Callable, Sequence

#: Radio terrestre medio en metros (IUGG). Escala única de la métrica geodésica:
#: 1 grado de latitud sobre un meridiano son ~111 195 m y un par antípodal
#: exactamente ``pi * EARTH_RADIUS_M``. En metros y no en kilómetros porque el
#: enunciado (2.2.3) escribe los radios así: ``distancia(...) < 5000`` para 5 km.
EARTH_RADIUS_M = 6_371_008.8

#: Nombre canónico de la métrica euclidiana (plano, en unidades de la coordenada).
EUCLIDEAN = "euclidean"
#: Nombre canónico de la métrica geodésica (haversine, en metros).
HAVERSINE = "haversine"

#: Métricas soportadas por ``distance(a, b[, 'metrica'])``.
METRIC_NAMES: tuple[str, ...] = (EUCLIDEAN, HAVERSINE)

Point2D = Sequence[float]
DistanceFn = Callable[[Point2D, Point2D], float]


def euclidean_distance(a: Point2D, b: Point2D) -> float:
    """Distancia euclidiana en el plano entre ``a`` y ``b``.

    ``a`` y ``b`` son pares ``(x, y)`` en las unidades de la coordenada (grados
    si el punto se lee como par lon/lat). Es ``hypot(dx, dy)``: independiente del
    orden de los operandos y cero solo cuando los puntos coinciden.
    """
    x1, y1 = _coordinates(a)
    x2, y2 = _coordinates(b)
    return math.hypot(x1 - x2, y1 - y2)


def haversine_distance(a: Point2D, b: Point2D) -> float:
    """Distancia geodésica en kilómetros entre dos puntos lon/lat.

    Aplica la fórmula del haversine sobre la esfera de radio
    :data:`EARTH_RADIUS_M`::

        d = 2 * R * asin(sqrt(sin²(Δφ/2) + cos(φ₁) * cos(φ₂) * sin²(Δλ/2)))

    con las latitudes en radianes. Es simétrica, da ``0.0`` para el mismo punto y
    ``pi * R`` para un par antípodal; además, como ``lon`` se resta antes de
    pasar a radianes, el salto de la antimeridiana (179° vs -179°) es el corto.

    Rechaza coordenadas no finitas y fuera de rango geográfico.
    """
    lon1, lat1 = _geographic_coordinates(a)
    lon2, lat2 = _geographic_coordinates(b)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    sin_dphi = math.sin((phi2 - phi1) / 2)
    sin_dlambda = math.sin(math.radians(lon2 - lon1) / 2)
    h = sin_dphi**2 + math.cos(phi1) * math.cos(phi2) * sin_dlambda**2
    # El redondeo en coma flotante puede dejar h en (0, 1] + epsilon; sin este
    # recorte asin(>1) lanzaría ValueError en el par antípodal.
    h = min(1.0, max(0.0, h))
    # El factor se aplica al arco (2 * asin) y no al radio para que el par
    # antípodal dé exactamente ``pi * EARTH_RADIUS_M`` y el borde se pueda
    # comparar con ese valor sin sorpresas de redondeo.
    return EARTH_RADIUS_M * (2.0 * math.asin(math.sqrt(h)))


#: Registro de métricas por nombre canónico, para despacho por ``distance``.
METRIC_FUNCTIONS: dict[str, DistanceFn] = {
    EUCLIDEAN: euclidean_distance,
    HAVERSINE: haversine_distance,
}


def normalize_metric(name: str) -> str:
    """Nombre canónico (minúsculas) de una métrica, o ``ValueError`` si no existe.

    No distingue mayúsculas ni tolera espacios: ``'Haversine'``, ``'HAVersine'``
    y ``'haversine'`` son la misma métrica. Lanza :class:`ValueError` con los
    nombres válidos para que cada capa la reporte con su propio tipo de error.
    """
    canonical = name.strip().lower()
    if canonical not in METRIC_FUNCTIONS:
        raise ValueError(
            f"unknown distance metric {name!r}, expected one of: "
            + ", ".join(repr(metric) for metric in METRIC_NAMES)
        )
    return canonical


def distance(a: Point2D, b: Point2D, metric: str = EUCLIDEAN) -> float:
    """Distancia entre ``a`` y ``b`` según ``metric``.

    Es la implementación de ``distance(a, b)`` /
    ``distance(a, b, 'metrica')``: el nombre se normaliza con
    :func:`normalize_metric`, así que un ``'Haversine'`` escrito a mano o con
    espacios invade el mismo camino que el literal del parser.
    """
    return METRIC_FUNCTIONS[normalize_metric(metric)](a, b)


def _coordinates(point: Point2D) -> tuple[float, float]:
    """Valida que el punto sea un par finito y devuelve ``(x, y)``."""
    x, y = _pair(point)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"distance() requires finite coordinates, got ({x}, {y})")
    return x, y


def _geographic_coordinates(point: Point2D) -> tuple[float, float]:
    """Además de finito, exige que el punto sea un par lon/lat en rango."""
    x, y = _coordinates(point)
    if not -180.0 <= x <= 180.0:
        raise ValueError(f"haversine() requires longitude in [-180, 180], got {x}")
    if not -90.0 <= y <= 90.0:
        raise ValueError(f"haversine() requires latitude in [-90, 90], got {y}")
    return x, y


def _pair(point: Point2D) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"distance() requires 2D points, got {len(point)} coordinates")
    x, y = point[0], point[1]
    if type(x) is not float and type(x) is not int:
        raise ValueError(f"distance() requires numeric coordinates, got {type(x).__name__}")
    if type(y) is not float and type(y) is not int:
        raise ValueError(f"distance() requires numeric coordinates, got {type(y).__name__}")
    return float(x), float(y)
