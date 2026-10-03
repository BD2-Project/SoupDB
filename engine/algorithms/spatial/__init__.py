from .bounds import euclidean_min_distance, haversine_latitude_lower_bound
from .geometry import Polygon2D
from .knn import SpatialHit, knn_hits, knn_search
from .metrics import (
    EARTH_RADIUS_M,
    SpatialMetric,
    euclidean_distance,
    euclidean_metric,
    haversine_distance,
    haversine_metric,
)
from .polygon_query import iter_polygon_entries, polygon_search
from .predicates import point_in_polygon, point_on_segment
from .validation import (
    point_from_latlon,
    validate_cartesian_point,
    validate_earth_radius,
    validate_geographic_point,
)

__all__ = [
    "EARTH_RADIUS_M",
    "SpatialMetric",
    "euclidean_distance",
    "euclidean_metric",
    "euclidean_min_distance",
    "haversine_distance",
    "haversine_latitude_lower_bound",
    "haversine_metric",
    "point_from_latlon",
    "validate_cartesian_point",
    "validate_earth_radius",
    "validate_geographic_point",
    "Polygon2D",
    "point_in_polygon",
    "point_on_segment",
    "SpatialHit",
    "knn_hits",
    "knn_search",
    "iter_polygon_entries",
    "polygon_search",
]
