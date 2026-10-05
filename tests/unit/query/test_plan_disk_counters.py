"""El plan reporta los accesos a disco reales.

El costo en accesos es la métrica que evalúa el curso, porque no depende de la
máquina. Si el `PlanNode` reporta cero en todos los operadores, el panel de plan
de ejecución no dice nada útil.
"""

from pathlib import Path

from engine.common.catalog import Catalog
from engine.query import execute_sql
from engine.query.operators import PlanNode
from engine.query.parser import parse
from engine.query.planner import plan

# Pocas páginas en memoria: con el buffer pool por defecto una tabla chica entra
# entera y no habría ninguna lectura física que contar.
BUFFER = 4
FILAS = 400


def _poblar(tmp_path: Path) -> Catalog:
    catalog = Catalog(tmp_path, buffer_capacity=BUFFER)
    execute_sql("CREATE TABLE alumnos (id INT, nombre VARCHAR(100), nota INT)", catalog)
    for i in range(1, FILAS + 1):
        execute_sql(
            f"INSERT INTO alumnos VALUES ({i}, 'alumno {i} con texto de relleno', {i % 21})",
            catalog,
        )
    return catalog


def _nodos(node: PlanNode) -> list[PlanNode]:
    return [node, *(descendant for child in node.children for descendant in _nodos(child))]


def test_catalog_exposes_aggregated_counters(tmp_path: Path) -> None:
    catalog = _poblar(tmp_path)
    assert catalog.disk_manager.reads > 0
    assert catalog.disk_manager.writes > 0
    catalog.close()


def test_a_scan_reports_physical_reads(tmp_path: Path) -> None:
    catalog = _poblar(tmp_path)
    root = plan(parse("SELECT * FROM alumnos WHERE nota >= 14"), catalog).root
    root.open()
    while root.next() is not None:
        pass
    node = root.explain()
    root.close()

    scan = next(item for item in _nodos(node) if item.op == "TableScan")
    assert scan.disk_reads > 0, "un recorrido completo debe leer páginas de disco"
    catalog.close()


def test_counters_are_cumulative_towards_the_root(tmp_path: Path) -> None:
    """Un operador incluye la E/S de sus hijos, como hace PostgreSQL."""
    catalog = _poblar(tmp_path)
    root = plan(parse("SELECT * FROM alumnos WHERE nota >= 14 ORDER BY id"), catalog).root
    root.open()
    while root.next() is not None:
        pass
    node = root.explain()
    root.close()

    for item in _nodos(node):
        hijos = max((child.disk_reads for child in item.children), default=0)
        assert item.disk_reads >= hijos
    catalog.close()
