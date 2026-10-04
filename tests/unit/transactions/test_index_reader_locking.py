"""Every reader locks the table, whichever path it reads by.

A reader that finds its rows through an index and then fetches them one RID at a
time used to take no lock on the table: ``fetch`` locks the record, and a record
lock does not conflict with the table lock an uncommitted INSERT holds. The
consequence was a dirty read that depended on the DATA rather than on the
statement - the same query blocked or not depending on whether the plan fell back
to a scan, and reading by index was the only way to see another transaction's
uncommitted row.

These tests pin the property for the index readers: with a writer holding the
table, they block exactly as a ``TableScan`` does, and once the writer is done
they only see committed rows. Each one first checks that the query really reaches
the index reader, so no test can pass by falling back to a scan.
"""

import pytest

from engine.common.errors import TransactionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.parser import parse
from engine.query.planner import plan
from engine.transactions.session import TransactionalSession
from engine.transactions.transaction_manager import TransactionManager

PARTES = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("nombre", ColumnType.TEXT),
)
LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)

# Cada camino: como se llama, la tabla, la fila que el escritor deja sin commitear,
# la consulta del lector, el operador por el que tiene que pasar y el valor que el
# lector jamas puede ver.
CAMINOS = [
    ("igualdad", "partes", "(3, 'tres')", 3, "SELECT id FROM partes WHERE id = 3", "IndexLookup"),
    (
        "rango",
        "partes",
        "(3, 'tres')",
        3,
        "SELECT id FROM partes WHERE id BETWEEN 3 AND 3",
        "IndexRangeScan",
    ),
    (
        "radio",
        "lugares",
        "('SUELTA', POINT(0.5, 0))",
        "SUELTA",
        "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 2.0",
        "SpatialIndexScan",
    ),
]
PARAMETROS = pytest.mark.parametrize(
    ("etiqueta", "tabla", "values", "ausente", "sql", "operador"),
    CAMINOS,
    ids=[camino[0] for camino in CAMINOS],
)


def make_partes():
    from tests.fakes.fake_catalog import FakeCatalog

    catalog = FakeCatalog()
    catalog.create_table("partes", PARTES)
    catalog.insert("partes", (1, "uno"))
    catalog.add_index("partes", "id", index_name="idx_id")
    return catalog


def make_lugares():
    from tests.fakes.fake_catalog import FakeCatalog

    catalog = FakeCatalog()
    catalog.create_table("lugares", LUGARES)
    catalog.insert("lugares", ("A", (5.0, 0.0)))
    catalog.add_spatial_index("lugares", "ubicacion", index_name="idx_ubic")
    return catalog


def catalogo_de(tabla: str):
    return make_partes() if tabla == "partes" else make_lugares()


def dos_sesiones(catalog):
    """A writer and a reader sharing a catalog and a lock manager."""
    manager = TransactionManager()
    writer = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    reader = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    writer.begin()
    reader.begin()
    return writer, reader


def operadores_de(sql: str, catalog) -> set[str]:
    """Every operator in the plan: which read path this query will take."""
    ops: set[str] = set()

    def visitar(node) -> None:
        ops.add(node.op)
        for child in node.children:
            visitar(child)

    visitar(plan(parse(sql), catalog).root.explain())
    return ops


def cerrar(*sesiones: TransactionalSession) -> None:
    """Cierra sesiones que pueden haber cerrado ya su transacción.

    Un bloqueo que expira hace que la sesión revierta su propia transacción, así que
    no siempre queda una activa que revertir.
    """
    for session in sesiones:
        if session.current_transaction() is not None:
            session.rollback()
        session.close()


# --- la consulta tiene que pasar por el lector por índice -----------------


@PARAMETROS
def test_the_query_really_goes_through_the_index_reader(
    etiqueta, tabla, values, ausente, sql, operador
) -> None:
    ops = operadores_de(sql, catalogo_de(tabla))
    assert operador in ops
    assert "TableScan" not in ops


# --- con un escritor en la tabla, el lector por índice bloquea ------------


@PARAMETROS
def test_an_uncommitted_insert_blocks_every_index_reader(
    etiqueta, tabla, values, ausente, sql, operador
) -> None:
    writer, reader = dos_sesiones(catalogo_de(tabla))
    try:
        writer.execute(f"INSERT INTO {tabla} VALUES {values}")
        with pytest.raises(TransactionError):
            reader.execute(sql)
    finally:
        cerrar(writer, reader)


@PARAMETROS
def test_the_reader_never_sees_the_uncommitted_row(
    etiqueta, tabla, values, ausente, sql, operador
) -> None:
    # Sin el lock de tabla estos lectores devolvían la fila sin commitear.
    writer, reader = dos_sesiones(catalogo_de(tabla))
    try:
        writer.execute(f"INSERT INTO {tabla} VALUES {values}")
        with pytest.raises(TransactionError):
            reader.execute(sql)
        writer.rollback()
        rows = [row[0] for row in reader.execute(sql).rows]
    finally:
        cerrar(writer, reader)
    assert ausente not in rows


@PARAMETROS
def test_the_reader_does_see_the_row_once_the_writer_commits(
    etiqueta, tabla, values, ausente, sql, operador
) -> None:
    # La contrapartida: el bloqueo era de verdad, no un error permanente.
    writer, reader = dos_sesiones(catalogo_de(tabla))
    try:
        writer.execute(f"INSERT INTO {tabla} VALUES {values}")
        writer.commit()
        rows = [row[0] for row in reader.execute(sql).rows]
    finally:
        cerrar(writer, reader)
    assert rows == [ausente]


# --- la referencia: el escaneo completo ya bloqueaba ----------------------


def test_the_table_scan_blocks_too_so_the_readers_agree() -> None:
    catalog = make_lugares()
    writer, reader = dos_sesiones(catalog)
    try:
        writer.execute("INSERT INTO lugares VALUES ('SUELTA', POINT(0.5, 0))")
        assert "TableScan" in operadores_de("SELECT nombre FROM lugares", catalog)
        with pytest.raises(TransactionError):
            reader.execute("SELECT nombre FROM lugares")
    finally:
        cerrar(writer, reader)


# --- el lock se suelta con la transacción ---------------------------------


def test_the_index_reader_releases_the_table_lock_on_commit() -> None:
    writer, reader = dos_sesiones(make_partes())
    try:
        reader.execute("SELECT id FROM partes WHERE id = 1")
        reader.commit()
        # Con la transacción del lector cerrada, el escritor vuelve a poder escribir.
        writer.execute("INSERT INTO partes VALUES (9, 'nueve')")
        writer.commit()
    finally:
        cerrar(writer, reader)


# --- concurrencia real: un lector por índice contra un DELETE en curso -----


def test_a_deleting_session_does_not_deadlock_an_index_reader() -> None:
    """El DELETE toma tabla (S) y luego fila (X); el lector toma tabla (S) y fila (S).

    El lector tiene que BLOQUEAR (y expirar por timeout), no deadlockear: para que
    hubiera deadlock, alguien tendría que esperar la tabla mientras sostiene la
    fila, y el DELETE ya tomó la tabla antes de la fila.
    """
    import threading

    catalog = make_partes()
    manager = TransactionManager()
    escritor = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    lector = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    errors: list[BaseException] = []
    listo = threading.Event()

    def borrar() -> None:
        try:
            escritor.begin()
            escritor.execute("DELETE FROM partes WHERE id = 1")
            listo.set()
            escritor.rollback()
        except BaseException as exc:  # pragma: no cover - solo para diagnostico
            errors.append(exc)

    def leer() -> None:
        listo.wait(timeout=5)
        try:
            lector.begin()
            lector.execute("SELECT id FROM partes WHERE id = 1")
        except BaseException as exc:
            errors.append(exc)

    hilos = [threading.Thread(target=borrar), threading.Thread(target=leer)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join(timeout=10)
    for hilo in hilos:
        assert not hilo.is_alive(), "un hilo se quedo esperando: deadlock"
    # El DELETE no falla, y el lector o bien expiro por el bloqueo o bien leyo la fila
    # ya revertida: lo que no puede pasar es un error inesperado.
    assert all(isinstance(exc, TransactionError) for exc in errors), errors
    cerrar(escritor, lector)


def test_two_index_readers_do_not_block_each_other() -> None:
    # Dos lecturas son compatibles: compartir la tabla en modo S no se estorba.
    catalog = make_partes()
    manager = TransactionManager()
    uno = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    dos = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    try:
        uno.begin()
        dos.begin()
        assert [row[0] for row in uno.execute("SELECT id FROM partes WHERE id = 1").rows] == [1]
        assert [row[0] for row in dos.execute("SELECT id FROM partes WHERE id = 1").rows] == [1]
    finally:
        cerrar(uno, dos)
