"""Render de expresiones del AST como SQL legible.

El panel de plan de ejecución muestra el detalle de cada operador. Sin esto el
predicado aparece como el ``repr`` del dataclass —
``CompareExpr(left=ColumnRef(name='nota'), op='>=', right=Literal(value=14))`` —
que es ilegible para quien mira la consulta, no el código.
"""

from engine.query.ast import (
    BetweenExpr,
    BinaryExpr,
    ColumnRef,
    CompareExpr,
    DistanceExpr,
    Expr,
    FunctionExpr,
    InExpr,
    IntersectsExpr,
    IsNullExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    PointExpr,
    PolygonExpr,
)


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def format_expr(expr: Expr) -> str:
    """Devuelve la expresión como se escribiría en SQL."""
    if isinstance(expr, Literal):
        return _literal(expr.value)
    if isinstance(expr, ColumnRef):
        return expr.name
    if isinstance(expr, (CompareExpr, BinaryExpr)):
        return f"{format_expr(expr.left)} {expr.op} {format_expr(expr.right)}"
    if isinstance(expr, LogicalExpr):
        # Los operandos lógicos se parentizan: sin eso `a AND (b OR c)` y
        # `(a AND b) OR c` se verían igual y dirían cosas distintas.
        return f"({format_expr(expr.left)} {expr.op} {format_expr(expr.right)})"
    if isinstance(expr, NotExpr):
        return f"NOT {format_expr(expr.operand)}"
    if isinstance(expr, IsNullExpr):
        return f"{format_expr(expr.value)} IS {'NOT ' if expr.negated else ''}NULL"
    if isinstance(expr, BetweenExpr):
        value = format_expr(expr.value)
        return f"{value} BETWEEN {format_expr(expr.lo)} AND {format_expr(expr.hi)}"
    if isinstance(expr, InExpr):
        items = ", ".join(format_expr(item) for item in expr.items)
        return f"{format_expr(expr.value)} IN ({items})"
    if isinstance(expr, LikeExpr):
        return f"{format_expr(expr.value)} LIKE {format_expr(expr.pattern)}"
    if isinstance(expr, FunctionExpr):
        distinct = "DISTINCT " if expr.distinct else ""
        argument = format_expr(expr.arg) if expr.arg is not None else "*"
        return f"{expr.name}({distinct}{argument})"
    if isinstance(expr, PointExpr):
        # Se muestra como se escribe en SQL: latitud y después longitud.
        return f"POINT({format_expr(expr.y)}, {format_expr(expr.x)})"
    if isinstance(expr, DistanceExpr):
        metric = f", '{expr.metric}'" if expr.metric else ""
        return f"distancia({format_expr(expr.left)}, {format_expr(expr.right)}{metric})"
    if isinstance(expr, PolygonExpr):
        vertices = ", ".join(format_expr(vertex) for vertex in expr.vertices)
        return f"POLYGON({vertices})"
    if isinstance(expr, IntersectsExpr):
        return f"intersects({format_expr(expr.left)}, {format_expr(expr.right)})"
    # Un nodo nuevo no debe romper el plan: se degrada al repr en vez de fallar.
    return str(expr)
