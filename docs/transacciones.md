# Transacciones y concurrencia

Dominio que coordina el acceso simultáneo de múltiples usuarios a la base de datos de forma segura, implementando **transacciones** (`BEGIN`/`END TRANSACTION`), **control de concurrencia pesimista** (strict 2PL con locks S/U/X) y una **demostración con hilos**.

## Diseño: estrategia desacoplada

El control de concurrencia está desacoplado detrás de un contrato (`engine/transactions/base.py`) para poder **intercambiar estrategias** sin tocar a los llamadores.

```python
class ConcurrencyStrategy(ABC):
    def acquire(self, tx_id, resource, mode, timeout_ms=None) -> None: ...
    def release(self, tx_id, resource) -> None: ...
    def release_all(self, tx_id) -> None: ...
    def locks_held(self, tx_id) -> frozenset[Resource]: ...
    def is_locked(self, resource) -> bool: ...
    def close(self) -> None: ...
```

- Implementación actual: **`StrictTwoPhaseLocking`** (pesimista).
- Futuras: optimista, MVCC, ordenamiento por timestamps — basta con implementar el ABC e inyectarlo en `LockManager`.

## Contratos

| Concepto | Definición |
|---|---|
| `LockMode` | `SHARED` (S, lectura), `UPDATE` (U, intermedio), `EXCLUSIVE` (X, escritura) |
| `TransactionState` | `active`, `partially_committed`, `committed`, `failed`, `aborted` |
| Granularidad | Tabla `("table", name)` y registro `("record", RID)` |

### Matriz de compatibilidad (strict 2PL)

| Solicitado \ Sostenido | S | U | X |
|---|---|---|---|
| **S** | ✓ | ✓ | ✗ |
| **U** | ✓ | ✗ | ✗ |
| **X** | ✗ | ✗ | ✗ |

- Los locks **se mantienen hasta el `COMMIT`/`ROLLBACK`** (estricto), lo que garantiza aislamiento y planificaciones serializables.
- Re-adquisición de la misma transacción: *upgrade* (p. ej. S → X) sin duplicar el recurso; se cuenta la referencia.

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `base.py` | Contrato `ConcurrencyStrategy`, `LockMode`, `TransactionState` |
| `strategies/strict_2pl.py` | Estrategia pesimista strict 2PL |
| `lock_manager.py` | Facade que delega en la estrategia inyectada |
| `transaction_manager.py` | Ciclo de vida ACID + **undo journal** para atomicidad |
| `session.py` | `TransactionalSession`: ejecuta sentencias con locks transparentes |

## Uso

### Sesión explícita (`BEGIN`/`COMMIT`/`ROLLBACK`)

```python
from engine.transactions.session import TransactionalSession

session = TransactionalSession(catalog)
tx = session.begin()
session.execute("INSERT INTO accounts VALUES (3, 300)")
session.commit()          # persiste y libera locks
# o session.rollback()    # deshace (undo journal) y libera locks
session.close()
```

### Contexto transaccional (commitea o deshace solo)

```python
with session.transaction():
    session.execute("SELECT * FROM accounts")
    session.execute("DELETE FROM accounts WHERE id = 1")
```

### Autocommit

Sin transacción activa, cada sentencia se ejecuta en una transacción implícita de una sola sentencia:

```python
rows = session.execute("SELECT * FROM accounts")
```

### Múltiples sesiones comparten el control de concurrencia

Para que dos sesiones se bloqueen entre sí, comparten el mismo `TransactionManager` (y por tanto el `LockManager`):

```python
from engine.transactions.transaction_manager import TransactionManager

manager = TransactionManager()
s1 = TransactionalSession(catalog, transaction_manager=manager)
s2 = TransactionalSession(catalog, transaction_manager=manager)
```

## Deadlocks

- **Detección por wait-for graph:** cuando una transacción queda bloqueada, se registra en el grafo de precedencia; si se forma un ciclo, la transacción que lo detecta es abortada con `DeadlockDetected`.
- **Timeout** (`LockNotGranted`): si el lock no se concede dentro de `LOCK_TIMEOUT_MS` (env, por defecto `3000`), la transacción se aborta.
- En ambos casos la sesión **hace rollback automático** y lanza `TransactionError`.

## Demostración con hilos

```bash
uv run python scripts/demo_concurrency.py
```

Ejecuta transferencias simultáneas sobre la tabla `accounts`:

1. **Sin control de concurrencia** → actualizaciones perdidas (race condition): el saldo final es menor al esperado.
2. **Con transacciones strict 2PL** → resultado serializable: el saldo final coincide con el esperado.

## Decisiones y limitaciones

- **Undo journal en memoria** para dar atomicidad al `ROLLBACK` (insert/remove por tabla). La persistencia/duración (WAL) se implementa en la fase 2 de recuperación.
- El motor no soporta `UPDATE`; la demo usa el patrón `DELETE` + `INSERT` para modelar el read-modify-write.
- Los locks de índices no se deshacen en el rollback (pendiente de la fase de recuperación); las pruebas usan tablas sin índices.

Ver API reference en [api/transactions](api/transactions.md).