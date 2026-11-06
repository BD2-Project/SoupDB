# Protocolo de red (v1)

Contrato congelado entre el **gestor SoupDB** (servidor) y el **driver rsoup** (cliente). Ambos codecs implementan exactamente este formato: Python en `engine/transactions/protocol.py` y Rust en `rsoup/src/protocol.rs`.

> Referencias del driver: [repo](https://github.com/BD2-Project/rsoup) · [documentación](https://github.com/BD2-Project/rsoup/tree/main/docs) · [resumen del protocolo](https://github.com/BD2-Project/rsoup/blob/main/docs/protocolo.md). Ver también [driver.md](driver.md).

## Frame

Todos los enteros van en **big-endian**.

```
┌────────┬─────────┬─────────┬─────────────┬────────────────┐
│ magic  │ version │ opcode  │ length      │ payload        │
│ 2B "SP"│ 1B 0x01 │ 1B      │ 4B (u32 BE) │ (length bytes) │
└────────┴─────────┴─────────┴─────────────┴────────────────┘
```

- `magic`: `0x53 0x50` ("SP") para validar el framing.
- `version`: `0x01`. Si un lado recibe otra versión, responde/aborta con error de protocolo.
- `length`: longitud del payload (sin contar el header de 8 bytes).

## Opcodes

| Opcode | Valor | Dirección | Significado |
|---|---|---|---|
| `PING` | `0x01` | cliente → gestor | Heartbeat / handshake |
| `PONG` | `0x02` | gestor → cliente | Respuesta a PING |
| `BEGIN` | `0x03` | cliente → gestor | Inicia transacción |
| `COMMIT` | `0x04` | cliente → gestor | Confirma transacción |
| `ROLLBACK` | `0x05` | cliente → gestor | Deshace transacción |
| `QUERY` | `0x06` | cliente → gestor | Ejecuta SQL |
| `OK` | `0x10` | gestor → cliente | Éxito (DML/DDL con `affected`) |
| `RESULT` | `0x11` | gestor → cliente | Resultado de SELECT |
| `ERROR` | `0x12` | gestor → cliente | Error con código + mensaje |

## Payloads

### QUERY
```
length u32 BE | sql (UTF-8, length bytes)
```

### OK
```
affected u32 BE
```

### ERROR
```
code u8 | message_length u16 BE | message (UTF-8)
```

Códigos de error: `GENERIC=0x00`, `PARSE=0x01`, `EXECUTION=0x02`, `TRANSACTION=0x03`, `LOCK_TIMEOUT=0x04`, `DEADLOCK=0x05`, `RECORD_NOT_FOUND=0x06`, `CONTRACT=0x07`, `UNSUPPORTED=0x08`.

### RESULT
```
columns_count u16 BE
  por columna: name_length u16 BE | name (UTF-8) | type_code u8 | length u16 BE (0 si no aplica)
rows_count u32 BE
  por celda: value_tag u8 | valor
```

Tags de valor: `NULL=0x00` (sin bytes), `INT=0x01` (i32 BE), `FLOAT=0x02` (f64 BE), `TEXT=0x03` (len u16 BE + UTF-8), `BOOL=0x04` (1 byte 0/1), `POINT=0x05` (dos doubles big-endian, x = longitud e y = latitud).

Códigos de tipo de columna: `INT=0x01`, `FLOAT=0x02`, `VARCHAR=0x03`, `TEXT=0x04`, `BOOL=0x05`, `POINT=0x06`.

## Comportamiento

- El gestor responde a cada request con un frame; nunca envía frames no solicitados.
- `BEGIN/COMMIT/ROLLBACK` mapean al `TransactionalSession` de la conexión; `QUERY` se ejecuta dentro de la transacción activa (o autocommit si no hay una).
- Versión y magic inválidos se rechazan como `ProtocolError`.
---

# Protocolo v2 (schema-driven + prepared statements)

La versión `0x02` es retrocompatible a nivel de framing: el header de 8 bytes y
los opcodes existentes no cambian. El servidor responde con la **misma versión**
que usó el cliente en el frame de request. La versión `0x01` sigue soportada
para los seis tipos nativos.

## Header de columnas en RESULT

v1 enviaba un código numérico de tipo. v2 envía el nombre del tipo y cómo debe
interpretarse el valor:

```
[ncols u16]
por columna:
  [name_len u16][name UTF-8]
  [type_name_len u16][type_name UTF-8]   # "INT", "TEXT", "POINT", "VECTOR"...
  [encoding u8]                          # 0 native, 1 fixed, 2 variable, 3 opaque
  [size u16]                             # bytes fijos (0 si variable/opaco)
  [length u16]                           # longitud declarada de VARCHAR
[row_count u32]
filas: valores con los tags existentes (NULL/INT/FLOAT/TEXT/BOOL/POINT/BYTES)
```

Un tipo desconocido por el cliente se transporta como `encoding=opaque` y el
cliente lo expone como bytes + `type_name`; no necesita conocerlo para seguir
funcionando.

## Prepared statements

| Opcode | Valor | Payload |
|---|---|---|
| `PREPARE` | `0x07` | `[name_len u16][name][sql_len u32][sql]` |
| `EXECUTE` | `0x08` | `[name_len u16][name][nparams u16][tag+valor...]` |
| `DEALLOCATE` | `0x09` | `[name_len u16][name]` |

- `PREPARE` parsea una sola vez y guarda el AST en la conexión.
- `EXECUTE` reemplaza `$1..$n` **en el AST** (nunca interpola SQL) y ejecuta con
  la sesión transaccional de esa conexión.
- `DEALLOCATE` libera el statement preparado. Cerrar la conexión libera todos.
