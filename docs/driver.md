# Driver rsoup

El **driver `rsoup`** ([BD2-Project/rsoup](https://github.com/BD2-Project/rsoup)) es el cliente en **Rust** que los consumidores usan para conectarse al **pool de conexiones del gestor**. Se comunica por **TCP/IP** (capa 4) usando el **protocolo binario v1** (ver [protocolo](protocolo.md)).

## Cadena de comunicación

```
+---------------------+      TCP/IP (protocolo v1)      +--------------------------+
| Cliente (Svelte/     | <----------------------------> | SoupDB gestor (Python)   |
| SoupChef + Tauri)    |   QUERY / BEGIN / COMMIT /     | ConnectionHandler (pool) |
|        |             |   ROLLBACK + RESULT/OK/ERROR    | DRIVER_PORT 55432        |
|        | invoke      |                                | MAX_CONNECTIONS 100      |
|        v             |                                |                          |
|   rsoup (Rust)       |                                +--------------------------+
|   driver cliente     |
+---------------------+
```

- **Pool de conexiones:** lo gestiona el gestor (`engine/transactions/connection_handler.py`): listener, worker por conexión y límite `MAX_CONNECTIONS`.
- **Driver:** abre una conexión al puerto del gestor y expone una API de alto nivel (`Connection`, `TransactionManager`, `Transaction`).
- **Frontend:** SoupChef integra el driver como comando de Tauri en el backend Rust (`src-tauri`) y lo invoca desde Svelte con `invoke`.

## Referencias

| Recurso | Enlace |
|---|---|
| Repositorio del driver | https://github.com/BD2-Project/rsoup |
| Documentación del driver | https://github.com/BD2-Project/rsoup/tree/main/docs |
| API pública (firmas) | https://github.com/BD2-Project/rsoup/blob/main/docs/api.md |
| Ejemplos con Tauri | https://github.com/BD2-Project/rsoup/blob/main/docs/ejemplos.md |
| Protocolo v1 | [protocolo.md](protocolo.md) de este repo |

## Parámetros de conexión

| Variable | Default | Uso |
|---|---|---|
| `DRIVER_HOST` | `0.0.0.0` | Interface donde escucha el gestor |
| `DRIVER_PORT` | `55432` | Puerto del pool del gestor |
| `MAX_CONNECTIONS` | `100` | Conexiones simultáneas del gestor |

El driver cliente usa `DRIVER_HOST`/`DRIVER_PORT` para conectarse (ver documentación del driver).