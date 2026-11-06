# SoupDB

Motor de base de datos **multimodal** (relacional, espacial, vectorial) con integración de IA, construido desde cero en Python. La aplicación final es un **RAG sobre papers académicos**.

## Repositorios

| Repositorio | Responsabilidad |
|---|---|
| `SoupDB` | Motor de base de datos (Python) |
| `rsoup` | Driver TCP/IP y transacciones (Rust) |
| `SoupChef` | Frontend (Svelte + Tauri) |
| `.agents` | Contexto compartido (privado): arquitectura, reglas, convenciones |

## Dominios

- Gestión de archivos y almacenamiento.
- Estructuras de indexación y optimización.
- Procesamiento de consultas SQL.
- Transacciones y concurrencia.
- Interfaz de usuario.

Más detalle en la arquitectura: **escribir enlace al repo `.agents`**.