# SoupDB

Motor de base de datos **multimodal** (relacional, espacial, vectorial) con integración de IA, construido desde cero en Python. La aplicación final (Parte 5) es una
**aplicación de análisis de similitud de audio**: busca posibles coincidencias entre un audio
consultado y las canciones originales de un catálogo (copyright/remixes). Ver
[API de copyright](copyright_api.md).

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