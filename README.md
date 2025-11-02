# PydanticAI desde cero

Curso práctico para crear **agentes** con `pydantic-ai`.  
- Módulo 0: Introducción  
- Módulo 1: Agentes básicos  
- Módulo 2: Contexto y validación  
- Módulo 3: Integración con LLMs

## Desarrollo local
Este repo usa **Dev Containers** (recomendado). Abre con VS Code → “Reopen in Container”.

## Construir la web localmente
```bash
jupyter-book clean --all .
jupyter-book build .

## Ver el curso en local

estando en _build/html ejecuta:
```bash
python -m http.server 8000

y abre http://localhost:8000 en tu navegador (VS Code hará el port forwarding automáticamente).