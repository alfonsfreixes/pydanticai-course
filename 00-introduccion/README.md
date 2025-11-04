# Módulo 0 · Introducción a PydanticAI

**PydanticAI** es un framework Python para construir aplicaciones GenAI con validación de tipos y observabilidad integrada. Desarrollado por el equipo de Pydantic, alcanzó V1 en septiembre 2025 con política de estabilidad de API.

**Versión actual:** PydanticAI 1.9.1 (octubre 2025) | Pydantic 2.12.3 | Python ≥3.10

---

## ¿Por qué PydanticAI?

- **Type-safe**: Validación Pydantic nativa para entradas/salidas
- **Model-agnostic**: OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama y compatibles OpenAI
- **Observability**: Integración directa con Pydantic Logfire
- **Pythonic**: Control de flujo nativo sin DSLs

---

## Prerrequisitos

- Python 3.10+ (recomendado 3.12)
- Conocimientos Python: async/await, type hints, Pydantic
- API key de OpenAI o Anthropic

---

## 1. Setup del proyecto

### 1.1. Instalar uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell admin)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2. Inicializar proyecto

```bash
mkdir pydanticai-course && cd pydanticai-course
uv init --name pydanticai-course
mkdir 00-introduccion 01-agentes-basicos 02-contexto-validacion 03-integracion-llms
```

### 1.3. Dependencias y configuración

```bash
uv add pydantic-ai pydantic-settings
```

**`pyproject.toml`:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pydanticai-course"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic-ai>=1.9.0",
    "pydantic-settings>=2.11.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
```

```bash
uv pip install -e .
```

**`.env`:**

```bash
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-..."
```

**`config.py`:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

settings = Settings()
```

---

## 2. Conceptos fundamentales

### Agent

Un `Agent` encapsula:
- Modelo LLM
- Instrucciones (system prompt)
- Tools/funciones disponibles
- Esquema de salida (opcional)
- Dependencias (opcional)

### Model + Provider

PydanticAI usa un patrón explícito `Model(..., provider=Provider(...))` que permite:
- Pasar API keys programáticamente
- Personalizar clientes HTTP
- Configurar endpoints custom

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(model)
```

---

## 3. Hello World

**`00-introduccion/hello_agent.py`:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(
    model,
    instructions="Responde de forma concisa en una frase."
)

try:
    result = agent.run_sync('¿De dónde viene "Hello World"?')
    print(result.output)
except Exception as e:
    print(f"Error: {e}")
```

```bash
uv run python 00-introduccion/hello_agent.py
```

---

## 4. Configuración de proveedores

### Anthropic Claude (recomendado para agentes)

**Modelos disponibles (noviembre 2025):**
- `claude-sonnet-4-5-20250929` - Más inteligente (\$3/\$15)
- `claude-opus-4-1` - Máxima capacidad (\$15/\$75)
- `claude-haiku-4-5` - Más rápido (\$1/\$5)

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)
```

### OpenAI GPT

**Modelos disponibles (noviembre 2025):**
- `gpt-5` - Modelo unificado con reasoning
- `gpt-5-mini` - Eficiente
- `gpt-5-codex` - Especializado en código

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)
```

### Otros proveedores

PydanticAI soporta **Gemini, Groq, Mistral, Cohere** y cualquier proveedor compatible con OpenAI API (OpenRouter, Together, Fireworks, Ollama, etc.). Ver [documentación oficial](https://ai.pydantic.dev/models/overview/).

---

## 5. Ejemplos prácticos

### 5.1. Salida estructurada

**`00-introduccion/structured_output.py`:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    highlights: list[str] = Field(max_length=3)

model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=CityInfo,
    instructions="Extrae info de la ciudad mencionada."
)

city = agent.run_sync("Cuéntame sobre Barcelona").output
print(f"Ciudad: {city.name}")
print(f"País: {city.country}")
print(f"Población: {city.population:,}")
print(f"Destacados:")
for highlight in city.highlights:
    print(f"  - {highlight}")
```

### 5.2. Tools

**`00-introduccion/agent_with_tools.py`:**

```python
from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(model)

@agent.tool
def get_current_time(ctx: RunContext[None]) -> str:
    """Devuelve la hora actual del sistema."""
    return datetime.now().strftime("%H:%M:%S")

print(agent.run_sync("¿Qué hora es?").output)
```

### 5.3. Instrucciones dinámicas

**`00-introduccion/dynamic_instructions.py`:**

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(model, deps_type=str)

@agent.instructions
def get_instructions(ctx: RunContext[str]) -> str:
    return f"Eres un asistente. El usuario es {ctx.deps}."

result = agent.run_sync("¿Cuál es tu función?", deps="María")
print(result.output)
```

### 5.4. Streaming

**`00-introduccion/streaming_example.py`:**

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(model)

async def stream_response():
    async with agent.run_stream("Explica async/await en Python") as result:
        async for text in result.stream_text():
            print(text, end="", flush=True)
    print()  # Nueva línea al final

asyncio.run(stream_response())
```

**Ejecutar ejemplos:**

```bash
uv run python 00-introduccion/structured_output.py
uv run python 00-introduccion/agent_with_tools.py
uv run python 00-introduccion/dynamic_instructions.py
uv run python 00-introduccion/streaming_example.py
```
---

## 6. Observabilidad con Logfire (opcional)

**Configurar (primera vez desde terminal)**
```bash
uv run logfire auth
```

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# Todos los agentes quedan monitoreados
agent = Agent(model)
```

Ver trazas en [logfire.pydantic.dev](https://logfire.pydantic.dev) con tokens, tiempos y tool calls.

---

## 7. Troubleshooting

| Error | Solución |
|-------|----------|
| `No API key found` | Verifica `.env` y que `config.py` carga correctamente |
| `Module not found` | Ejecuta con `uv run python script.py` |
| `Python too old` | `uv python install 3.12 && uv python pin 3.12` |

---

## 8. Ejercicios

### Básico

**1. Modificar el primer agente**
- Archivo: `00-introduccion/ejercicio_01_modificar.py`
- Objetivo: Cambia las instrucciones de `hello_agent.py` para que responda como un pirata
- Conceptos: Parámetro `instructions`

**2. Experimentar con prompts**
- Archivo: `00-introduccion/ejercicio_02_prompts.py`
- Objetivo: Crea un agente y prueba 3 preguntas diferentes sobre Python
- Conceptos: `agent.run_sync()`, diferentes inputs

**3. Cambiar de modelo**
- Archivo: `00-introduccion/ejercicio_03_modelo.py`
- Objetivo: Si tienes OpenAI, modifica `hello_agent.py` para usar GPT-5 en lugar de Claude
- Conceptos: Diferentes modelos y providers

### Opcional

**4. Comparar respuestas**
- Archivo: `00-introduccion/ejercicio_04_comparar.py`
- Objetivo: Si tienes 2 API keys, ejecuta el mismo prompt en ambos modelos y compara
- Conceptos: Múltiples modelos (solo si aplica)

**5. Manejo de errores**
- Archivo: `00-introduccion/ejercicio_05_errores.py`
- Objetivo: Modifica `hello_agent.py` para mostrar un mensaje personalizado si falla
- Conceptos: `try/except`, mensajes de error claros

**6. Script de verificación personalizado**
- Archivo: `00-introduccion/ejercicio_06_verificar.py`
- Objetivo: Crea tu propio script de verificación que pruebe que tu agente funciona
- Conceptos: Testing básico, validación de setup

---

## 9. Próximos pasos

**Módulo 1** - Agentes avanzados, tools complejas, validación y reflection
**Módulo 2** - Contexto, conversación e historial
**Módulo 3** - Producción, optimización y deployment

---

## Referencias

- [PydanticAI Docs](https://ai.pydantic.dev)
- [Models & Providers](https://ai.pydantic.dev/models/overview/)
- [API Reference](https://ai.pydantic.dev/api/agent/)
- [Anthropic Claude](https://docs.claude.com/en/docs/about-claude/models/overview)
- [OpenAI Models](https://platform.openai.com/docs/models/)

---
