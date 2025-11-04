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
- `claude-sonnet-4-5-20250929` - Más inteligente
- `claude-opus-4-1` - Máxima capacidad
- `claude-haiku-4-5` - Más rápido

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
---
## 1) Respuesta de atención al cliente (tono y brevedad)

**Archivo:** `00-introduccion/ej01_atencion_cliente.py`
**Objetivo:** Configura un agente que devuelva una respuesta **en 2 frases** a una queja de cliente con tono profesional y empático.
**Conceptos:** `instructions`, `run_sync`, patrón `Model(..., provider=Provider(...))`.

**Starter:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel as Model
from pydantic_ai.providers.openai import OpenAIProvider as Provider
from config import settings

model = Model("gpt-5", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(model, instructions="Responde como agente de soporte: tono empático, 2 frases, ofrece siguiente paso claro.")
print(agent.run_sync("El envío llegó dañado y nadie responde.").output)
```

**Criterio de aceptación:** 1) Máx. 2 frases. 2) Tono profesional. 3) Incluye un siguiente paso (p.ej., abrir ticket o reembolso).

---

## 2) Cualificación rápida de leads (salida estructurada)

**Archivo:** `00-introduccion/ej02_lead_scoring.py`
**Objetivo:** Extrae de un mensaje libre los campos `company`, `need`, `budget_eur` (int, opcional) y `fit` ∈ {low, medium, high}.
**Conceptos:** `output_type` con `pydantic.BaseModel`.

**Starter:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel as Model
from pydantic_ai.providers.openai import OpenAIProvider as Provider
from config import settings

class Lead(BaseModel):
    company: str
    need: str
    budget_eur: int | None = None
    fit: str = Field(pattern="^(low|medium|high)$")

model = Model("gpt-5", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(model, output_type=Lead, instructions="Extrae y normaliza campos para cualificación de lead.")
lead = agent.run_sync("Somos RetailNova, buscamos forecast de demanda, presupuesto ~15k€.").output
print(lead.model_dump())
```

**Criterio de aceptación:** Devuelve dict con claves exactas; rechaza `fit` fuera del patrón (valida el tipado).

---

## 3) Tool de negocio: cálculo de IVA y total

**Archivo:** `00-introduccion/ej03_tool_iva_total.py`
**Objetivo:** Implementa una tool `calc_total(con_impuestos: bool)` que: suma líneas, aplica IVA 21% si `con_impuestos=True`, y devuelve `{subtotal, iva, total}`.
**Conceptos:** `@agent.tool`, `RunContext`, tools simples.

**Starter:**

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel as Model
from pydantic_ai.providers.anthropic import AnthropicProvider as Provider
from config import settings

model = Model("claude-haiku-4-5", provider=Provider(api_key=settings.anthropic_api_key))
agent = Agent(model)

@agent.tool
def calc_total(ctx: RunContext[None], lineas: list[float], con_impuestos: bool=True) -> dict:
    subtotal = round(sum(lineas), 2)
    iva = round(subtotal * 0.21, 2) if con_impuestos else 0.0
    return {"subtotal": subtotal, "iva": iva, "total": round(subtotal + iva, 2)}

print(agent.run_sync("Carrito: [49.9, 10.0, 3.1]. Calcula total con IVA. Usa la tool.").output)
```

**Criterio de aceptación:** Devuelve los tres campos con 2 decimales; el agente llama a la tool.

---

## 4) Instrucciones dinámicas por marca (branding)

**Archivo:** `00-introduccion/ej04_branding_dinamico.py`
**Objetivo:** Ajusta el tono según la marca pasada en `deps`: `"ICSO"` (directo y técnico) o `"TeachTools"` (didáctico y cercano).
**Conceptos:** `deps_type`, `@agent.instructions`.

**Starter:**

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel as Model
from pydantic_ai.providers.openai import OpenAIProvider as Provider
from config import settings

model = Model("gpt-5-mini", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(model, deps_type=str)

@agent.instructions
def brand_instructions(ctx: RunContext[str]) -> str:
    brand = ctx.deps
    if brand == "ICSO":
        return "Tono: directo, técnico; 3 frases máximo."
    return "Tono: didáctico, cercano; ejemplos simples; 3 frases máximo."

print(agent.run_sync("Explica nuestra propuesta de valor en 3 frases.", deps="ICSO").output)
print(agent.run_sync("Explica nuestra propuesta de valor en 3 frases.", deps="TeachTools").output)
```

**Criterio de aceptación:** Cambia el estilo claramente entre marcas, manteniendo límite de 3 frases.

---

## 5) Manejo de errores y fallback elegante

**Archivo:** `00-introduccion/ej05_fallback_error.py`
**Objetivo:** Si la llamada falla (p.ej., sin API key), captura la excepción y devuelve un mensaje corporativo con pasos de remedio.
**Conceptos:** `try/except`, UX de error profesional.

**Starter:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel as Model
from pydantic_ai.providers.openai import OpenAIProvider as Provider
from config import settings

model = Model("gpt-5", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(model, instructions="Resumen en una frase del correo del cliente.")

try:
    print(agent.run_sync("Cliente: pide roadmap y precios para Q1.").output)
except Exception as e:
    print("No hemos podido procesar tu solicitud ahora. Verifica la API key y reintenta; si persiste, contacta soporte con el código: E-LLM-001.")
```

**Criterio de aceptación:** En fallo, muestra mensaje corporativo único (no traza), con acción y código interno.

---

## 6) Smoke test de negocio (verificación mínima)

**Archivo:** `00-introduccion/ej06_smoke_negocio.py`
**Objetivo:** Ejecuta un prompt fijo (“Redacta un asunto y 3 bullets para un email comercial sobre X”) y verifica rápido: longitud < 120 palabras y contiene “Asunto:”.
**Conceptos:** prueba mínima “listo para demo”.

**Starter:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel as Model
from pydantic_ai.providers.anthropic import AnthropicProvider as Provider
from config import settings

model = Model("claude-haiku-4-5", provider=Provider(api_key=settings.anthropic_api_key))
agent = Agent(model, instructions="Devuelve: 'Asunto: ...' y 3 viñetas claras, máx. 120 palabras.")

out = agent.run_sync("Redacta un asunto y 3 bullets para un email comercial sobre auditoría de datos.").output
ok = "Asunto:" in out and len(out.split()) < 120
print("OK" if ok else "FAIL")
print(out)
```

**Criterio de aceptación:** Imprime “OK” y la propuesta formateada; si no, “FAIL”.

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
