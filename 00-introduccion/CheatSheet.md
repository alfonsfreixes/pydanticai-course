# PydanticAI Cheat Sheet - Módulo 0

Referencia rápida de comandos y patrones más usados con **uv** y **Pydantic Settings**.

---

## 🚀 Setup inicial (una sola vez)

```bash
# 1. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# o en Windows PowerShell:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Crear proyecto
mkdir pydanticai-course && cd pydanticai-course
uv init --name pydanticai-course

# 3. Crear estructura de módulos
mkdir 00-introduccion 01-agentes-basicos 02-contexto-validacion 03-integracion-llms

# 4. Instalar dependencias
uv add pydantic-ai pydantic-settings

# 5. Crear config.py y .env (ver sección Configuración)
```

---

## 🔧 Configuración con Pydantic Settings

### Archivo `config.py` (raíz del proyecto)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

settings = Settings()
```

### Archivo `.env` (raíz del proyecto)

```bash
# .env
ANTHROPIC_API_KEY="sk-ant-tu-clave-aqui"
OPENAI_API_KEY="sk-tu-clave-aqui"
GOOGLE_API_KEY=""
GROQ_API_KEY=""
```

### Usar en tus scripts

```python
from config import settings  # Se carga automáticamente desde .env
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')
# Las API keys ya están disponibles
```

---

## 🤖 Crear Agentes

### Agente básico

```python
from pydantic_ai import Agent
from config import settings

agent = Agent('openai:gpt-5')
result = agent.run_sync('¿Qué es Python?')
print(result.output)
```

### Con instrucciones

```python
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Eres un experto en Python. Responde de forma concisa.'
)
```

### Instrucciones dinámicas

```python
from pydantic_ai import RunContext

@agent.instructions
def get_instructions(ctx: RunContext[str]) -> str:
    return f"El usuario es {ctx.deps}"
```

---

## 🎯 Modelos disponibles (noviembre 2025)

```python
from config import settings

# OpenAI (2025)
agent = Agent('openai:gpt-5')           # Modelo principal
agent = Agent('openai:o3-mini')         # Razonamiento rápido
agent = Agent('openai:o4-mini')         # Razonamiento eficiente

# Anthropic Claude 4
agent = Agent('anthropic:claude-sonnet-4-5')  # Más inteligente
agent = Agent('anthropic:claude-sonnet-4-0')  # Equilibrado
agent = Agent('anthropic:claude-opus-4-1')    # Máxima capacidad

# Google Gemini 2.5
agent = Agent('google-gla:gemini-2.5-flash')  # Ultra rápido
agent = Agent('google-gla:gemini-2.5-pro')    # Mayor capacidad

# Groq (ultra rápido)
agent = Agent('groq:llama-3.3-70b-versatile')

# Ollama (local)
agent = Agent('ollama:llama3.2')
```

---

## 📊 Salidas estructuradas

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from config import settings

class CityInfo(BaseModel):
    name: str
    country: str
    population: int = Field(gt=0)
    famous_for: str

agent = Agent('openai:gpt-5', output_type=CityInfo)
result = agent.run_sync('Info sobre París')
city = result.output  # Tipo CityInfo validado
```

---

## 🛠️ Tools / Funciones

### Tool básica

```python
from datetime import datetime
from pydantic_ai import RunContext

@agent.tool
def get_time(ctx: RunContext[None]) -> str:
    """Obtiene la hora actual."""
    return datetime.now().strftime("%H:%M")
```

### Tool con parámetros

```python
@agent.tool
def sumar(ctx: RunContext[None], a: float, b: float) -> float:
    """Suma dos números."""
    return a + b
```

### Tool sin contexto

```python
@agent.tool_plain
def get_pi() -> float:
    """Retorna el valor de PI."""
    return 3.14159
```

---

## 🔄 Modos de ejecución

### Síncrono (simple)

```python
result = agent.run_sync('prompt')
print(result.output)
```

### Asíncrono (mejor performance)

```python
result = await agent.run('prompt')
print(result.output)
```

### Streaming (tiempo real)

```python
async with agent.run_stream('prompt') as result:
    async for text in result.stream_text():
        print(text, end='', flush=True)
```

---

## 💉 Dependency Injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from config import settings

@dataclass
class Deps:
    user_id: int
    db: Database

agent = Agent('openai:gpt-5', deps_type=Deps)

@agent.tool
def get_user_data(ctx: RunContext[Deps]) -> dict:
    """Obtiene datos del usuario."""
    return ctx.deps.db.get_user(ctx.deps.user_id)

# Ejecutar con dependencias
result = agent.run_sync('query', deps=Deps(user_id=123, db=db))
```

---

## 📈 Información de uso

```python
result = agent.run_sync('prompt')

# Tokens usados
usage = result.usage()
print(f"Tokens entrada: {usage.request_tokens}")
print(f"Tokens salida: {usage.response_tokens}")
print(f"Total: {usage.total_tokens}")
```

---

## 🎨 Configuración del modelo

```python
from pydantic_ai import ModelSettings

settings = ModelSettings(
    temperature=0.7,
    max_tokens=1000,
    timeout=30.0
)

# En el agente
agent = Agent('openai:gpt-5', model_settings=settings)

# En runtime
result = agent.run_sync('prompt', model_settings=settings)
```

---

## 🔍 Observabilidad con Logfire

```python
import logfire
from pydantic_ai import Agent
from config import settings

# Configurar (primera vez desde terminal)
# uv run logfire auth

# En tu código
logfire.configure()
logfire.instrument_pydantic_ai()

# Todos los agentes son observables en:
# https://logfire.pydantic.dev
```

---

## ✅ Patrones comunes

### Pattern 1: Extractor de datos

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from config import settings

class Data(BaseModel):
    name: str
    value: float

agent = Agent('openai:gpt-5', output_type=Data)
result = agent.run_sync("Apple vale $180")
# result.output.name == "Apple"
# result.output.value == 180.0
```

### Pattern 2: Multi-tool agent

```python
from pydantic_ai import Agent, RunContext
from config import settings

agent = Agent('openai:gpt-5')

@agent.tool
def tool_a(ctx: RunContext[None]) -> str:
    return "Resultado A"

@agent.tool
def tool_b(ctx: RunContext[None]) -> str:
    return "Resultado B"

@agent.tool
def tool_c(ctx: RunContext[None]) -> str:
    return "Resultado C"

# El modelo decide cuál usar
```

### Pattern 3: Context-aware

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from config import settings

@dataclass
class User:
    name: str
    level: int

agent = Agent('openai:gpt-5', deps_type=User)

@agent.instructions
def instructions(ctx: RunContext[User]) -> str:
    return f"Usuario: {ctx.deps.name}, nivel: {ctx.deps.level}"

result = agent.run_sync('query', deps=current_user)
```

---

## 🚨 Troubleshooting

### Error: No API key found

```bash
# Verifica tu archivo .env
cat .env  # Linux/macOS
type .env  # Windows

# Debe contener:
OPENAI_API_KEY="sk-..."
```

### Error: Settings validation error

```bash
# Asegúrate de tener pydantic-settings instalado
uv add pydantic-settings

# Verifica que config.py existe en la raíz
```

### Error: Module 'pydantic_ai' not found

```bash
# Ejecuta siempre con uv run
uv run python tu_script.py

# O reinstala
uv add pydantic-ai
```

### Error: No se encuentra .env

```bash
# Ejecuta desde la raíz del proyecto
cd pydanticai-course
uv run python 00-introduccion/hello_agent.py
```

---

## 🎯 Comandos útiles con uv

```bash
# Ejecutar script
uv run python 00-introduccion/hello_agent.py

# Verificar instalación
uv run python verify_setup.py

# Agregar dependencia
uv add nombre-paquete

# Ver dependencias instaladas
uv pip list

# Sincronizar dependencias
uv sync

# Actualizar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 📚 Recursos rápidos

- **Docs**: https://ai.pydantic.dev
- **Ejemplos**: https://ai.pydantic.dev/examples/
- **API Ref**: https://ai.pydantic.dev/api/agent/
- **GitHub**: https://github.com/pydantic/pydantic-ai
- **Pydantic Settings**: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **uv Docs**: https://docs.astral.sh/uv/

---

## 💡 Tips rápidos

1. **Usa uv run**: Siempre ejecuta con `uv run python script.py`
2. **config.py primero**: Crea tu configuración antes de cualquier script
3. **Valida siempre**: Usa Pydantic models para outputs
4. **Type hints**: Fundamentales para auto-completado
5. **Logfire early**: Configúralo desde el principio
6. **Async cuando puedas**: Mejor performance
7. **Tools pequeñas**: Una función, una responsabilidad
8. **Deps inyectadas**: Mejor que variables globales
9. **Streaming UX**: Para mejor experiencia de usuario
10. **Cost tracking**: Monitorea tokens desde el día 1

---

## 📋 Estructura del proyecto

```
pydanticai-course/
├── config.py                    # ⭐ Configuración con Pydantic Settings
├── .env                         # ⭐ API keys (no subir a git)
├── verify_setup.py              # Script de verificación
├── pyproject.toml               # Dependencias (gestionado por uv)
├── .python-version              # Versión de Python
├── 00-introduccion/             # Módulo 0
├── 01-agentes-basicos/          # Módulo 1
├── 02-contexto-validacion/      # Módulo 2
└── 03-integracion-llms/         # Módulo 3
```

---

## 🔑 Ejemplo completo mínimo

```python
# 00-introduccion/hello_agent.py
from pydantic_ai import Agent
from config import settings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Responde de forma concisa.'
)

result = agent.run_sync('¿Qué es Python?')
print(result.output)
```

**Ejecutar:**
```bash
uv run python 00-introduccion/hello_agent.py
```

---

**¡Con esto tienes todo lo esencial para trabajar con PydanticAI! 🚀**

> **Nota**: Esta cheat sheet está actualizada a noviembre 2025 con uv, Pydantic Settings y los modelos LLM más recientes.