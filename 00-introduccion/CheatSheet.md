# PydanticAI Cheat Sheet - Módulo 0

Referencia rápida de los comandos y patrones más usados.

---

## 🚀 Instalación rápida

```bash
# Instalación completa
pip install pydantic-ai

# Solo OpenAI
pip install 'pydantic-ai-slim[openai]'

# Solo Anthropic
pip install 'pydantic-ai-slim[anthropic]'

# Verificar instalación
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
```

---

## 🔑 Configurar API Keys

```bash
# En terminal (Linux/Mac)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# En Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# O crear archivo .env
echo 'OPENAI_API_KEY="sk-..."' > .env
```

---

## 🤖 Crear Agentes

### Agente básico

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')
result = agent.run_sync('¿Qué es Python?')
print(result.output)
```

### Con instrucciones

```python
agent = Agent(
    'openai:gpt-4o-mini',
    instructions='Eres un experto en Python. Responde de forma concisa.'
)
```

### Instrucciones dinámicas

```python
@agent.instructions
def get_instructions(ctx: RunContext[str]) -> str:
    return f"El usuario es {ctx.deps}"
```

---

## 🎯 Modelos de diferentes proveedores

```python
# OpenAI
agent = Agent('openai:gpt-4o-mini')
agent = Agent('openai:gpt-4o')

# Anthropic
agent = Agent('anthropic:claude-sonnet-4-0')
agent = Agent('anthropic:claude-opus-4-0')

# Google Gemini
agent = Agent('google-gla:gemini-2.5-flash')
agent = Agent('google-gla:gemini-2.5-pro')

# Groq (rápido)
agent = Agent('groq:llama-3.3-70b-versatile')

# Ollama (local)
agent = Agent('ollama:llama3.2')
```

---

## 📊 Salidas estructuradas

```python
from pydantic import BaseModel, Field

class CityInfo(BaseModel):
    name: str
    country: str
    population: int = Field(gt=0)

agent = Agent('openai:gpt-4o', output_type=CityInfo)
result = agent.run_sync('Info sobre París')
city = result.output  # Tipo CityInfo validado
```

---

## 🛠️ Tools / Funciones

### Tool básica

```python
@agent.tool
def get_time(ctx: RunContext[None]) -> str:
    """Obtiene la hora actual."""
    return datetime.now().strftime("%H:%M")
```

### Tool con parámetros

```python
@agent.tool
def sumar(a: float, b: float) -> float:
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

@dataclass
class Deps:
    user_id: int
    db: Database

agent = Agent('openai:gpt-4o-mini', deps_type=Deps)

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
agent = Agent('openai:gpt-4o', model_settings=settings)

# En runtime
result = agent.run_sync('prompt', model_settings=settings)
```

---

## 🔍 Debugging con Logfire

```python
import logfire

# Configurar (primera vez)
# logfire auth

# En tu código
logfire.configure()
logfire.instrument_pydantic_ai()

# Ahora todos los agentes son observables en:
# https://logfire.pydantic.dev
```

---

## ✅ Patrones comunes

### Pattern 1: Extractor de datos

```python
class Data(BaseModel):
    name: str
    value: float

agent = Agent('openai:gpt-4o', output_type=Data)
result = agent.run_sync("Apple vale $180")
# result.output.name == "Apple"
# result.output.value == 180.0
```

### Pattern 2: Multi-tool agent

```python
@agent.tool
def tool_a(ctx): ...

@agent.tool
def tool_b(ctx): ...

@agent.tool
def tool_c(ctx): ...

# El modelo decide cuál usar
```

### Pattern 3: Context-aware

```python
@agent.instructions
def instructions(ctx: RunContext[User]) -> str:
    return f"Usuario: {ctx.deps.name}, nivel: {ctx.deps.level}"

result = agent.run_sync('query', deps=current_user)
```

---

## 🚨 Errores comunes

### Error: No API key found

```bash
# Solución
export OPENAI_API_KEY="tu-key"
```

### Error: Python version too old

```bash
# Solución: actualizar a Python 3.10+
pyenv install 3.12
pyenv local 3.12
```

### Error: Module not found

```bash
# Solución: instalar dependencias
pip install 'pydantic-ai-slim[openai]'
```

---

## 📚 Recursos rápidos

- **Docs**: https://ai.pydantic.dev
- **Ejemplos**: https://ai.pydantic.dev/examples/
- **API Ref**: https://ai.pydantic.dev/api/agent/
- **GitHub**: https://github.com/pydantic/pydantic-ai
- **Slack**: https://logfire.pydantic.dev/docs/join-slack/

---

## 🎯 Comandos útiles

```bash
# Ejecutar ejemplos
python ejemplos_basicos.py

# Verificar setup
python verificar_setup.py

# Jupyter notebook
jupyter notebook notebook_intro.ipynb

# Ver tokens usados
python -c "from pydantic_ai import Agent; a=Agent('openai:gpt-4o-mini'); r=a.run_sync('hola'); print(r.usage())"
```

---

## 💡 Tips rápidos

1. **Empieza simple**: Un agente básico primero
2. **Valida siempre**: Usa Pydantic models para outputs
3. **Type hints**: TypeScript para Python, úsalos
4. **Logfire early**: Configúralo desde el principio
5. **Async cuando puedas**: Mejor performance
6. **Tools pequeñas**: Una función, una responsabilidad
7. **Tests con mocks**: Usa TestModel para testing
8. **Deps inyectadas**: Mejor que variables globales
9. **Streaming UX**: Para mejor experiencia de usuario
10. **Cost tracking**: Monitorea tokens desde el día 1

---

**¡Esto es todo lo que necesitas para empezar! 🚀**