# Módulo 0 · Introducción a PydanticAI

**PydanticAI** es un framework de agentes Python diseñado para construir aplicaciones GenAI de producción de forma rápida, segura y con validación de tipos. Creado por el equipo de Pydantic, aplica la misma filosofía de diseño que hizo de FastAPI un estándar en desarrollo web.

## ¿Por qué PydanticAI?

- **Validación nativa**: Construido por el equipo que creó Pydantic, la capa de validación de OpenAI SDK, Anthropic SDK, LangChain, LlamaIndex y más
- **Model-agnostic**: Soporte para OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, Groq, Ollama, y cualquier proveedor compatible con OpenAI API
- **Type-safe**: Detección de errores en tiempo de desarrollo, no en producción
- **Observabilidad integrada**: Integración nativa con Pydantic Logfire para debugging y monitoreo

---

## Objetivos del módulo

Al completar este módulo, serás capaz de:

1. Instalar y configurar PydanticAI en tu entorno
2. Crear y ejecutar tu primer agente básico
3. Configurar diferentes proveedores LLM (OpenAI, Anthropic, Gemini, etc.)
4. Comprender la arquitectura fundamental de un agente PydanticAI

---

## Prerrequisitos

- Python **3.10 o superior**
- Conocimientos básicos de Python (funciones, tipos, async/await)
- Cuenta y API key de al menos un proveedor LLM (OpenAI, Anthropic, etc.)

---

## 1. Instalación

### Instalación completa (recomendada para empezar)

```bash
pip install pydantic-ai
```

Esta instalación incluye:
- Framework core de PydanticAI
- Dependencias para todos los modelos soportados
- Pydantic Logfire para observabilidad

### Instalación slim (para producción)

Si solo vas a usar un proveedor específico:

```bash
# Solo para OpenAI
pip install 'pydantic-ai-slim[openai]'

# Solo para Anthropic
pip install 'pydantic-ai-slim[anthropic]'

# Solo para Gemini
pip install 'pydantic-ai-slim[gemini]'
```

### Verificación de instalación

```bash
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
```

---

## 2. Hello World con PydanticAI

El ejemplo más simple posible:

```python
from pydantic_ai import Agent

# Crear un agente con instrucciones básicas
agent = Agent(
    'anthropic:claude-sonnet-4-0',
    instructions='Responde de forma concisa en una sola frase.'
)

# Ejecutar el agente de forma síncrona
result = agent.run_sync('¿De dónde viene "Hello World"?')
print(result.output)
```

**Salida esperada:**
```
El primer uso conocido de "hello, world" fue en un libro de texto sobre el lenguaje C en 1974.
```

### Ejecutar el ejemplo

```bash
# Configurar tu API key
export ANTHROPIC_API_KEY="tu-api-key-aqui"

# Ejecutar
python hello_agent.py
```

---

## 3. Configuración de proveedores

### OpenAI (GPT-4, GPT-4o, GPT-4o-mini)

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')

# Configurar API key como variable de entorno
# export OPENAI_API_KEY="sk-..."
```

### Anthropic (Claude)

```python
agent = Agent('anthropic:claude-sonnet-4-0')

# export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google Gemini

```python
agent = Agent('google-gla:gemini-2.5-flash')

# export GOOGLE_API_KEY="..."
```

### Groq (ultra rápido)

```python
agent = Agent('groq:llama-3.3-70b-versatile')

# export GROQ_API_KEY="..."
```

### Ollama (local)

```python
agent = Agent('ollama:llama3.2')

# No requiere API key, pero necesitas Ollama corriendo localmente
```

### Proveedores compatibles con OpenAI API

PydanticAI puede usar cualquier proveedor compatible con OpenAI API:

```python
from pydantic_ai.models.openai import OpenAIModel

# DeepSeek
model = OpenAIModel(
    'deepseek-chat',
    base_url='https://api.deepseek.com',
    api_key='tu-deepseek-key'
)

# OpenRouter
model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    base_url='https://openrouter.ai/api/v1',
    api_key='tu-openrouter-key'
)

agent = Agent(model)
```

---

## 4. Ejemplos prácticos básicos

### Ejemplo 1: Agente con instrucciones dinámicas

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o-mini', deps_type=str)

@agent.instructions
def get_instructions(ctx: RunContext[str]) -> str:
    """Instrucciones que cambian según el contexto"""
    return f"Eres un asistente útil. El usuario se llama {ctx.deps}."

# Ejecutar con dependencias
result = agent.run_sync(
    "¿Cuál es tu función?",
    deps="María"
)
print(result.output)
# "Hola María, mi función es ayudarte con tus preguntas..."
```

### Ejemplo 2: Diferentes modos de ejecución

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')

# 1. Síncrono (más simple)
result_sync = agent.run_sync("¿Qué es Python?")
print(result_sync.output)

# 2. Asíncrono (mejor para I/O)
async def main():
    result_async = await agent.run("¿Qué es Python?")
    print(result_async.output)

asyncio.run(main())

# 3. Streaming (respuesta en tiempo real)
async def stream_example():
    async with agent.run_stream("Explica qué es Python") as result:
        async for text in result.stream_text():
            print(text, end='', flush=True)

asyncio.run(stream_example())
```

### Ejemplo 3: Salida estructurada con Pydantic

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class CityInfo(BaseModel):
    """Información estructurada sobre una ciudad"""
    name: str = Field(description="Nombre de la ciudad")
    country: str = Field(description="País donde se encuentra")
    population: int = Field(description="Población aproximada")
    famous_for: str = Field(description="Por qué es famosa")

# Crear agente que devuelve datos estructurados
agent = Agent(
    'openai:gpt-4o',
    output_type=CityInfo,
    instructions="Extrae información sobre la ciudad mencionada."
)

result = agent.run_sync("Cuéntame sobre Barcelona")
city = result.output

# Datos validados y tipados
print(f"{city.name}, {city.country}")
print(f"Población: {city.population:,}")
print(f"Famosa por: {city.famous_for}")
```

### Ejemplo 4: Primera tool/función

```python
from datetime import datetime
from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o-mini',
    instructions="Puedes decir la fecha y hora actual cuando te lo pidan."
)

@agent.tool
def get_current_time(ctx: RunContext[None]) -> str:
    """Obtiene la fecha y hora actual del sistema."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

result = agent.run_sync("¿Qué hora es?")
print(result.output)
# "La hora actual es 2025-10-31 14:30:45"
```

---

## 5. Conceptos fundamentales

### ¿Qué es un Agente?

Un **Agent** en PydanticAI es:

- La interfaz principal para interactuar con LLMs
- Un contenedor que encapsula:
  - El modelo LLM a usar
  - Instrucciones del sistema (estáticas o dinámicas)
  - Herramientas/funciones que el modelo puede llamar
  - Esquema de salida (opcional)
  - Dependencias compartidas (opcional)

```python
from pydantic_ai import Agent

agent = Agent(
    model='openai:gpt-4o',              # Modelo a usar
    instructions='Eres un experto en...',  # Sistema prompt
    output_type=MiModelo,                # Salida estructurada
    deps_type=MisDependencias,          # Tipo de dependencias
    tools=[tool1, tool2],               # Herramientas disponibles
)
```

### Flujo de ejecución

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │ Prompt
       ▼
┌─────────────┐
│    Agent    │ ──► Instrucciones + Prompt
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │ ◄─► Tools (opcional)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Validación │ (Pydantic)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Respuesta  │
└─────────────┘
```

---

## 6. Estructura del proyecto

Estructura recomendada para el curso:

```
pydanticai-course/
├── 00-introduccion/
│   ├── README.md                 # Este archivo
│   ├── hello_agent.py           # Ejemplo básico
│   ├── structured_output.py     # Salidas estructuradas
│   ├── first_tool.py            # Primera herramienta
│   └── notebook_intro.ipynb     # Notebook interactivo
├── 01-agentes-basicos/
├── 02-tools-avanzadas/
└── requirements.txt
```

---

## 7. Configuración de Logfire (opcional pero recomendado)

Logfire es la plataforma de observabilidad de Pydantic. Te permite:
- Ver el flujo completo de conversaciones
- Debuggear llamadas a tools
- Monitorear costos y tokens
- Analizar performance

### Instalación y configuración

```bash
# Ya incluido con pydantic-ai (no con slim)
pip install pydantic-ai

# Autenticar (primera vez)
logfire auth

# En tu código
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# Ahora todos tus agentes serán monitoreados
```

Visita [logfire.pydantic.dev](https://logfire.pydantic.dev) para ver tus traces.

---

## 8. Comparación rápida con otros frameworks

| Característica | PydanticAI | LangChain | CrewAI |
|---------------|------------|-----------|---------|
| Type Safety | ✅ Completo | ⚠️ Parcial | ⚠️ Parcial |
| Validación | ✅ Pydantic | ⚠️ Manual | ⚠️ Manual |
| Observabilidad | ✅ Integrada | 🔧 Requiere setup | 🔧 Requiere setup |
| Curva aprendizaje | 🟢 Baja | 🟡 Media | 🟡 Media |
| Control flow | Python nativo | Chains/LCEL | Predefinido |

---

## 9. Troubleshooting común

### Error: "No API key found"

```bash
# Asegúrate de exportar la variable correcta
export OPENAI_API_KEY="sk-..."

# En Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# O crear archivo .env
echo 'OPENAI_API_KEY="sk-..."' > .env
```

### Error: "Python version too old"

```bash
python --version  # Debe ser 3.10+

# Actualizar con pyenv
pyenv install 3.12
pyenv local 3.12
```

### Error de importación con modelos slim

```bash
# Si instalaste slim sin las dependencias necesarias
pip install 'pydantic-ai-slim[openai]'  # Añade el proveedor
```

---

## 10. Checklist de inicio

Antes de continuar al Módulo 1, verifica:

- [ ] Python 3.10+ instalado
- [ ] `pydantic-ai` instalado correctamente
- [ ] Al menos una API key configurada (OpenAI/Anthropic/Gemini)
- [ ] `hello_agent.py` ejecuta sin errores
- [ ] Entiendes qué es un Agent y para qué sirve
- [ ] (Opcional) Logfire configurado y funcionando

---

## 11. Ejercicios prácticos

### Nivel 1: Básico

1. **Multimodelo**: Crea un script que ejecute el mismo prompt en OpenAI, Anthropic y Gemini, y compare las respuestas.

2. **Traductor**: Crea un agente que traduzca texto a 3 idiomas diferentes usando salida estructurada:
   ```python
   class Translation(BaseModel):
       spanish: str
       french: str
       german: str
   ```

3. **Calculadora conversacional**: Agente con tools para sumar, restar, multiplicar y dividir.

### Nivel 2: Intermedio

4. **Extractor de información**: Crea un agente que extraiga datos estructurados de texto no estructurado:
   ```python
   class PersonInfo(BaseModel):
       name: str
       age: int
       occupation: str
       location: str
   
   # Entrada: "Juan tiene 30 años, es ingeniero y vive en Madrid"
   ```

5. **Selector de modelo**: Crea una función que seleccione automáticamente el mejor modelo según el tipo de tarea (creatividad → Claude, análisis → GPT-4, velocidad → Gemini Flash).

6. **Contador de tokens**: Implementa un wrapper que cuente tokens y costos de cada ejecución.

### Nivel 3: Avanzado

7. **Mini-framework**: Crea una clase base `BaseAgent` con logging automático, manejo de errores y reintentos.

8. **Comparador A/B**: Sistema que ejecuta el mismo prompt en 2 modelos diferentes y guarda estadísticas (tiempo, tokens, calidad).

9. **Agente con memoria**: Implementa un agente que mantenga contexto entre múltiples ejecuciones (sin usar conversation history del módulo siguiente).

---

## 12. Recursos adicionales

### Documentación oficial

- **Sitio principal**: [ai.pydantic.dev](https://ai.pydantic.dev)
- **API Reference**: [ai.pydantic.dev/api/agent/](https://ai.pydantic.dev/api/agent/)
- **Ejemplos oficiales**: [ai.pydantic.dev/examples/](https://ai.pydantic.dev/examples/)
- **llms.txt**: [ai.pydantic.dev/llms.txt](https://ai.pydantic.dev/llms.txt)

### Comunidad

- **GitHub**: [github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)
- **Slack**: [logfire.pydantic.dev/docs/join-slack/](https://logfire.pydantic.dev/docs/join-slack/)
- **Documentación Pydantic**: [docs.pydantic.dev](https://docs.pydantic.dev)

### Video tutoriales

- Tutorial oficial de PydanticAI (YouTube)
- FastAPI + PydanticAI integration
- Building production agents with PydanticAI

---

## 13. Próximos pasos

En el **Módulo 1 - Agentes Básicos** aprenderás:

- Prompting avanzado con instrucciones dinámicas
- Salidas estructuradas complejas con validación
- Creación de herramientas/tools personalizadas
- Manejo de errores y validación con reflection
- Dependency injection para contexto compartido

---

## Referencias

1. Pydantic AI - Página principal: https://ai.pydantic.dev
2. Installation Guide: https://ai.pydantic.dev/install/
3. Agents Documentation: https://ai.pydantic.dev/agents/
4. Models & Providers: https://ai.pydantic.dev/models/overview/
5. Function Tools: https://ai.pydantic.dev/tools/
6. Structured Output: https://ai.pydantic.dev/output/
7. Examples: https://ai.pydantic.dev/examples/
8. Pydantic Logfire: https://pydantic.dev/logfire

---

**¡Estás listo para comenzar tu viaje con PydanticAI! 🚀**

> 💡 **Tip**: Ejecuta todos los ejemplos de este módulo antes de continuar. La mejor forma de aprender es practicando.