# Módulo 0 · Introducción a PydanticAI

**PydanticAI** es un framework Python para construir aplicaciones GenAI con validación de tipos y observabilidad integrada. Desarrollado por el equipo de Pydantic, alcanzó V1 en septiembre 2025 con política de estabilidad de API.

**Versión actual:** PydanticAI 1.10+ | Pydantic 2.12 | Python ≥3.11

---

## ¿Por qué PydanticAI?

Antes de empezar con el código, entendamos qué hace especial a PydanticAI y por qué lo usamos en este curso:

- **Type-safe**: Validación Pydantic nativa para entradas/salidas — el framework valida automáticamente que las respuestas del LLM cumplan el esquema que defines
- **Model-agnostic**: OpenAI, Anthropic, Gemini, Groq, Mistral, Ollama y compatibles OpenAI — cambiar de proveedor es cambiar una línea de código
- **Observability**: Integración directa con Pydantic Logfire — monitorea tus agentes en producción sin configuración adicional
- **Pythonic**: Control de flujo nativo sin DSLs — escribes Python normal, no aprendes un lenguaje nuevo

En este módulo vas a instalar el entorno, configurar tus primeros agentes y entender los conceptos fundamentales. Al final tendrás agentes funcionales que ejecutarás con `uv run python`.

---

## Prerrequisitos

Antes de comenzar, asegúrate de tener:

- Python 3.11 o superior (recomendado 3.12 para mejor rendimiento)
- Conocimientos de Python: async/await, type hints básicos, y familiaridad con Pydantic
- Una API key de OpenAI o Anthropic (puedes obtenerlas gratis para pruebas)

Si no estás familiarizado con Pydantic, no te preocupes: iremos explicando los conceptos según aparezcan.

---

## 1. Setup del proyecto

Vamos a configurar el entorno de trabajo. Usaremos **uv**, un gestor de paquetes Python ultrarrápido que reemplaza pip y poetry con un rendimiento significativamente superior.

### 1.1. Instalar uv

Si no tienes **uv** instalado en tu sistema, vamos a instalarlo ahora. Elige el comando según tu sistema operativo:

**En Linux o macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**En Windows (PowerShell como administrador):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Después de instalarlo, cierra y vuelve a abrir tu terminal para que los cambios surtan efecto. Verifica que funciona ejecutando `uv --version`.

---

### 1.2. Inicializar proyecto

Ahora vamos a crear la estructura del proyecto. Estos comandos crean el directorio principal y los subdirectorios para cada módulo del curso:

```bash
mkdir pydanticai-course && cd pydanticai-course
uv init --name pydanticai-course
mkdir 00-introduccion 01-agentes-basicos 02-contexto-validacion 03-integracion-llms
```

Este comando `uv init` crea la estructura base del proyecto con `pyproject.toml` y otros archivos necesarios.

---

### 1.3. Dependencias y configuración

Vamos a instalar las dos dependencias principales del curso:

```bash
uv add pydantic-ai pydantic-settings
```

**¿Qué hace cada paquete?**
- `pydantic-ai`: El framework de agentes que usaremos
- `pydantic-settings`: Gestión de configuración y variables de entorno (para las API keys)

Ahora configuremos el archivo `pyproject.toml` para que Python encuentre correctamente nuestro código:

**`pyproject.toml`:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pydanticai-course"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic-ai>=1.9.0",
    "pydantic-settings>=2.11.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
```

Finalmente, instala el proyecto en modo editable para que los cambios que hagas se reflejen inmediatamente:

```bash
uv pip install -e .
```

---

**Configuración de API keys:**

Para que tus agentes puedan comunicarse con los LLMs, necesitan las credenciales de acceso. Vamos a guardarlas de forma segura en un archivo `.env` que **nunca** subirás a GitHub.

Crea un archivo **`.env`** en la raíz del proyecto:

```bash
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-..."
```

⚠️ **Importante**: Añade `.env` a tu `.gitignore` para no subir las keys accidentalmente.

---

**Crear el archivo de configuración:**

Ahora vamos a crear un archivo Python que lea estas variables de entorno de forma tipada y segura:

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

**¿Qué hace este código?**
- Define una clase `Settings` que espera dos API keys opcionales
- Pydantic Settings las carga automáticamente desde `.env`
- `case_sensitive=False` permite escribir las variables en mayúsculas o minúsculas
- La instancia `settings` estará disponible en todo tu código

Este patrón es estándar en aplicaciones Python profesionales y te protege de errores comunes con variables de entorno.

---

## 2. Conceptos fundamentales

Antes de escribir código, entendamos los bloques de construcción de PydanticAI. Estos conceptos aparecerán constantemente en el curso.

### Agent (Agente)

Un `Agent` es la pieza central de PydanticAI. Encapsula toda la lógica necesaria para interactuar con un LLM:

- **Modelo LLM**: Qué modelo usar (GPT-5, Claude Sonnet, etc.)
- **Instrucciones**: El "system prompt" que define el comportamiento del agente
- **Tools/funciones**: Capacidades adicionales que puede ejecutar (consultar bases de datos, llamar APIs, etc.)
- **Esquema de salida**: Formato estructurado de la respuesta (opcional pero recomendado)
- **Dependencias**: Recursos externos que el agente necesita (bases de datos, configuración, etc.)

Piensa en un Agent como un "empleado especializado" que tiene instrucciones claras, herramientas específicas y un formato de reporte definido.

---

### Model + Provider

PydanticAI usa un patrón explícito `Model(..., provider=Provider(...))` que te da control total sobre cómo se conecta al LLM. Esto permite:

- **Pasar API keys programáticamente** sin variables de entorno globales
- **Personalizar clientes HTTP** para configuraciones corporativas (proxies, timeouts, etc.)
- **Configurar endpoints custom** si usas modelos auto-hospedados

**Ejemplo básico del patrón:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# Configuración explícita del modelo y proveedor
model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

# El agente usa este modelo configurado
agent = Agent(model)
```

Este patrón puede parecer verboso al principio, pero te da flexibilidad máxima en producción.

---

## 3. Hello World

Ahora sí, vamos a crear tu primer agente funcional. Este ejemplo mínimo te muestra la estructura básica que seguirás en todo el curso.

**¿Qué vamos a lograr?**  
Un agente que responde preguntas de forma concisa. Le preguntaremos sobre el origen de "Hello World" y veremos cómo procesa la consulta.

**`00-introduccion/hello_agent.py`:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

# Configurar modelo y proveedor
model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

# Crear agente con instrucciones simples
agent = Agent(
    model,
    instructions="Responde de forma concisa en una frase."
)

# Ejecutar una consulta
try:
    result = agent.run_sync('¿De dónde viene "Hello World"?')
    print(result.output)
except Exception as e:
    print(f"Error: {e}")
```

**Ejecuta el ejemplo:**

```bash
uv run python 00-introduccion/hello_agent.py
```

**¿Qué está pasando aquí?**
1. Importamos las clases necesarias y la configuración
2. Creamos un modelo Claude Sonnet con nuestro API key
3. Creamos un agente con instrucciones para respuestas concisas
4. Usamos `run_sync()` para hacer la consulta de forma síncrona
5. El resultado viene en `result.output` como un string

**Deberías ver** una respuesta tipo: _"'Hello World' se popularizó con el libro 'The C Programming Language' de Kernighan y Ritchie en 1978."_

Este es el flujo básico de cualquier agente en PydanticAI: configurar → instruir → ejecutar → obtener resultado.

---

## 4. Configuración de proveedores

PydanticAI soporta múltiples proveedores de LLM. Vamos a ver los dos principales que usaremos en el curso y cómo configurarlos correctamente.

### Anthropic Claude (recomendado para agentes)

Claude sobresale en tareas que requieren razonamiento, seguir instrucciones complejas y mantener coherencia en conversaciones largas. Es nuestra opción principal para ejemplos empresariales.

**Modelos disponibles (noviembre 2025):**
- `claude-sonnet-4-5-20250929` - **Recomendado**: equilibrio perfecto entre inteligencia y velocidad
- `claude-opus-4-1` - Máxima capacidad para problemas muy complejos
- `claude-haiku-4-5` - El más rápido, ideal para tareas simples

**Configuración:**

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)
```

**Cuándo usar cada modelo:**
- **Sonnet 4.5**: Tu opción por defecto para el 90% de los casos
- **Opus 4.1**: Análisis financiero complejo, generación de código crítico, razonamiento multi-paso
- **Haiku 4.5**: Clasificación simple, extracción de datos, respuestas rápidas

---

### OpenAI GPT

GPT es versátil y está optimizado para código. Si tu caso de uso involucra mucha generación o análisis de código, GPT puede ser mejor opción.

**Modelos disponibles (noviembre 2025):**
- `gpt-5` - Modelo unificado con capacidades de reasoning integradas
- `gpt-5-mini` - Versión eficiente, excelente relación calidad-precio
- `gpt-5-codex` - Especializado en tareas de programación

**Configuración:**

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)
```

**Cuándo usar GPT vs Claude:**
- **Claude**: Análisis de negocio, generación de contenido, seguir instrucciones complejas
- **GPT**: Generación de código, APIs REST, debugging, refactoring

---

### Otros proveedores

PydanticAI también soporta **Gemini, Groq, Mistral, Cohere** y cualquier proveedor compatible con OpenAI API (OpenRouter, Together, Fireworks, Ollama local, etc.). 

Para configuraciones de estos proveedores, consulta la [documentación oficial](https://ai.pydantic.dev/models/overview/).

---

## 5. Ejemplos prácticos

Ahora que entiendes los conceptos básicos, vamos a explorar las capacidades principales de PydanticAI con ejemplos ejecutables. Cada ejemplo demuestra una feature clave que usarás en proyectos reales.

### 5.1. Salida estructurada

**¿Por qué es importante?**  
En aplicaciones de producción, no quieres texto libre que luego tienes que parsear. Quieres datos estructurados que puedes validar y usar directamente en tu código.

**¿Qué vamos a lograr?**  
Un agente que extrae información de una ciudad y la devuelve en un formato estructurado (nombre, país, población, highlights).

**`00-introduccion/structured_output.py`:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

# Define el esquema de salida
class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    highlights: list[str] = Field(max_length=3)

# Configura el modelo
model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

# Crea el agente con output_type
agent = Agent(
    model,
    output_type=CityInfo,  # ¡Aquí está la magia!
    instructions="Extrae info de la ciudad mencionada."
)

# Ejecuta y obtén datos estructurados
city = agent.run_sync("Cuéntame sobre Barcelona").output

# Ahora tienes un objeto Python tipado
print(f"Ciudad: {city.name}")
print(f"País: {city.country}")
print(f"Población: {city.population:,}")
print(f"Destacados:")
for highlight in city.highlights:
    print(f"  - {highlight}")
```

**Ejecuta:**
```bash
uv run python 00-introduccion/structured_output.py
```

**¿Qué acabas de ver?**
- Definiste un esquema con Pydantic (`CityInfo`)
- PydanticAI automáticamente instruyó al LLM para devolver JSON que cumpla ese esquema
- El LLM generó la respuesta estructurada
- Pydantic validó que cumple el esquema (ej: población es int, máximo 3 highlights)
- Recibiste un objeto Python tipado que tu IDE autocompleta

**Beneficios en producción:**
- ✅ Sin parsing manual de strings
- ✅ Validación automática de tipos
- ✅ Errores claros si el LLM genera datos inválidos
- ✅ Código type-safe que tu IDE entiende

---

### 5.2. Tools

**¿Qué son las tools?**  
Las tools permiten que el agente ejecute código Python cuando necesita información que no tiene. Por ejemplo: consultar una base de datos, llamar una API, calcular algo preciso, o leer la hora actual.

**¿Qué vamos a lograr?**  
Un agente que puede responder "¿qué hora es?" ejecutando código Python real para obtener la hora del sistema.

**`00-introduccion/agent_with_tools.py`:**

```python
from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

# Configura el modelo
model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(model)

# Define una tool con el decorador @agent.tool
@agent.tool
def get_current_time(ctx: RunContext[None]) -> str:
    """Devuelve la hora actual del sistema."""
    return datetime.now().strftime("%H:%M:%S")

# El agente decide automáticamente cuándo usar la tool
print(agent.run_sync("¿Qué hora es?").output)
```

**Ejecuta:**
```bash
uv run python 00-introduccion/agent_with_tools.py
```

**¿Qué está pasando?**
1. Le preguntas al agente "¿Qué hora es?"
2. El agente **analiza** la pregunta y **decide** que necesita la tool `get_current_time`
3. El agente **ejecuta** tu función Python
4. El agente **usa** el resultado para formular su respuesta

**Salida esperada**: _"Son las 14:23:45"_ (con la hora real de tu sistema)

**Puntos clave:**
- El docstring de la tool es **crucial**: el LLM lo lee para decidir cuándo usarla
- `RunContext[None]` es obligatorio (lo veremos más en Módulo 1)
- El agente puede llamar múltiples tools en una sola consulta
- Las tools pueden devolver cualquier tipo de dato (str, dict, list, objetos Pydantic, etc.)

---

### 5.3. Instrucciones dinámicas

**¿Por qué dinámicas?**  
A veces las instrucciones del agente deben cambiar según el contexto: quién es el usuario, qué hora es, qué permisos tiene, etc.

**¿Qué vamos a lograr?**  
Un agente que personaliza su comportamiento según quién le habla, usando el sistema de dependencias (`deps`).

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

# deps_type indica qué tipo de dependencias espera el agente
agent = Agent(model, deps_type=str)

# El decorador @agent.instructions hace las instrucciones dinámicas
@agent.instructions
def get_instructions(ctx: RunContext[str]) -> str:
    # ctx.deps contiene la dependencia que pasamos en run_sync
    return f"Eres un asistente. El usuario es {ctx.deps}."

# Pasamos el nombre del usuario como deps
result = agent.run_sync("¿Cuál es tu función?", deps="María")
print(result.output)
```

**Ejecuta:**
```bash
uv run python 00-introduccion/dynamic_instructions.py
```

**¿Qué hace diferente?**
- Las instrucciones se generan **en tiempo de ejecución**
- Usas `deps` para inyectar contexto (usuario, configuración, conexión DB, etc.)
- El agente personaliza su comportamiento sin cambiar código

**Casos de uso empresariales:**
- Permisos por rol (admin ve más datos que empleado)
- Tono según cliente (formal para bancos, casual para startups)
- Límites por plan de suscripción

---

### 5.4. Streaming

**¿Qué es streaming?**  
En lugar de esperar toda la respuesta del LLM, recibes el texto "palabra por palabra" a medida que se genera. Esto mejora enormemente la experiencia de usuario.

**¿Qué vamos a lograr?**  
Un agente que responde progresivamente, mostrando texto en tiempo real como ChatGPT.

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
    # run_stream devuelve un async context manager
    async with agent.run_stream("Explica async/await en Python") as result:
        # Itera sobre los fragmentos de texto
        async for text in result.stream_text():
            print(text, end="", flush=True)
    print()  # Nueva línea al final

# Ejecuta la función async
asyncio.run(stream_response())
```

**Ejecuta:**
```bash
uv run python 00-introduccion/streaming_example.py
```

**¿Qué está pasando?**
- `run_stream()` inicia una respuesta en modo streaming
- `stream_text()` devuelve fragmentos de texto a medida que llegan
- `print(..., end="", flush=True)` muestra el texto sin saltos de línea

**Deberías ver** el texto aparecer progresivamente como si alguien estuviera escribiendo.

**Cuándo usar streaming:**
- ✅ Interfaces de chat (Gradio, Streamlit, web)
- ✅ Respuestas largas donde el usuario necesita feedback inmediato
- ❌ Procesamiento batch donde no hay usuario esperando
- ❌ Cuando necesitas la respuesta completa antes de continuar

---

**Ejecutar todos los ejemplos:**

Una vez creados los 4 archivos, pruébalos en orden:

```bash
uv run python 00-introduccion/structured_output.py
uv run python 00-introduccion/agent_with_tools.py
uv run python 00-introduccion/dynamic_instructions.py
uv run python 00-introduccion/streaming_example.py
```

Estos 4 ejemplos cubren los patrones fundamentales que usarás constantemente: salida estructurada, tools, instrucciones dinámicas y streaming.

---

## 6. Observabilidad con Logfire (opcional)

**¿Qué es Logfire?**  
Pydantic Logfire es una plataforma de observabilidad diseñada específicamente para aplicaciones con LLMs. Te permite ver en tiempo real:

- Qué prompts se enviaron
- Cuántos tokens consumiste
- Qué tools se llamaron y en qué orden
- Cuánto tiempo tardó cada operación
- Errores y excepciones

**¿Por qué es opcional?**  
Para aprender PydanticAI no lo necesitas. Pero en producción es **invaluable** para debugging y optimización.

**Configurar (solo la primera vez):**

Desde tu terminal, ejecuta:

```bash
uv run logfire auth
```

Esto abrirá tu navegador para autenticar con tu cuenta de Pydantic (es gratis para proyectos pequeños).

**Usar en tu código:**

```python
import logfire

# Configura una sola vez al inicio de tu aplicación
logfire.configure()

# Instrumenta PydanticAI automáticamente
logfire.instrument_pydantic_ai()

# Todos tus agentes quedan monitoreados automáticamente
agent = Agent(model)
```

Una vez configurado, cada vez que ejecutes un agente podrás ver las trazas en [logfire.pydantic.dev](https://logfire.pydantic.dev) con información detallada de tokens, tiempos y llamadas a tools.

**En este curso** no usaremos Logfire en los ejemplos para mantenerlos simples, pero es altamente recomendado que lo pruebes por tu cuenta.

---

## 7. Troubleshooting

Problemas comunes y sus soluciones:

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `No API key found` | El archivo `.env` no existe o está mal ubicado | Verifica que `.env` está en la raíz del proyecto y contiene las keys correctas |
| `Module 'config' not found` | No ejecutaste con `uv run` | Siempre ejecuta: `uv run python script.py` |
| `Python version too old` | Usas Python < 3.11 | Ejecuta: `uv python install 3.12 && uv python pin 3.12` |
| `Model not found` | Nombre de modelo incorrecto | Verifica que usas los nombres exactos de la sección 4 |
| `Rate limit error` | Demasiadas llamadas a la API | Espera 1 minuto o usa una API key con más cuota |

**Consejo general**: Si algo falla, verifica primero:
1. ¿Ejecutaste con `uv run python`?
2. ¿Existe el archivo `.env` con las keys correctas?
3. ¿Está el archivo en el directorio correcto del módulo?

---

## 8. Ejercicios

Ahora que has visto los ejemplos, es tu turno de practicar. Estos ejercicios refuerzan los conceptos del módulo con casos de uso empresariales.

---

## 1) Respuesta de atención al cliente (tono y brevedad)

**Archivo:** `00-introduccion/ej01_atencion_cliente.py`  
**Tiempo estimado:** 10 minutos

**Objetivo:** Configura un agente que devuelva una respuesta **en 2 frases máximo** a una queja de cliente con tono profesional y empático.

**Conceptos que practicas:** `instructions`, `run_sync`, patrón `Model(..., provider=Provider(...))`.

**Starter code:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel as Model
from pydantic_ai.providers.openai import OpenAIProvider as Provider
from config import settings

model = Model("gpt-5", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(
    model, 
    instructions="Responde como agente de soporte: tono empático, 2 frases máximo, ofrece siguiente paso claro."
)

print(agent.run_sync("El envío llegó dañado y nadie responde.").output)
```

**Criterio de aceptación:** 
1. Máximo 2 frases en la respuesta
2. Tono profesional y empático
3. Incluye un siguiente paso concreto (ej: abrir ticket, reembolso, contacto)

---

## 2) Cualificación rápida de leads (salida estructurada)

**Archivo:** `00-introduccion/ej02_lead_scoring.py`  
**Tiempo estimado:** 15 minutos

**Objetivo:** Extrae de un mensaje libre los campos `company`, `need`, `budget_eur` (int, opcional) y `fit` ∈ {low, medium, high}.

**Conceptos que practicas:** `output_type` con `BaseModel`, validación de campos con `Field()`.

**Starter code:**

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
agent = Agent(
    model, 
    output_type=Lead, 
    instructions="Extrae y normaliza campos para cualificación de lead."
)

lead = agent.run_sync("Somos RetailNova, buscamos forecast de demanda, presupuesto ~15k€.").output
print(lead.model_dump())
```

**Criterio de aceptación:** 
1. Devuelve un dict con las 4 claves exactas
2. Valida que `fit` solo puede ser 'low', 'medium' o 'high'
3. `budget_eur` es opcional (puede ser `None`)

---

## 3) Tool de negocio: cálculo de IVA y total

**Archivo:** `00-introduccion/ej03_tool_iva_total.py`  
**Tiempo estimado:** 20 minutos

**Objetivo:** Implementa una tool `calc_total(lineas: list[float], con_impuestos: bool)` que sume las líneas, aplique IVA 21% si `con_impuestos=True`, y devuelva `{subtotal, iva, total}`.

**Conceptos que practicas:** `@agent.tool`, `RunContext`, tools simples con lógica de negocio.

**Starter code:**

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel as Model
from pydantic_ai.providers.anthropic import AnthropicProvider as Provider
from config import settings

model = Model("claude-haiku-4-5", provider=Provider(api_key=settings.anthropic_api_key))
agent = Agent(model)

@agent.tool
def calc_total(ctx: RunContext[None], lineas: list[float], con_impuestos: bool = True) -> dict:
    """Calcula el total de un carrito con o sin IVA (21%)."""
    subtotal = round(sum(lineas), 2)
    iva = round(subtotal * 0.21, 2) if con_impuestos else 0.0
    return {"subtotal": subtotal, "iva": iva, "total": round(subtotal + iva, 2)}

print(agent.run_sync("Carrito: [49.9, 10.0, 3.1]. Calcula total con IVA. Usa la tool.").output)
```

**Criterio de aceptación:** 
1. Devuelve los tres campos con 2 decimales
2. El agente **llama a la tool** (no calcula manualmente)
3. La respuesta menciona subtotal, IVA y total

---

## 4) Instrucciones dinámicas por marca (branding)

**Archivo:** `00-introduccion/ej04_branding_dinamico.py`  
**Tiempo estimado:** 15 minutos

**Objetivo:** Ajusta el tono según la marca pasada en `deps`: `"ICSO"` (directo y técnico) vs `"TeachTools"` (didáctico y cercano).

**Conceptos que practicas:** `deps_type`, `@agent.instructions`, personalización de comportamiento.

**Starter code:**

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

**Criterio de aceptación:** 
1. Las dos respuestas tienen **estilos claramente diferentes**
2. Ambas respetan el límite de 3 frases
3. ICSO suena técnico y directo; TeachTools suena didáctico y cercano

---

## 5) Manejo de errores y fallback elegante

**Archivo:** `00-introduccion/ej05_fallback_error.py`  
**Tiempo estimado:** 15 minutos

**Objetivo:** Si la llamada falla (ej: sin API key), captura la excepción y devuelve un mensaje corporativo con pasos de remedio.

**Conceptos que practicas:** `try/except`, UX de error profesional, manejo robusto.

**Starter code:**

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
    print(
        "No hemos podido procesar tu solicitud ahora. "
        "Verifica la API key y reintenta; si persiste, contacta soporte con el código: E-LLM-001."
    )
```

**Criterio de aceptación:** 
1. En caso de fallo, muestra un mensaje corporativo **sin traza técnica**
2. El mensaje incluye una **acción concreta** (verificar key, contactar soporte)
3. Incluye un **código de error** único para tracking interno

---

## 6) Smoke test de negocio (verificación mínima)

**Archivo:** `00-introduccion/ej06_smoke_negocio.py`  
**Tiempo estimado:** 10 minutos

**Objetivo:** Ejecuta un prompt fijo ("Redacta un asunto y 3 bullets para un email comercial sobre X") y verifica rápido: longitud < 120 palabras y contiene "Asunto:".

**Conceptos que practicas:** Prueba mínima "listo para demo", validación post-generación.

**Starter code:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel as Model
from pydantic_ai.providers.anthropic import AnthropicProvider as Provider
from config import settings

model = Model("claude-haiku-4-5", provider=Provider(api_key=settings.anthropic_api_key))
agent = Agent(
    model, 
    instructions="Devuelve: 'Asunto: ...' y 3 viñetas claras, máx. 120 palabras."
)

out = agent.run_sync("Redacta un asunto y 3 bullets para un email comercial sobre auditoría de datos.").output
ok = "Asunto:" in out and len(out.split()) < 120

print("✅ OK" if ok else "❌ FAIL")
print(out)
```

**Criterio de aceptación:** 
1. Imprime "✅ OK" si la respuesta cumple ambos criterios
2. Imprime "❌ FAIL" si no cumple
3. Muestra la propuesta generada en cualquier caso

---

**Consejo para los ejercicios**: No te preocupes si los primeros intentos no funcionan perfectamente. Iterar sobre las instrucciones del agente es parte normal del proceso. Experimenta con diferentes prompts hasta obtener el comportamiento deseado.

---

## 9. Próximos pasos

**¡Felicidades por completar el Módulo 0!** Ya sabes crear agentes básicos, configurar proveedores, estructurar salidas y usar tools simples.

**En el Módulo 1** aprenderás:
- **Agentes avanzados**: tools complejas con validación Pydantic de parámetros
- **Result validators**: garantizar calidad de las respuestas antes de devolverlas
- **Reflection patterns**: el agente revisa y mejora su propia salida
- **Retries configurables**: manejar errores y reintentos de forma inteligente
- **Dependency injection**: integrar bases de datos, APIs y servicios externos

**En el Módulo 2** cubrirás:
- Gestión de **contexto conversacional** y historial multi-turno
- **Streaming** avanzado para interfaces de usuario
- Conexión con **fuentes de datos reales** (CSV, bases de datos, APIs)

**En el Módulo 3** construirás:
- **Agentes jerárquicos** y workflows complejos
- **Integración con herramientas** externas (búsqueda web, generación de gráficos)
- **Despliegue en producción** con Docker, Gradio y monitoreo

Continúa con el [Módulo 1: Agentes Avanzados](../01-agentes-avanzados/README.md) cuando estés listo.

---

## Referencias

Documentación oficial que complementa este módulo:

- [PydanticAI Docs](https://ai.pydantic.dev) - Documentación principal
- [Models & Providers](https://ai.pydantic.dev/models/overview/) - Guía completa de modelos
- [API Reference](https://ai.pydantic.dev/api/agent/) - Referencia detallada de la API
- [Anthropic Claude Models](https://docs.claude.com/en/docs/about-claude/models/overview) - Especificaciones de Claude
- [OpenAI Models](https://platform.openai.com/docs/models/) - Especificaciones de GPT

---

**© 2025 Alfons Freixes | PydanticAI Course**