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

## 1. Configuración del proyecto paso a paso

Esta sección te guiará para crear tu entorno de desarrollo desde cero usando **uv**, el gestor de paquetes moderno de Python.

> 💡 **¿Por qué uv?** Es significativamente más rápido que pip, maneja automáticamente entornos virtuales, y simplifica la gestión de dependencias. Es la herramienta estándar que usaremos en todo el curso.

### 1.1. Instalar uv

Ejecuta el comando correspondiente a tu sistema operativo:

**Linux y macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell como administrador):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verificar instalación:**
```bash
uv --version
# Debe mostrar algo como: uv 0.5.x
```

Si no reconoce el comando, reinicia tu terminal o sesión.

### 1.2. Crear la estructura del proyecto

Ejecuta estos comandos en tu terminal (funcionan en Linux, macOS y Windows):

```bash
# Crear carpeta principal y entrar
mkdir pydanticai-course
cd pydanticai-course

# Inicializar proyecto con uv
uv init --name pydanticai-course

# Crear estructura de módulos del curso
mkdir 00-introduccion 01-agentes-basicos 02-contexto-validacion 03-integracion-llms
```

Tu estructura quedará así:

```
pydanticai-course/
├── pyproject.toml              # Configuración del proyecto
├── .python-version             # Versión de Python del proyecto
├── .gitignore                  # Generado por uv
├── README.md                   # Generado por uv
├── .env                        # Variables de entorno (lo crearemos)
├── config.py                   # Configuración con Pydantic Settings
├── 00-introduccion/            # Módulo 0: Introducción
├── 01-agentes-basicos/         # Módulo 1: Agentes básicos
├── 02-contexto-validacion/     # Módulo 2: Contexto y validación
└── 03-integracion-llms/        # Módulo 3: Integración con LLMs
```

### 1.3. Instalar dependencias

Desde la raíz del proyecto (`pydanticai-course/`):

```bash
# Instalar PydanticAI y pydantic-settings
uv add pydantic-ai pydantic-settings
```

Esto instalará:
- **pydantic-ai**: Framework core + todos los modelos soportados + Logfire
- **pydantic-settings**: Para manejar configuración con validación de tipos

### 1.4. Configurar variables de entorno

**Paso 1: Crear archivo .env**

En la raíz del proyecto, crea el archivo `.env` con tu editor favorito y pega este contenido:

```bash
# .env
# API Keys para diferentes proveedores LLM
ANTHROPIC_API_KEY=""
OPENAI_API_KEY=""
GOOGLE_API_KEY=""
GROQ_API_KEY=""
```

**Paso 2: Agregar tus API keys**

Edita el archivo `.env` y agrega al menos una API key:

```bash
# .env
ANTHROPIC_API_KEY="sk-ant-tu-clave-real-aqui"
OPENAI_API_KEY="sk-tu-clave-real-aqui"
GOOGLE_API_KEY=""
GROQ_API_KEY=""
```

> 💡 **¿Dónde conseguir API keys?**
> - **Anthropic Claude**: [console.anthropic.com](https://console.anthropic.com) - Créditos gratis para empezar
> - **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
> - **Google Gemini**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) - Gratis
> - **Groq**: [console.groq.com/keys](https://console.groq.com/keys) - Gratis y ultra rápido

### 1.5. Crear configuración con Pydantic Settings

Crea el archivo `config.py` en la raíz del proyecto con tu editor de texto preferido:

```python
"""
Configuración del proyecto usando Pydantic Settings.
Las variables se cargan automáticamente desde el archivo .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración de API keys para diferentes proveedores LLM.
    
    Pydantic Settings:
    - Carga automáticamente desde .env
    - Valida tipos automáticamente
    - Proporciona errores claros si falta alguna variable requerida
    """
    
    # API Keys (opcional = None permite que no estén todas configuradas)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,  # ANTHROPIC_API_KEY = anthropic_api_key
        extra='ignore'  # Ignorar variables extra en .env
    )
    
    def get_configured_providers(self) -> list[str]:
        """Retorna lista de proveedores con API key configurada."""
        providers = []
        if self.anthropic_api_key:
            providers.append("Anthropic")
        if self.openai_api_key:
            providers.append("OpenAI")
        if self.google_api_key:
            providers.append("Google")
        if self.groq_api_key:
            providers.append("Groq")
        return providers


# Instancia global de configuración
settings = Settings()
```

### 1.6. Verificar la instalación

Crea un script de verificación `verify_setup.py` en la raíz del proyecto:

```python
"""Script de verificación del setup del proyecto."""
from config import settings

print("🔍 Verificando instalación...\n")

# Verificar PydanticAI
try:
    import pydantic_ai
    print(f"✅ PydanticAI instalado: v{pydantic_ai.__version__}")
except ImportError:
    print("❌ PydanticAI no está instalado")
    exit(1)

# Verificar Pydantic Settings
try:
    import pydantic_settings
    print(f"✅ Pydantic Settings instalado: v{pydantic_settings.__version__}")
except ImportError:
    print("❌ Pydantic Settings no está instalado")
    exit(1)

# Verificar configuración
print("\n📋 API Keys configuradas:")
providers = settings.get_configured_providers()

if not providers:
    print("⚠️  No hay API keys configuradas")
    print("\nℹ️  Edita el archivo .env y agrega al menos una API key")
    exit(1)

for provider in providers:
    print(f"✅ {provider}")

print(f"\n🎉 ¡Setup completado! Puedes usar {len(providers)} proveedor(es).")
```

**Ejecutar verificación:**

```bash
uv run python verify_setup.py
```

**Salida esperada:**
```
🔍 Verificando instalación...

✅ PydanticAI instalado: v1.8.0
✅ Pydantic Settings instalado: v2.6.0

📋 API Keys configuradas:
✅ Anthropic
✅ OpenAI

🎉 ¡Setup completado! Puedes usar 2 proveedor(es).
```

### 1.7. Cómo ejecutar scripts

Siempre ejecuta tus scripts con `uv run`:

```bash
# Desde la raíz del proyecto
uv run python 00-introduccion/hello_agent.py
uv run python verify_setup.py
```

`uv run` automáticamente:
- Activa el entorno virtual
- Instala dependencias si faltan
- Ejecuta el script

---

## 2. Hello World con PydanticAI

Ahora que ya tienes todo configurado, vamos a crear tu primer agente.

### Crear el primer agente

Crea el archivo `00-introduccion/hello_agent.py`:

```python
"""
Primer agente con PydanticAI.
Demuestra el uso básico de un agente con instrucciones simples.
"""
from pydantic_ai import Agent
from config import settings

# Crear un agente con Claude Sonnet 4.5
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Responde de forma concisa en una sola frase.'
)

# Ejecutar el agente de forma síncrona
result = agent.run_sync('¿De dónde viene "Hello World"?')
print(result.output)
```

**Ejecutar el ejemplo:**

```bash
# Desde la raíz del proyecto
uv run python 00-introduccion/hello_agent.py
```

**Salida esperada:**
```
El primer uso conocido de "hello, world" fue en un libro de texto sobre el lenguaje C en 1974.
```

> 💡 **Nota**: El script importa `settings` de `config.py`, que automáticamente carga las variables de entorno del archivo `.env`. Si obtienes un error de API key, verifica que tu `.env` contiene la clave correcta.

---

## 3. Configuración de proveedores

> 💡 **Nota**: Todos los ejemplos asumen que ya tienes configurado tu archivo `.env` y `config.py`. Las API keys se cargan automáticamente desde ahí.

### OpenAI (GPT-5, O3, O4-mini)

```python
from pydantic_ai import Agent
from config import settings

# GPT-5 (modelo principal de OpenAI en 2025)
agent = Agent('openai:gpt-5')

# O3-mini (modelo de razonamiento rápido)
agent = Agent('openai:o3-mini')

# O4-mini (modelo de razonamiento eficiente)
agent = Agent('openai:o4-mini')
```

Requiere `OPENAI_API_KEY` en tu archivo `.env`.

### Anthropic (Claude 4 y Claude Sonnet 4.5)

```python
from pydantic_ai import Agent
from config import settings

agent = Agent('anthropic:claude-sonnet-4-5')  # Modelo más inteligente
agent = Agent('anthropic:claude-sonnet-4-0')  # Alternativa equilibrada
agent = Agent('anthropic:claude-opus-4-1')    # Máxima capacidad
```

Requiere `ANTHROPIC_API_KEY` en tu archivo `.env`.

### Google Gemini 2.5

```python
from pydantic_ai import Agent
from config import settings

agent = Agent('google-gla:gemini-2.5-flash')  # Ultra rápido y eficiente
agent = Agent('google-gla:gemini-2.5-pro')    # Mayor capacidad
```

Requiere `GOOGLE_API_KEY` en tu archivo `.env`.

### Groq (ultra rápido)

```python
from pydantic_ai import Agent
from config import settings

agent = Agent('groq:llama-3.3-70b-versatile')
```

Requiere `GROQ_API_KEY` en tu archivo `.env`.

### Ollama (local)

```python
from pydantic_ai import Agent

agent = Agent('ollama:llama3.2')
```

No requiere API key, pero necesitas tener [Ollama](https://ollama.com) instalado y corriendo localmente.

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

> 💡 **Nota**: Todos estos ejemplos asumen que ya tienes configurado `config.py` y tu archivo `.env` con al menos una API key. Las variables de entorno se cargan automáticamente.

### Ejemplo 1: Agente con instrucciones dinámicas

```python
from pydantic_ai import Agent, RunContext
from config import settings

agent = Agent('openai:gpt-5', deps_type=str)

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
from config import settings

agent = Agent('openai:gpt-5')

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
from config import settings

class CityInfo(BaseModel):
    """Información estructurada sobre una ciudad"""
    name: str = Field(description="Nombre de la ciudad")
    country: str = Field(description="País donde se encuentra")
    population: int = Field(description="Población aproximada")
    famous_for: str = Field(description="Por qué es famosa")

# Crear agente que devuelve datos estructurados
agent = Agent(
    'openai:gpt-5',
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
from config import settings

agent = Agent(
    'openai:gpt-5',
    instructions="Puedes decir la fecha y hora actual cuando te lo pidan."
)

@agent.tool
def get_current_time(ctx: RunContext[None]) -> str:
    """Obtiene la fecha y hora actual del sistema."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

result = agent.run_sync("¿Qué hora es?")
print(result.output)
# "La hora actual es 2025-11-02 14:30:45"
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
    model='openai:gpt-5',                   # Modelo a usar
    instructions='Eres un experto en...',   # Sistema prompt
    output_type=MiModelo,                   # Salida estructurada
    deps_type=MisDependencias,              # Tipo de dependencias
    tools=[tool1, tool2],                   # Herramientas disponibles
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

## 6. Observabilidad con Logfire (opcional pero recomendado)

Logfire es la plataforma de observabilidad de Pydantic. Te permite:
- Ver el flujo completo de conversaciones con tu agente
- Debuggear llamadas a tools en tiempo real
- Monitorear costos y consumo de tokens
- Analizar performance de tus agentes

### Instalación

Logfire ya viene incluido con la instalación completa de `pydantic-ai`:

```bash
# Si instalaste pydantic-ai (no slim), ya tienes Logfire
uv add pydantic-ai

# Si usaste la versión slim, agrégalo así:
uv add 'pydantic-ai-slim[logfire]'
```

### Configurar Logfire en tu proyecto

```bash
# Autenticar (solo la primera vez)
uv run logfire auth

# Seguir las instrucciones en el navegador
```

### Usar Logfire en tu código

```python
import logfire
from pydantic_ai import Agent
from config import settings

# Configurar Logfire (una vez al inicio)
logfire.configure()
logfire.instrument_pydantic_ai()

# Ahora todos tus agentes serán monitoreados automáticamente
agent = Agent('openai:gpt-5')
result = agent.run_sync('¿Qué es Python?')
```

### Ver tus traces

Visita [logfire.pydantic.dev](https://logfire.pydantic.dev) para ver:
- Cada llamada al LLM
- Tokens consumidos
- Tiempo de respuesta
- Contenido completo de cada interacción
- Llamadas a tools y sus resultados

> 💡 **Tip**: Logfire es invaluable para debugging y optimización. Te muestra exactamente qué está pasando en cada interacción con el LLM.

---

## 7. Comparación rápida con otros frameworks

| Característica | PydanticAI | LangChain | CrewAI |
|---------------|------------|-----------|---------|
| Type Safety | ✅ Completo | ⚠️ Parcial | ⚠️ Parcial |
| Validación | ✅ Pydantic | ⚠️ Manual | ⚠️ Manual |
| Observabilidad | ✅ Integrada | 🔧 Requiere setup | 🔧 Requiere setup |
| Curva aprendizaje | 🟢 Baja | 🟡 Media | 🟡 Media |
| Control flow | Python nativo | Chains/LCEL | Predefinido |

---

## 8. Troubleshooting común

### Error: "No API key found"

**Solución:**
1. Verifica que el archivo `.env` existe en la raíz del proyecto
2. Verifica que la API key está correctamente escrita (sin espacios extra)
3. Ejecuta el script de verificación:

```bash
uv run python verify_setup.py
```

### Error: "Python version too old"

```bash
# Verificar versión actual
python --version  # Debe ser 3.10+

# Con uv puedes instalar y usar una versión específica
uv python install 3.12
uv python pin 3.12

# Verificar que se aplicó
python --version
```

### Error: "Module 'pydantic_ai' not found"

**Solución:**
```bash
# Asegúrate de ejecutar con uv run
uv run python tu_script.py

# O instala de nuevo las dependencias
uv sync
```

### Error: "Settings validation error"

Si `config.py` falla al cargar:

```bash
# Verifica que pydantic-settings está instalado
uv add pydantic-settings

# Verifica que el .env tiene el formato correcto
cat .env  # Linux/macOS
type .env  # Windows
```

### El script no encuentra el archivo .env

Asegúrate de ejecutar los scripts desde la raíz del proyecto:

```bash
# ✅ Correcto (desde pydanticai-course/)
uv run python 00-introduccion/hello_agent.py

# ❌ Incorrecto (desde 00-introduccion/)
cd 00-introduccion
uv run python hello_agent.py  # No encontrará .env ni config.py
```

### Error: "command not found: uv"

Si instalaste uv pero no lo reconoce:

**Linux/macOS:**
```bash
# Agregar uv al PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Hacer permanente (añadir a ~/.bashrc o ~/.zshrc)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Windows:**
Reinicia la terminal o sesión de PowerShell después de instalar uv.

---

## 9. Checklist de inicio

Antes de continuar al Módulo 1, verifica que completaste todos estos pasos:

### Instalación y Configuración
- [ ] `uv` instalado y funcionando (`uv --version`)
- [ ] Proyecto creado con `uv init` en `pydanticai-course/`
- [ ] Estructura de carpetas creada con los 4 módulos:
  - [ ] `00-introduccion/`
  - [ ] `01-agentes-basicos/`
  - [ ] `02-contexto-validacion/`
  - [ ] `03-integracion-llms/`

### Dependencias
- [ ] `pydantic-ai` instalado (`uv add pydantic-ai`)
- [ ] `pydantic-settings` instalado (`uv add pydantic-settings`)

### Configuración
- [ ] Archivo `.env` creado en la raíz con al menos una API key
- [ ] Archivo `config.py` creado con la clase `Settings`
- [ ] Script `verify_setup.py` creado

### Validación
- [ ] `uv run python verify_setup.py` muestra ✅ para al menos un proveedor
- [ ] `uv run python 00-introduccion/hello_agent.py` ejecuta correctamente
- [ ] Entiendes qué es un Agent y para qué sirve
- [ ] Entiendes cómo funciona Pydantic Settings para cargar variables de entorno

### Opcional
- [ ] Logfire configurado y funcionando

Si todos los checks están completos, ¡estás listo para el Módulo 1: Agentes Básicos! 🚀

---

## 10. Ejercicios prácticos

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

## 11. Recursos adicionales

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

## 12. Próximos pasos

### Módulo 1 - Agentes Básicos

En el próximo módulo aprenderás:

- Construcción y ejecución de diferentes tipos de agentes
- Prompting avanzado con instrucciones dinámicas
- Salidas estructuradas complejas con validación
- Creación de herramientas/tools personalizadas
- Manejo de errores y validación con reflection
- Dependency injection para contexto compartido

### Módulo 2 - Contexto y Validación

- Gestión avanzada del contexto de conversación
- Validación robusta de entradas y salidas
- Manejo de estados y memoria entre ejecuciones
- Estrategias de validación con Pydantic

### Módulo 3 - Integración con LLMs

- Conexión y configuración de diferentes proveedores
- Optimización de prompts y parámetros
- Estrategias de despliegue en producción
- Monitoreo y observabilidad con Logfire

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