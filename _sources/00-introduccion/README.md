# Módulo 0 · Fundamentos y Filosofía

Bienvenido al curso. En este primer módulo no vamos a construir un "chat bot" simple; vamos a establecer los cimientos de una aplicación de IA lista para producción.

El objetivo es entender **por qué** usamos PydanticAI frente a otras librerías (como LangChain) y configurar un entorno de desarrollo profesional que escale.

**Stack tecnológico:** Python 3.12+ | PydanticAI | uv | Pydantic Settings | Ruff

---

## 1. La Filosofía: "Type-safe AI"

La mayoría de frameworks de agentes tratan los prompts y respuestas como cadenas de texto (`str`). Esto funciona para prototipos, pero en producción es una pesadilla de mantenimiento.

**PydanticAI** invierte este modelo:
1. **El Schema es la Ley:** Definimos modelos Pydantic estrictos para lo que entra y lo que sale.
2. **Validación en Runtime:** Si el LLM alucina un formato incorrecto, el framework lo captura y (opcionalmente) pide al LLM que corrija, sin que tu código explote.
3. **Python Nativo:** No hay DSLs (Domain Specific Languages) extraños. Es solo Python moderno con `async/await`.

---

## 2. Setup del Entorno Enterprise

Usaremos **uv** no solo porque es rápido, sino porque gestiona versiones de Python y entornos virtuales de forma determinista, algo crítico en entornos corporativos.

### 2.1. Inicialización del Proyecto

```bash
# Crear estructura base
mkdir pydanticai-course && cd pydanticai-course
uv init --name pydanticai-course --python 3.12

# Crear estructura de módulos para el curso
mkdir 00-introduccion 01-core-patterns 02-contexto 03-avanzado 04-produccion
```

### 2.2. Dependencias Core

Instalaremos las librerías necesarias para todo el curso. Observa que incluimos `devtools` para depuración visual.

```bash
# Dependencias principales
uv add pydantic-ai pydantic-settings devtools

# Dependencias de desarrollo (linter/formatter)
uv add --dev ruff
```

### 2.3. Configuración del Proyecto

**Actualiza tu `pyproject.toml`** para configurar el proyecto como paquete instalable y configurar Ruff:

```toml
[project]
name = "pydanticai-course"
version = "0.1.0"
description = "Curso completo de PydanticAI - De cero a producción"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "devtools>=0.12.2",
    "pydantic-ai>=1.24.0",
    "pydantic-settings>=2.12.0",
]

# Especificar módulos importables en la raíz
[tool.setuptools]
py-modules = ["config"]

# Dependencias de desarrollo
[dependency-groups]
dev = [
    "ruff>=0.14.6",
]

# ════════════════════════════════════════════════════════════
# 🔧 RUFF - Linter y Formatter
# ════════════════════════════════════════════════════════════

[tool.ruff]
line-length = 100
target-version = "py312"

exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "*.egg-info",
    ".ruff_cache",
    "build",
    "dist",
]

[tool.ruff.lint]
select = [
    "E",     # pycodestyle - errores
    "W",     # pycodestyle - warnings
    "F",     # pyflakes
    "I",     # isort - ordenar imports
    "UP",    # pyupgrade
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "SIM",   # flake8-simplify
]

ignore = [
    "E501",   # Línea demasiado larga
    "B008",   # Function calls en defaults (común en Pydantic)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"**/0*.py" = ["E402"]

[tool.ruff.lint.isort]
known-first-party = ["config", "utils"]
lines-after-imports = 2
combine-as-imports = true

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"
```

### 2.4. Configuración de VSCode

Crea el directorio `.vscode` y el archivo `settings.json` dentro:

```bash
mkdir .vscode
```

**.vscode/settings.json:**

```jsonc
{
  // ═══════════════════════════════════════════════════════
  // RUFF - Configuración para servidor nativo
  // ═══════════════════════════════════════════════════════
  "ruff.nativeServer": "on",

  // ═══════════════════════════════════════════════════════
  // PYTHON - Configuración general
  // ═══════════════════════════════════════════════════════
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.rulers": [100]
  },

  "python.linting.enabled": false,
  "python.analysis.typeCheckingMode": "basic",

  // ═══════════════════════════════════════════════════════
  // FILES - Limpieza visual
  // ═══════════════════════════════════════════════════════
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.ruff_cache": true,
    "**/*.egg-info": true
  },

  // ═══════════════════════════════════════════════════════
  // EDITOR - Mejoras generales
  // ═══════════════════════════════════════════════════════
  "editor.formatOnSaveMode": "file",
  "files.insertFinalNewline": true,
  "files.trimTrailingWhitespace": true
}
```

### 2.5. Gestión de Secretos (The Right Way)

Nunca hardcodeamos claves. Usaremos `pydantic-settings` para cargar configuración de forma tipada. Esto nos permite fallar rápido si falta una variable crítica.

**1. Crea el archivo `.env` en la raíz:**

```bash
OPENAI_API_KEY=sk-tu-clave-aqui
# Opcional: LOGFIRE_TOKEN=...
```

**2. Añádelo a `.gitignore`:**

```bash
# Crea o actualiza .gitignore
cat >> .gitignore << EOF
# Secrets
.env
.env.local

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/

# Ruff
.ruff_cache/

# IDE
.vscode/
.idea/
EOF
```

**3. Crea `config.py` en la raíz:**

Este archivo actuará como nuestro "Singleton" de configuración.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    # Configuración para cargar .env automáticamente
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )


# Instancia global validada al importar
settings = Settings()
```

### 2.6. Instalar el Proyecto en Modo Editable

Este paso es **crucial** para que los imports funcionen correctamente desde cualquier script:

```bash
# Instalar el proyecto en modo editable
uv pip install -e .

# Verificar que funciona
uv run python -c "from config import settings; print('✅ Config cargado correctamente')"
```

---

## 3. El Modelo Mental: Agent, Model & Context

Antes de escribir código, entiende las tres piezas clave:

1. **Model:** La "conexión tonta" con el proveedor (OpenAI, Anthropic). Solo envía y recibe texto/tokens.
2. **Agent:** El "cerebro". Contiene las instrucciones (System Prompt), gestiona el historial, reintentos y herramientas.
3. **RunContext:** El "estado". Inyecta dependencias (DBs, APIs) en el momento de la ejecución. (Veremos esto a fondo en el Módulo 1).

---

## 4. Hello World Enterprise: Clasificador de Tickets

En lugar de preguntar "Hola mundo", vamos a crear un componente útil: un clasificador de tickets de soporte que devuelve **JSON estructurado**, no texto.

Crea el archivo `00-introduccion/ticket_classifier.py`:

```python
import asyncio
from typing import Literal

from devtools import debug
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import settings


# 1. DEFINICIÓN DEL DOMINIO (Type-safety)
# Definimos estrictamente qué esperamos recibir. El LLM NO puede salirse de aquí.
class SupportTicketParams(BaseModel):
    category: Literal["technical", "billing", "access", "feature_request"]
    urgency_score: int = Field(
        ge=1, le=10, 
        description="1 es trivial, 10 es crítico/bloqueante"
    )
    sentiment: Literal["angry", "neutral", "happy"]
    summary: str = Field(
        description="Resumen ejecutivo del problema en máximo 10 palabras"
    )


# 2. CONFIGURACIÓN DEL AGENTE
# Configuramos el modelo inyectando el Provider explícitamente con nuestra config.
model = OpenAIChatModel(
    "gpt-4o-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=SupportTicketParams,
    system_prompt=(
        "Eres un sistema experto de triaje (triage) para soporte técnico. "
        "Analiza el mensaje del usuario, extrae la intención y clasifícalo estrictamente. "
        "Sé conservador con la urgencia: solo problemas que detienen el negocio son 9 o 10."
    ),
)


# 3. EJECUCIÓN ASÍNCRONA
async def main():
    # Simulamos un input de un usuario real
    user_input = (
        "¡No puedo entrar al sistema! Me da error 500 todo el rato y tengo "
        "que presentar los impuestos hoy o me multan. ¡Arregladlo YA!"
    )

    print(f"📥 Input Usuario: {user_input}\n")

    # Ejecutamos el agente. PydanticAI gestiona la validación del JSON por debajo.
    result = await agent.run(user_input)

    # 'result.output' ya es una instancia de nuestra clase SupportTicketParams
    # ¡Tenemos autocompletado en el IDE!
    print("📤 Resultado Estructurado:")
    debug(result.output)

    # Lógica de negocio basada en tipos (imposible con strings planos)
    if result.output.urgency_score >= 9 and result.output.category == "technical":
        print("\n🚨 ALERTA: Escalando a equipo de guardia inmediatamente.")


if __name__ == "__main__":
    asyncio.run(main())
```

### Ejecuta el código:

```bash
# Desde la raíz del proyecto
uv run python 00-introduccion/ticket_classifier.py
```

**Salida esperada:**

```
📥 Input Usuario: ¡No puedo entrar al sistema! Me da error 500 todo el rato y tengo que presentar los impuestos hoy o me multan. ¡Arregladlo YA!

📤 Resultado Estructurado:
00-introduccion\ticket_classifier.py:54 main                                                                                                                                                                                         
    result.output: SupportTicketParams(
        category='access',
        urgency_score=10,
        sentiment='angry',
        summary='Error 500 al intentar acceder al sistema',
    ) (SupportTicketParams)
```

### ¿Qué acabamos de ver?

1. **Abstracción del Prompt Engineering:** No tuvimos que decirle "Devuelve JSON" o "No incluyas markdown". PydanticAI inyectó el esquema de `SupportTicketParams` en el prompt del sistema automáticamente.

2. **Validación Robusta:** `urgency_score` tiene un validador `ge=1, le=10`. Si el LLM devolviera `100`, PydanticAI capturaría el error, le enviaría el error de vuelta al LLM y reintentaría automáticamente.

3. **Developer Experience:** El objeto `result.output` es puro Python tipado. Puedes usar `result.output.category` y tu IDE sabe que es uno de los 4 literales definidos. Ruff te advertirá si intentas compararlo con un valor inválido.

4. **Imports Automáticos:** Gracias a la instalación en modo editable (`uv pip install -e .`), el import `from config import settings` funciona desde cualquier script sin manipular `sys.path`.

---

## 5. Comandos Útiles para el Desarrollo

```bash
# Formatear código con Ruff
uv run ruff format .

# Verificar y corregir errores de linting
uv run ruff check --fix .

# Todo en uno (recomendado antes de commits)
uv run ruff check --fix . && uv run ruff format .

# Ejecutar un script específico
uv run python 00-introduccion/ticket_classifier.py
```

---

## 6. Estructura Final del Proyecto

```
pydanticai-course/
├── pyproject.toml           # Configuración del proyecto y Ruff
├── .env                     # Secretos (NO COMMITEAR)
├── .gitignore              
├── config.py                # Configuración global
├── .vscode/
│   └── settings.json        # Configuración de VSCode
├── 00-introduccion/
│   └── ticket_classifier.py # Primer ejemplo
├── 01-core-patterns/
├── 02-contexto/
├── 03-avanzado/
└── 04-produccion/
```

---

## 📚 Recursos Adicionales

- [Documentación oficial de PydanticAI](https://ai.pydantic.dev)
- [Guía de Ruff](https://docs.astral.sh/ruff/)
- [uv Documentation](https://github.com/astral-sh/uv)

---