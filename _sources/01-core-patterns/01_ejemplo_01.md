# Ejemplo 01: CRM Agent (Service Layer Pattern)

En aplicaciones empresariales ("Enterprise"), **nunca** deberías escribir consultas SQL o lógica de negocio compleja directamente dentro de una función decorada con `@agent.tool`.

El patrón correcto es:
1.  **Service Layer:** Una clase Python agnóstica de IA que gestiona la lógica (ej: `CustomerService`).
2.  **Dependency Injection:** Inyectar este servicio en el agente via `deps`.
3.  **Tool:** Una capa fina que simplemente llama al servicio y formatea la respuesta para el LLM.

Este ejemplo muestra cómo construir un Agente de CRM siguiendo este principio, asegurando que el código de negocio sea reutilizable y testeable.

## 📝 Código

Crea el archivo `01-core-patterns/crm_agent.py`:

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated

from devtools import debug
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import settings


# --- 1. DOMINIO (Modelos puros) ---
class Customer(BaseModel):
    id: int
    name: str
    email: str
    status: str  # active, churned, lead


# --- 2. SERVICE LAYER (Lógica de Negocio) ---
class CustomerService:
    """
    Servicio de acceso a datos de clientes.
    Agnóstico de framework - puede usarse con PydanticAI, FastAPI, CLI, etc.
    """

    def __init__(self):
        # Simulación de Base de Datos
        self._db = {
            1: Customer(id=1, name="Alice Corp", email="alice@corp.com", status="active"),
            2: Customer(id=2, name="Bob Ltd", email="bob@ltd.com", status="churned"),
            3: Customer(id=3, name="Charlie Inc", email="charlie@inc.com", status="active"),
        }

    async def get_by_email(self, email: str) -> Customer | None:
        """Busca cliente por coincidencia exacta de email."""
        await asyncio.sleep(0.05)
        for c in self._db.values():
            if c.email == email:
                return c
        return None

    async def search_by_name(self, query: str) -> list[Customer]:
        """Búsqueda difusa (case-insensitive) por nombre."""
        await asyncio.sleep(0.05)
        return [c for c in self._db.values() if query.lower() in c.name.lower()]


# --- 3. INYECCIÓN DE DEPENDENCIAS ---
@dataclass
class CRMDeps:
    crm: CustomerService
    request_id: str  # Útil para tracing/logs


# --- 4. CONFIGURACIÓN DEL AGENTE ---
model = OpenAIChatModel(
    "gpt-4o",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    deps_type=CRMDeps,
    system_prompt=(
        "Eres un asistente de CRM útil y eficiente. "
        "Usa las herramientas disponibles para buscar información de clientes. "
        "Si encuentras múltiples resultados, lístalos de forma resumida."
    ),
)


# --- 5. TOOLS (La "pegatina" entre el LLM y el Servicio) ---
@agent.tool
async def lookup_customer(
    ctx: RunContext[CRMDeps],
    query: Annotated[str, Field(description="Email exacto o nombre parcial del cliente")],
) -> str:
    """
    Busca un cliente en la base de datos del CRM.
    Detecta automáticamente si la query es un email o un nombre.
    """
    service = ctx.deps.crm

    if "@" in query:
        customer = await service.get_by_email(query)
        if customer:
            return f"✅ Encontrado: {customer.model_dump_json()}"
        return f"❌ No se encontró ningún cliente con el email '{query}'."
    else:
        results = await service.search_by_name(query)
        if not results:
            return f"❌ No se encontraron clientes con el nombre '{query}'."

        output = f"🔎 Se encontraron {len(results)} clientes:\n"
        for c in results:
            output += f"- ID {c.id}: {c.name} ({c.email}) [Estado: {c.status}]\n"
        return output


# --- 6. EJECUCIÓN ---
async def main():
    crm_service = CustomerService()
    deps = CRMDeps(crm=crm_service, request_id="req-1234")

    print("--- CASO 1: Búsqueda por Email (Exacta) ---")
    q1 = "¿Cuál es el estatus de bob@ltd.com?"
    print(f"User: {q1}")

    try:
        result = await agent.run(q1, deps=deps)
        debug(result.output)
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n--- CASO 2: Búsqueda por Nombre (Difusa) ---")
    q2 = "Dame información de la empresa Alice"
    print(f"User: {q2}")

    try:
        result = await agent.run(q2, deps=deps)
        debug(result.output)
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Análisis del Patrón

1.  **Reutilización:** La clase `CustomerService` puede ser utilizada por otros sistemas (una API FastAPI, un script de migración, etc.) porque no depende de `pydantic-ai`.
2.  **Facilidad para el LLM:** La tool `lookup_customer` simplifica la decisión del modelo. En lugar de tener dos tools (`search_by_email`, `search_by_name`) y esperar que el LLM elija la correcta, ofrecemos una sola entrada inteligente que enruta internamente.
3.  **Seguridad de Tipos:** Al definir `deps_type=CRMDeps` en el Agente, obtenemos autocompletado y validación estática al acceder a `ctx.deps.crm` dentro de la herramienta.