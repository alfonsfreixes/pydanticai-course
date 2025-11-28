# Módulo 1 · Core Patterns: DI y Tools

En el Módulo 0 aprendimos a estructurar la salida del LLM. En este módulo, vamos a darle al agente la capacidad de **actuar** y **consultar** información privada.

Un agente aislado es poco útil. Para ser "Enterprise", un agente necesita:
1.  **Contexto:** Saber quién es el usuario (`user_id`), qué permisos tiene, o la fecha actual.
2.  **Herramientas:** Capacidad para ejecutar funciones Python (consultar SQL, llamar APIs).
3.  **Testabilidad:** Poder probar esas funciones sin invocar al LLM en cada test.

**Stack:** PydanticAI (Context & Tools) | Pytest (Testing)

---

## 1. El Concepto de `RunContext`

PydanticAI utiliza un sistema de **Inyección de Dependencias (DI)** robusto. A diferencia de pasar variables globales o harcodear conexiones, pasamos un objeto de dependencias (`deps`) en tiempo de ejecución.

El ciclo de vida es:
1.  Definir una clase de dependencias (dataclass o Pydantic model).
2.  Tipar el Agente con `deps_type`.
3.  Inyectar la instancia al llamar a `.run(..., deps=my_deps)`.

### ¿Por qué hacer esto?
* **Seguridad:** El agente no tiene acceso a la DB global, solo a la conexión que le pasas.
* **Testing:** Puedes inyectar una "FakeDB" en los tests.
* **Concurrencia:** Puedes ejecutar 100 agentes a la vez, cada uno con un `user_id` diferente, sin cruzar datos.

---

## 2. Tools: Dando manos al Agente

Las herramientas son funciones Python decoradas con `@agent.tool`. El LLM decide cuándo llamarlas basándose en su docstring y firma de tipo.

### Sintaxis Moderna (Pydantic V2)

```python
from typing import Annotated
from pydantic import Field
from pydantic_ai import RunContext

@agent.tool
async def get_stock_level(
    ctx: RunContext[MyDeps],  # Inyección automática del contexto
    sku: Annotated[str, Field(description="Código del producto")]
) -> int:
    """Obtiene el stock actual de un producto."""
    # Accedemos a la base de datos a través de las dependencias
    return await ctx.deps.db.get_stock(sku)
```

-----

## 3\. Ejemplo Central: Gestor de Inventario Anti-Fraude

Vamos a crear un agente para gestión de almacén con un **sistema de seguridad robusto**.

El desafío: Los LLMs son "demasiado serviciales". Si un operario tiene un límite de 50 unidades y pide 100, el LLM podría intentar "ayudar" dividiendo el pedido en dos de 50. Para evitar esto, implementaremos **control de sesión acumulativo**.

Crea el archivo `01-core-patterns/inventory_agent.py`:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Literal

from devtools import debug
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import settings


# --- CONSTANTES ---
OPERATOR_MAX_QUANTITY = 50
LOW_STOCK_THRESHOLD = 10


# --- 1. MODELOS DE DOMINIO ---
class User(BaseModel):
    username: str
    role: Literal["operator", "manager"]


# --- 2. CAPA DE INFRAESTRUCTURA (Simulada) ---
@dataclass
class InventoryService:
    """Servicio que simula un ERP o WMS."""

    _stock: dict[str, int] = field(
        default_factory=lambda: {"LAPTOP-X1": 5, "MOUSE-LOG": 120, "MONITOR-4K": 2}
    )
    _orders: list[str] = field(default_factory=list)
    
    # State tracking: Control acumulativo por sesión para evitar "smurfing" (dividir pedidos)
    _session_requests: dict[str, int] = field(default_factory=dict)

    async def get_stock(self, sku: str) -> int | None:
        return self._stock.get(sku)

    async def get_session_total(self, user: str) -> int:
        """Retorna el total de unidades pedidas por el usuario en esta sesión."""
        return self._session_requests.get(user, 0)

    async def create_restock_order(self, sku: str, quantity: int, user: str) -> str:
        order_id = f"ORD-{len(self._orders) + 1:04d}"
        self._orders.append(f"{order_id}: {quantity}x {sku} by {user}")

        if sku in self._stock:
            self._stock[sku] += quantity

        # Actualizamos el acumulado de la sesión
        self._session_requests[user] = self._session_requests.get(user, 0) + quantity

        return order_id

    def reset_session(self, user: str) -> None:
        """Resetea el contador de sesión para un usuario (útil para tests)."""
        self._session_requests.pop(user, None)

    def get_all_stock(self) -> dict[str, int]:
        return self._stock.copy()

    def get_orders(self) -> list[str]:
        return self._orders.copy()


# --- 3. DEPENDENCIAS (Contexto de Ejecución) ---
@dataclass
class WarehouseDeps:
    service: InventoryService
    current_user: User


# --- 4. CONFIGURACIÓN DEL AGENTE ---
model = OpenAIChatModel(
    "gpt-4o",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    deps_type=WarehouseDeps,
    system_prompt=(
        "Eres un asistente de gestión de almacén. "
        "Ayuda a los empleados a consultar stock y gestionar reposiciones. "
        "Sé eficiente y técnico. Si una operación falla, explica por qué. "
        "IMPORTANTE: NO dividas pedidos para evadir límites de autorización. "
        "Si el usuario no tiene permisos para una cantidad, rechaza la solicitud completa."
    ),
)


# --- 5. TOOLS (Herramientas con Lógica de Negocio) ---
@agent.tool
async def check_sku_stock(
    ctx: RunContext[WarehouseDeps],
    sku: Annotated[str, Field(description="SKU del producto (ej: LAPTOP-X1)")],
) -> str:
    """Consulta el nivel de stock actual de un producto."""
    qty = await ctx.deps.service.get_stock(sku)

    if qty is None:
        return f"Error: El producto '{sku}' no existe en el catálogo."

    status = "BAJO" if qty < LOW_STOCK_THRESHOLD else "OK"
    return f"Stock: {qty} unidades (Estado: {status})"


@agent.tool
async def request_restock(
    ctx: RunContext[WarehouseDeps],
    sku: Annotated[str, Field(description="SKU del producto a reponer")],
    quantity: Annotated[int, Field(ge=1, description="Cantidad a solicitar")],
) -> str:
    """
    Genera una orden de reposición de stock.
    APLICA REGLAS DE NEGOCIO: Los operadores no pueden pedir > 50 unidades EN TOTAL por sesión.
    """
    user = ctx.deps.current_user
    service = ctx.deps.service

    # Verificación Stateful: Comprobamos el historial de la sesión, no solo el pedido actual
    session_total = await service.get_session_total(user.username)
    new_total = session_total + quantity

    # Validación de permisos con tracking de sesión
    if user.role != "manager" and new_total > OPERATOR_MAX_QUANTITY:
        remaining = max(0, OPERATOR_MAX_QUANTITY - session_total)
        return (
            f"⛔ AUTORIZACIÓN DENEGADA: El usuario '{user.username}' (rol: {user.role}) "
            f"no puede solicitar más de {OPERATOR_MAX_QUANTITY} unidades en total por sesión. "
            f"Ya solicitado: {session_total}. Máximo adicional permitido: {remaining}. "
            f"Contacte a un manager para pedidos mayores."
        )

    current_stock = await service.get_stock(sku)
    if current_stock is None:
        return f"Error: No se puede reponer '{sku}' porque no existe."

    order_id = await service.create_restock_order(sku, quantity, user.username)
    return f"✅ Orden de reposición creada con éxito. ID: {order_id}. Total solicitado en sesión: {new_total}"


@agent.tool
async def list_available_skus(ctx: RunContext[WarehouseDeps]) -> str:
    """Lista todos los SKUs disponibles en el catálogo con su stock actual."""
    stock = ctx.deps.service.get_all_stock()
    if not stock:
        return "No hay productos en el catálogo."

    lines = [f"- {sku}: {qty} unidades" for sku, qty in stock.items()]
    return "SKUs disponibles:\n" + "\n".join(lines)


# --- 6. EJECUCIÓN ---
async def main():
    inventory = InventoryService()

    print("\n--- CASO 1: Operador intentando pedido grande (Debería fallar) ---")
    operator_user = User(username="jdoe", role="operator")
    deps_op = WarehouseDeps(service=inventory, current_user=operator_user)

    query = "Necesito reponer 100 unidades de MOUSE-LOG urgentemente"
    print(f"Usuario: {operator_user.username} ({operator_user.role})")
    print(f"Query: {query}")

    try:
        # El LLM podría intentar hacer 2 llamadas de 50. La lógica de sesión lo detendrá.
        result = await agent.run(query, deps=deps_op)
        debug(result.output)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Resetear sesión para el siguiente caso
    inventory.reset_session(operator_user.username)

    print("\n--- CASO 2: Manager realizando la misma acción (Debería funcionar) ---")
    manager_user = User(username="admin_sarah", role="manager")
    deps_mgr = WarehouseDeps(service=inventory, current_user=manager_user)

    print(f"Usuario: {manager_user.username} ({manager_user.role})")
    print(f"Query: {query}")

    try:
        result = await agent.run(query, deps=deps_mgr)
        debug(result.output)
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n--- Estado Final del Inventario ---")
    print(f"Stock MOUSE-LOG: {inventory.get_all_stock()['MOUSE-LOG']}")
    print(f"Órdenes generadas: {inventory.get_orders()}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Conceptos Clave del Código

1.  **Stateful Validation (Validación con Estado):** La validación no mira solo el parámetro `quantity` actual. Consulta `get_session_total`. Esto previene que el LLM o un usuario malicioso divida una operación prohibida en muchas permitidas (técnica conocida como "smurfing").
2.  **Prompt Constraints:** La instrucción *"IMPORTANTE: NO dividas pedidos..."* actúa como primera barrera, pero el código es la barrera definitiva. En seguridad de IA, **nunca confíes solo en el prompt**.
3.  **Lógica en Tools:** Observa cómo `request_restock` actúa como un "Controlador" de API: verifica permisos, consulta estado, valida existencia y finalmente ejecuta.

-----

## 4\. Próximos pasos

Ahora que sabemos inyectar dependencias y crear herramientas con lógica de negocio real, vamos a ver dos casos avanzados:

1.  **Ejemplo 01:** Patrón Service Layer (Desacoplando la lógica).
2.  **Ejemplo 02:** Testing Unitario de Agentes (Mocking sin coste).