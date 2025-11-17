# Módulo 1 · Agentes Avanzados

**Bienvenido al Módulo 1.** En el módulo anterior aprendiste los fundamentos: configurar agentes, usar tools simples y estructurar salidas. Ahora vamos a profundizar en técnicas de producción que separan un prototipo de una aplicación empresarial.

En este módulo construirás agentes que:
- Validan sus propias respuestas antes de devolverlas
- Se autocorrigen cuando detectan errores
- Manejan casos edge de forma elegante
- Integran con sistemas externos de manera controlada

**Versión actual:** PydanticAI 1.10+ | Pydantic 2.12 | Python ≥3.11

---

## ¿Qué aprenderás en este módulo?

Este módulo te prepara para construir agentes de nivel empresarial. Los conceptos que cubriremos son los que marcan la diferencia entre "funciona en mi máquina" y "funciona en producción con clientes reales":

- **Tools empresariales con validación compleja**: Aprenderás a crear funciones que el LLM puede llamar, con validación automática de parámetros usando Pydantic
- **Result validators**: Implementarás verificaciones que garantizan la calidad de las respuestas antes de mostrarlas al usuario
- **Retries inteligentes**: Configurarás estrategias de reintento que permiten al agente recuperarse de errores sin intervención manual
- **Reflection patterns**: El agente revisará su propia salida y la mejorará iterativamente, como un editor humano
- **Orquestación de múltiples tools**: Gestionar agentes que deciden dinámicamente qué herramientas usar y en qué orden
- **Manejo profesional de errores**: Construir sistemas que fallan gracefully, sin exponer stack traces al usuario final

Al final del módulo, habrás evolucionado el caso de uso DataPulse AI del Módulo 0 en un sistema con validación, autocorrección y manejo robusto de errores.

---

## 1. Tools avanzadas con parámetros complejos

En el Módulo 0 viste tools simples (como `get_current_time()`). En producción, las tools suelen necesitar múltiples parámetros, validación de reglas de negocio y manejo de casos edge.

**¿Por qué es importante?**  
Un LLM puede "alucinar" parámetros inválidos. Si tu tool consulta una base de datos, quieres asegurarte de que los parámetros están validados **antes** de ejecutar queries. Aquí es donde brilla la combinación de PydanticAI + Pydantic: validación automática de tipos en tiempo de ejecución.

### 1.1. Tool con múltiples parámetros y validación

Vamos a crear un sistema de reporting de ventas. El agente podrá generar reportes filtrando por fecha, región y importe mínimo. La tool validará automáticamente que las fechas sean válidas, que la región sea una de las permitidas, y que el importe sea positivo.

**¿Qué vamos a construir?**  
Un agente analista de datos que puede responder preguntas tipo: "¿Cuánto facturamos en Europa en enero con pedidos superiores a 1000 EUR?"

**`01-agentes-avanzados/tool_avanzada.py`:**

```python
from datetime import date
from typing import Annotated, Literal, Dict, Any, List, TypedDict
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

class Transaction(TypedDict):
    date: str
    region: str
    amount: float
    product: str

# Base de datos mock de transacciones
# En producción, esto vendría de tu base de datos real
MOCK_TRANSACTIONS: List[Transaction] = [
    {"date": "2025-01-05", "region": "EU", "amount": 1250.00, "product": "Laptop Pro"},
    {"date": "2025-01-08", "region": "EU", "amount": 2100.00, "product": "Monitor 4K"},
    {"date": "2025-01-12", "region": "EU", "amount": 890.00, "product": "Mouse Wireless"},
    {"date": "2025-01-15", "region": "EU", "amount": 3400.00, "product": "Laptop Pro"},
    {"date": "2025-01-18", "region": "NA", "amount": 1850.00, "product": "Laptop Pro"},
    {"date": "2025-01-20", "region": "EU", "amount": 1120.00, "product": "Teclado Mecánico"},
    {"date": "2025-01-22", "region": "LATAM", "amount": 670.00, "product": "Mouse Wireless"},
    {"date": "2025-01-25", "region": "EU", "amount": 4200.00, "product": "Monitor 4K"},
    {"date": "2025-01-28", "region": "APAC", "amount": 2900.00, "product": "Laptop Pro"},
    {"date": "2025-01-30", "region": "EU", "amount": 1560.00, "product": "Laptop Pro"},
]

model = AnthropicModel(
    model_name="claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(
    model=model,
    instructions=
    "Eres un analista de datos empresarial. "
    "Usa la tool de reporting para responder sobre ventas. "
    "Las fechas deben estar en formato YYYY-MM-DD."
)

@agent.tool
async def generate_sales_report(
    ctx: RunContext[None],
    start_date: Annotated[str, Field(description="Fecha inicio en formato YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Fecha fin en formato YYYY-MM-DD")],
    region: Annotated[
        Literal["EU", "NA", "LATAM", "APAC"],
        Field(description="Región: EU, NA, LATAM o APAC")
    ],
    min_revenue: Annotated[
        float,
        Field(ge=0, description="Filtrar solo pedidos con importe mínimo en EUR")
    ] = 0.0
) -> Dict[str, Any]:
    """
    Genera un informe de ventas para el período y región especificados.
    
    Filtra transacciones por fecha, región y monto mínimo, devolviendo
    métricas agregadas y los productos más vendidos.
    
    Args:
        start_date: Fecha de inicio del período (YYYY-MM-DD)
        end_date: Fecha de fin del período (YYYY-MM-DD)
        region: Región geográfica (EU, NA, LATAM, APAC)
        min_revenue: Importe mínimo por transacción en EUR (por defecto 0)
        
    Returns:
        Dict con total_revenue, num_transactions, top_products y avg_ticket.
    """
    # Convertir strings a date objects para comparación
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    # Validación de regla de negocio: fechas coherentes
    if end < start:
        return {"error": "La fecha de fin debe ser posterior a la fecha de inicio"}
    
    # Filtrar transacciones según criterios
    filtered_transactions = [
        t for t in MOCK_TRANSACTIONS
        if (start <= date.fromisoformat(t["date"]) <= end
            and t["region"] == region
            and t["amount"] >= min_revenue)
    ]
    
    # Caso edge: sin resultados
    if not filtered_transactions:
        return {
            "period": f"{start_date} a {end_date}",
            "region": region,
            "total_revenue_eur": 0,
            "num_transactions": 0,
            "top_products": [],
            "avg_ticket_eur": 0,
            "message": "No se encontraron transacciones con los criterios especificados"
        }
    
    # Calcular métricas agregadas
    total_revenue = sum(t["amount"] for t in filtered_transactions)
    num_transactions = len(filtered_transactions)
    
    # Top 3 productos más vendidos por cantidad de ventas
    product_count: Dict[str, int] = {}
    for t in filtered_transactions:
        product = t["product"]
        product_count[product] = product_count.get(product, 0) + 1
    
    top_products = sorted(
        product_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    return {
        "period": f"{start_date} a {end_date}",
        "region": region,
        "total_revenue_eur": round(total_revenue, 2),
        "num_transactions": num_transactions,
        "top_products": [f"{prod} ({count} ventas)" for prod, count in top_products],
        "avg_ticket_eur": round(total_revenue / num_transactions, 2)
    }

# Ejemplos de uso que demuestran diferentes capacidades
print("=== Consulta 1: Todas las ventas de Europa en enero ===")
result1 = agent.run_sync(
    "¿Cuánto facturamos en Europa en enero de 2025?"
)
print(result1.output)

print("\n=== Consulta 2: Ventas en Europa con pedidos >1000 EUR ===")
result2 = agent.run_sync(
    "¿Cuánto facturamos en Europa entre el 1 y 31 de enero de 2025 "
    "con pedidos superiores a 1000 EUR?"
)
print(result2.output)

print("\n=== Consulta 3: Productos más vendidos en Europa ===")
result3 = agent.run_sync(
    "¿Cuáles fueron los productos más vendidos en Europa en la segunda quincena de enero de 2025?"
)
print(result3.output)
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/tool_avanzada.py
```

**¿Qué acabas de ver?**

Analicemos las piezas clave de este código:

1. **`Annotated[Type, Field(...)]`**: Esta combinación le dice a Pydantic:
   - Qué tipo esperas (`str`, `float`, `Literal[...]`)
   - Qué restricciones aplicar (`ge=0` para "greater or equal than 0")
   - Qué descripción mostrar al LLM para que entienda cómo usar el parámetro

2. **`Literal["EU", "NA", "LATAM", "APAC"]`**: El LLM **solo puede pasar** uno de estos valores. Si intenta pasar "Europa" o "Europe", Pydantic rechaza la llamada antes de ejecutar tu código.

3. **Docstring detallado**: El LLM lo lee para entender:
   - Qué hace la función
   - Qué parámetros necesita
   - Qué formato espera (ej: "YYYY-MM-DD")
   - Qué devuelve

4. **Validación en capas**:
   - Pydantic valida tipos y constraints (`ge=0`, `Literal`)
   - Tu código valida reglas de negocio (`end >= start`)
   - Manejo de caso sin resultados (devuelve estructura válida con mensaje)

5. **Datos mock realistas**: Usamos datos simulados para que el código funcione sin dependencias externas, pero la estructura es idéntica a lo que harías con una base de datos real.

**Beneficios en producción:**
- ✅ **Seguridad**: Imposible inyectar SQL porque los parámetros están validados
- ✅ **Debugging**: Si algo falla, sabes exactamente qué parámetro era inválido
- ✅ **Documentación viva**: El schema Pydantic es tu documentación
- ✅ **Testing**: Puedes probar la tool independientemente del agente

---

### 1.2. Múltiples tools y orquestación

En sistemas reales, un agente suele necesitar acceso a varias herramientas. El LLM decide automáticamente cuál usar según el contexto de la pregunta.

**¿Por qué es útil?**  
Imagina un asistente de e-commerce: necesita consultar inventario, rastrear pedidos, verificar precios, etc. No quieres un agente diferente para cada tarea. Quieres un agente que sepa qué tool usar para cada pregunta.

**¿Qué vamos a construir?**  
Un asistente que puede:
- Consultar stock de productos
- Rastrear el estado de pedidos
- Combinar ambas en una sola respuesta cuando sea relevante

**`01-agentes-avanzados/multi_tools.py`:**

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings
from typing import Dict, Any
from enum import Enum

class OrderStatus(str, Enum):
    """Estados posibles de un pedido."""
    PROCESSING = "processing"  # En preparación
    READY_TO_SHIP = "ready_to_ship"  # Listo para enviar
    SHIPPED = "shipped"  # Enviado
    DELIVERED = "delivered"  # Entregado
    CANCELLED = "cancelled"  # Cancelado

class Product(BaseModel):
    name: str
    stock: int
    price_eur: float

class Order(BaseModel):
    product_code: str
    quantity: int
    status: OrderStatus
    estimated_delivery_days: int | None = None

# Simulación de sistemas empresariales
INVENTORY: Dict[str, Product] = {
    "P001": Product(name="Laptop Pro", stock=45, price_eur=1299.00),
    "P002": Product(name="Mouse Wireless", stock=0, price_eur=29.99),
    "P003": Product(name="Monitor 4K", stock=12, price_eur=449.00),
}

ORDERS_DB: Dict[str, Order] = {
    "ORD-1001": Order(
        product_code="P001",
        quantity=2,
        status=OrderStatus.SHIPPED,
        estimated_delivery_days=2
    ),
    "ORD-1002": Order(
        product_code="P003",
        quantity=1,
        status=OrderStatus.PROCESSING,
        estimated_delivery_days=3
    ),
}

# Mapeo de estados a descripciones claras
STATUS_DESCRIPTIONS = {
    OrderStatus.PROCESSING: "En preparación en almacén",
    OrderStatus.READY_TO_SHIP: "Preparado, pendiente de recogida por transportista",
    OrderStatus.SHIPPED: "En tránsito",
    OrderStatus.DELIVERED: "Entregado",
    OrderStatus.CANCELLED: "Cancelado",
}

model = OpenAIChatModel(
    model_name="gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model=model,
    instructions=(
        "Eres un asistente de e-commerce profesional y conciso. "
        "Responde consultas sobre inventario y pedidos de forma directa y clara.\n\n"
        "IMPORTANTE:\n"
        "- Proporciona la información solicitada de forma directa\n"
        "- NO hagas preguntas adicionales ni ofrezcas opciones extras\n"
        "- Usa lenguaje ejecutivo y profesional\n"
        "- Si el usuario menciona un nombre de producto, usa search_product primero\n"
        "- Sé conciso: máximo 2-3 frases por respuesta"
    )
)

@agent.tool
def search_product(ctx: RunContext[None], query: str) -> Dict[str, Any]:
    """
    Busca productos por nombre o descripción.
    
    Usa esta función cuando el usuario mencione un producto por su nombre
    (ej: "Mouse Wireless", "Laptop") pero NO cuando mencione un código (ej: P001).
    
    Args:
        query: Nombre o parte del nombre del producto a buscar.
        
    Returns:
        Dict con lista de productos que coinciden con la búsqueda.
    """
    query_lower = query.lower()
    matches: list[Dict[str, Any]] = [
        {
            "product_code": code,
            "name": product.name,
            "match_score": 1.0 if query_lower in product.name.lower() else 0.5
        }
        for code, product in INVENTORY.items()
        if query_lower in product.name.lower()
    ]
    
    if not matches:
        return {
            "found": False,
            "message": f"No se encontraron productos que coincidan con '{query}'",
            "available_products": [
                {"code": code, "name": product.name}
                for code, product in INVENTORY.items()
            ]
        }
    
    return {
        "found": True,
        "matches": matches,
        "message": f"Se encontraron {len(matches)} productos"
    }

@agent.tool
def check_stock(ctx: RunContext[None], product_code: str) -> Dict[str, Any]:
    """
    Consulta el stock actual de un producto por su código.
    
    Usa esta función cuando tengas el código del producto (formato PXXX).
    Si solo tienes el nombre, usa search_product primero.
    
    Args:
        product_code: Código único del producto (ej: P001, P002, P003).
        
    Returns:
        Dict con información detallada del producto y disponibilidad.
    """
    product = INVENTORY.get(product_code)
    if not product:
        return {"error": f"Producto {product_code} no encontrado"}
    
    return {
        "product_code": product_code,
        "name": product.name,
        "stock_units": product.stock,
        "price_eur": product.price_eur,
        "available": product.stock > 0,
        "status_message": (
            f"Disponible ({product.stock} unidades)" 
            if product.stock > 0 
            else "Agotado"
        )
    }

@agent.tool
def track_order(ctx: RunContext[None], order_id: str) -> Dict[str, Any]:
    """
    Rastrea el estado de un pedido por su ID.
    
    Usa esta función cuando el usuario pregunte sobre el estado,
    seguimiento o ubicación de un pedido específico.
    
    Args:
        order_id: ID del pedido en formato ORD-XXXX (ej: ORD-1001).
        
    Returns:
        Dict con detalles del pedido, estado actual y tiempo estimado de entrega.
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": f"Pedido {order_id} no encontrado"}
    
    product = INVENTORY.get(order.product_code)
    status_desc = STATUS_DESCRIPTIONS.get(order.status, "Estado desconocido")
    
    result: Dict[str, Any] = {
        "order_id": order_id,
        "product_name": product.name if product else "Unknown",
        "quantity": order.quantity,
        "status": order.status.value,
        "status_description": status_desc,
    }
    
    # Agregar información de entrega solo si está disponible
    if order.estimated_delivery_days:
        result["estimated_delivery"] = f"{order.estimated_delivery_days} días hábiles"
    
    return result

# Pruebas que demuestran las mejoras
consulta: str = "¿Tenéis disponible el Mouse Wireless?"
print("=== Consulta 1: Respuesta directa sin preguntas adicionales ===\n")
print(consulta)

r1 = agent.run_sync(consulta)
print(r1.output)

consulta2: str = "¿Cuál es el estado de mi pedido ORD-1002?"
print("\n=== Consulta 2: Estado claro y tiempo estimado ===\n") 
print(consulta2)
r2 = agent.run_sync(consulta2)
print(r2.output)

consulta3: str = "Quiero comprar un Monitor 4K, ¿hay stock? Y de paso revisa el pedido ORD-1001"
print("\n=== Consulta 3: Información completa y concisa ===\n")
print(consulta3)
r3 = agent.run_sync(consulta3)
print(r3.output)
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/multi_tools.py
```
---

## Cómo funciona la orquestación

El agente toma decisiones en tiempo real:

1. **Analiza** la consulta e identifica intenciones
2. **Planifica** qué tools necesita y en qué orden
3. **Ejecuta** la cadena de tools secuencialmente
4. **Sintetiza** los resultados en una respuesta concisa

---

## Desglose de consultas

### Consulta 1: "¿Tenéis disponible el Mouse Wireless?"

**Flujo automático (2 tools):**
```
Usuario: "Mouse Wireless" (nombre, no código)
    ↓
search_product("Mouse Wireless") → {"product_code": "P002"}
    ↓
check_stock("P002") → {"stock_units": 0, "status": "Agotado"}
    ↓
Respuesta: "El Mouse Wireless (código P002) está agotado; stock 0 unidades."
```

**El agente aprendió solo** que debe buscar el código primero cuando el usuario menciona un nombre.

---

### Consulta 2: "¿Cuál es el estado de mi pedido ORD-1002?"

**Tool única con información estructurada:**
```
track_order("ORD-1002") → {
    "status_description": "En preparación en almacén",
    "estimated_delivery": "3 días hábiles"
}
```

**Diseño de estados:**
- ❌ Malo: `"status": "pending"` → ambiguo
- ✅ Bueno: `"status_description": "En preparación en almacén"` + tiempo estimado

---

### Consulta 3: "Monitor 4K, ¿hay stock? Revisa pedido ORD-1001"

**Orquestación compleja (3 tools, 2 intenciones):**
```
search_product("Monitor 4K") → P003
    ↓
check_stock("P003") → 12 unidades, 449 EUR
    ↓
track_order("ORD-1001") → enviado, 2 días
    ↓
Respuesta: combina ambas informaciones de forma profesional
```

**Demuestra:**
- Identificar múltiples intenciones en una pregunta
- Ejecutar tools en orden lógico correcto
- Sintetizar sin perder información clave

---

## Control del tono conversacional

**Problema:** Sin instrucciones explícitas, los LLMs son excesivamente conversacionales:
```
❌ "¡Claro! El Mouse Wireless está agotado. 😔
¿Te gustaría que:
1. Te avise cuando se reponga?
2. Busque alternativas?
..."

✅ "El Mouse Wireless (código P002) está agotado; stock 0 unidades."
```

**Solución:**
```python
instructions=(
    "Eres un asistente de e-commerce profesional y conciso.\n\n"
    "IMPORTANTE:\n"
    "- Proporciona la información de forma directa\n"
    "- NO hagas preguntas adicionales\n"
    "- Máximo 2-3 frases por respuesta"
)
```

**Claves:**
- Mayúsculas para énfasis (IMPORTANTE, NO)
- Límites concretos (máximo 2-3 frases)
- Prohibir comportamientos no deseados explícitamente

---

## Comparación: Tradicional vs PydanticAI

**Enfoque tradicional (rígido):**
```python
if "stock" in query and "pedido" in query:
    check_stock(); track_order()
elif "stock" in query:
    check_stock()
# ❌ No maneja sinónimos, paráfrasis ni encadenamiento
```

**Enfoque PydanticAI (flexible):**
```python
agent.run_sync(query)  # El LLM decide qué, cuándo y cómo
# ✅ Lenguaje natural completo
# ✅ Encadenamiento automático
# ✅ Escala a 10+ tools sin cambiar código
```

---

## Buenas prácticas

### 1. Docstrings descriptivos
```python
# ✅ El LLM lee esto para decidir cuándo usar la tool
"""
Busca productos por nombre o descripción.

Usa esta función cuando el usuario mencione un producto por su nombre
(ej: "Mouse Wireless") pero NO cuando mencione un código (ej: P001).
"""
```

### 2. Separación de responsabilidades
```python
# ✅ Cada tool hace UNA cosa
search_product()  # Solo busca códigos
check_stock()     # Solo consulta stock con código conocido

# ❌ Tool que hace demasiado
get_product_info()  # ¿Busca? ¿Consulta? ¿Ambos?
```

### 3. Estados claros
```python
# ✅ Enum + descripciones
class OrderStatus(str, Enum):
    PROCESSING = "processing"
    SHIPPED = "shipped"

STATUS_DESCRIPTIONS = {
    OrderStatus.PROCESSING: "En preparación en almacén",
    OrderStatus.SHIPPED: "En tránsito",
}
```

---

## Punto clave

> **Tu trabajo:** Diseñar tools bien separadas con docstrings claros  
> **Trabajo del LLM:** Orquestarlas inteligentemente
> 
> No programas el flujo — el agente lo descubre leyendo la documentación.


---

## 2. Validadores de salida y *Reflection*

Hasta ahora, el agente genera una respuesta y la devuelve al usuario inmediatamente.  
Pero, ¿qué pasa si la respuesta es demasiado vaga, genérica o contradice tus reglas de negocio?

Los **validadores de salida** (`@agent.output_validator`) te permiten interceptar la respuesta **antes** de que llegue al usuario, verificarla, y si no cumple tus criterios, hacer que el agente la regenere automáticamente mediante `ModelRetry`.

### 🔍 ¿Por qué es revolucionario?

Es como tener un **editor automático de calidad** que revisa todo lo que escribe el modelo.  
El agente itera hasta generar un resultado que pase las validaciones o hasta agotar el número de reintentos definidos.

---

### 2.1. Validación básica con `ModelRetry`

Vamos a construir un generador de **propuestas comerciales**. Queremos garantizar que cada propuesta:

- Mencione explícitamente el **nombre del cliente** en el título.  
- Use **lenguaje concreto y profesional**, evitando frases genéricas tipo “solución innovadora” o “líder del mercado”.  
- Incluya **beneficios bien desarrollados**, con un mínimo de 8 palabras cada uno.  
- Y, si el *brief* incluye un presupuesto, asegure que la **estimación económica sea coherente** con ese valor (±20%).

---

### 💡 ¿Qué vamos a construir?

Un sistema que genera propuestas profesionales y **se autocorrige** si detecta problemas de calidad.

Este ejemplo muestra cómo usar `@agent.output_validator` y `ModelRetry` para crear agentes que **piensan dos veces antes de responder**, validando sus salidas según criterios empresariales personalizados.


**`01-agentes-avanzados/result_validation.py`:**

```python
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from config import settings

class SalesProposal(BaseModel):
    """Propuesta comercial estructurada."""
    client_name: str = Field(min_length=3, description="Nombre del cliente")
    proposal_title: str = Field(min_length=10, max_length=100)
    budget_eur: Optional[float] = Field(default=None, gt=0, description="Presupuesto del brief si está disponible")
    estimated_value_eur: float = Field(gt=0, le=1_000_000)
    executive_summary: str = Field(min_length=50, max_length=300)
    key_benefits: list[str] = Field(min_length=3, max_length=5)

model = AnthropicModel(
    model_name="claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

MAX_RETRIES = 2

agent = Agent(
    model=model,
    output_type=SalesProposal,
    retries=MAX_RETRIES,
    instructions=(
        "Eres un consultor comercial que crea propuestas de valor para clientes B2B. "
        "Devuelve *exclusivamente* un objeto válido del tipo SalesProposal. "
        "Escribe en español, tono ejecutivo y concreto. Evita jerga y frases vacías. "
        "Si el brief trae presupuesto, rellena el campo 'budget_eur' con esa cifra exacta. "
        "Procura que 'estimated_value_eur' sea coherente con 'budget_eur' (si existe). "
        "Incluye beneficios específicos y, cuando proceda, cuantifícalos."
    ),
)

@agent.output_validator
async def validate_proposal_quality(
    ctx: RunContext[None],
    result: SalesProposal
) -> SalesProposal:
    """
    Valida la calidad de la propuesta comercial.
    Lanza ModelRetry con feedback accionable si detecta problemas.
    """
    # --------- Validación 1: Título debe mencionar al cliente ---------
    if result.client_name.lower() not in result.proposal_title.lower():
        raise ModelRetry(
            f"El título debe mencionar explícitamente al cliente '{result.client_name}'. "
            f"Título actual: '{result.proposal_title}'. Reescribe el título incluyendo el nombre del cliente."
        )

    # --------- Validación 2: Evitar resúmenes genéricos ---------
    summary_lc = result.executive_summary.lower()
    generic_phrases = {
        "solución innovadora",
        "la mejor opción",
        "excelente oportunidad",
        "líder del mercado",
        "state of the art",
        "mejores prácticas",
        "gran impacto",
        "resultados sin precedentes",
        "alto valor añadido",
    }
    if any(p in summary_lc for p in generic_phrases):
        raise ModelRetry(
            "El resumen ejecutivo contiene frases genéricas. "
            "Sustituye por lenguaje específico con datos concretos, métricas estimadas y supuestos claros."
        )

    # --------- Validación 3: Beneficios con sustancia (≥ 8 palabras) ---------
    short_benefits = [b for b in result.key_benefits if len(b.split()) < 8]
    if short_benefits:
        raise ModelRetry(
            f"Los siguientes beneficios son demasiado vagos: {short_benefits}. "
            "Desarrolla cada beneficio con al menos 8 palabras, con verbos de acción y, si procede, métricas estimadas."
        )

    # --------- Validación 4: Coherencia presupuesto vs estimación (simple y explícita) ---------
    # Si budget_eur viene informado, pedimos que estimated_value_eur no se desvíe >±20%
    if result.budget_eur is not None and result.budget_eur > 0:
        lower = 0.2 * result.budget_eur
        upper = 1.2 * result.budget_eur
        if not (lower <= result.estimated_value_eur <= upper):
            raise ModelRetry(
                f"El valor estimado ({result.estimated_value_eur:,.0f} EUR) no es coherente con el presupuesto "
                f"({result.budget_eur:,.0f} EUR). Ajusta la cifra o justifica brevemente (1-2 frases) la desviación."
            )

    return result

def run_example() -> None:
    prompt = (
        "Crea una propuesta para  DataPulse para implementar un sistema de IA de análisis de datos. "
        "Presupuesto estimado: 35.000 EUR. Entregables: pipeline de ingesta, dashboard ejecutivo y "
        "automatización de informes semanales."
    )

    try:
        result = agent.run_sync(prompt)
        proposal = result.output

        print("=" * 80)
        print(f"Cliente : {proposal.client_name}")
        print(f"Título  : {proposal.proposal_title}")
        if proposal.budget_eur is not None:
            print(f"Presupuesto (brief): {proposal.budget_eur:,.2f} EUR".replace(",", "X").replace(".", ",").replace("X", "."))
        print(f"Valor estimado     : {proposal.estimated_value_eur:,.2f} EUR".replace(",", "X").replace(".", ",").replace("X", "."))
        print("-" * 80)
        print("Resumen Ejecutivo:\n")
        print(proposal.executive_summary)
        print("\nBeneficios Clave:")
        for i, benefit in enumerate(proposal.key_benefits, 1):
            print(f"  {i}. {benefit}")
        print("=" * 80)

    except Exception as e:
        print(f"Error tras {MAX_RETRIES} intentos: {e}")


if __name__ == "__main__":
    run_example()
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/result_validation.py
```

```markdown
## ¿Qué está pasando aquí?

El **validator intercepta** la respuesta del agente antes de devolverla al usuario:

1. Agente genera propuesta inicial
2. `validate_proposal_quality()` revisa 4 criterios
3. Si falla → `ModelRetry` con feedback específico
4. Agente regenera incorporando el feedback
5. Repite hasta pasar validación o agotar `MAX_RETRIES=2`

---

## Las 4 validaciones empresariales

### 1. Cliente en el título
```python
if result.client_name.lower() not in result.proposal_title.lower():
    raise ModelRetry("El título debe mencionar 'DataPulse'")
```

### 2. Lenguaje concreto
```python
generic_phrases = {"solución innovadora", "líder del mercado", "alto valor añadido"}
# Rechaza frases vacías, exige datos concretos
```

### 3. Beneficios desarrollados (≥8 palabras)
```python
short_benefits = [b for b in result.key_benefits if len(b.split()) < 8]
if short_benefits:
    raise ModelRetry("Desarrolla beneficios con métricas concretas")
```

### 4. Coherencia presupuesto ±20%
```python
# Brief: 35.000 EUR → estimación debe estar entre 28k-42k
if not (0.8 * budget <= estimated <= 1.2 * budget):
    raise ModelRetry("Valor estimado incoherente con presupuesto")
```

---

## Ejemplo de iteración

**Intento 1:**
```
Título: "Propuesta de IA para análisis empresarial"
```
❌ "El título debe mencionar 'DataPulse'"

**Intento 2:**
```
Título: "Sistema de Análisis IA para DataPulse"
Resumen: "Una solución innovadora que transformará..."
```
❌ "Contiene frases genéricas"

**Intento 3:**
```
Título: "Sistema de Análisis IA para DataPulse"
Resumen: "Pipeline ML procesando 50k registros/día con dashboard ejecutivo..."
Beneficios: ["ROI positivo", "Mejor análisis"]
```
❌ "Beneficios demasiado vagos (< 8 palabras)"

**Resultado:** Si agota `MAX_RETRIES=2`, lanza excepción con el último error.

---

## Conceptos clave

- **`@agent.output_validator`**: Decora función que valida el resultado
- **`ModelRetry`**: Excepción que fuerza regeneración con feedback específico
- **`retries=2`**: Máximo de reintentos (3 intentos totales)
- **Feedback accionable**: No "está mal", sino **qué** y **cómo** corregir

---

## Beneficios empresariales

| Beneficio | Impacto |
|-----------|---------|
| Calidad consistente | Todas las propuestas cumplen estándares |
| Sin revisión manual | Autocorrección de errores comunes |
| Feedback trazable | Logs muestran qué validación falló |
| Costes controlados | `MAX_RETRIES` limita tokens gastados |

---

## Por qué funciona

**Sin validator:**
```
Agente: "Una innovadora solución líder del mercado..." ❌
```

**Con validator:**
```
Intento 1: "Una innovadora solución..."
Validator: ❌ "Elimina frases genéricas"
Intento 2: "Pipeline procesando 50k registros/día con reducción 
           estimada de 15h/semana en análisis manual..." ✅
```

El validator actúa como **control de calidad automático** que garantiza que ninguna propuesta sale sin cumplir tus criterios de negocio.
```

---

### 2.2. Reflection pattern avanzado

El **reflection pattern** lleva la validación un paso más allá: el agente actúa como su propio crítico, evaluando su trabajo y mejorándolo iterativamente.

**¿En qué se diferencia del validator básico?**
- **Validator básico**: Verificas reglas específicas (longitud, formato, etc.)
- **Reflection**: El agente evalúa **calidad subjetiva** (tono, claridad, profesionalidad)

**¿Qué vamos a construir?**  
Un sistema de redacción de emails corporativos que se autocritica y mejora hasta cumplir estándares profesionales.

**`01-agentes-avanzados/reflection_pattern.py`:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

class EmailDraft(BaseModel):
    """Borrador de email corporativo."""
    subject: str = Field(max_length=80)
    body: str = Field(min_length=100, max_length=800)
    call_to_action: str = Field(description="Llamada a la acción clara")
    tone_score: int = Field(ge=1, le=5, description="1=muy formal, 5=muy casual")

model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=EmailDraft,
    retries=3,  # Más reintentos para permitir refinamiento iterativo
    instructions=(
        "Eres un redactor corporativo experto. "
        "Creas emails profesionales, claros y orientados a la acción."
    )
)

@agent.result_validator
async def reflect_on_email_quality(
    ctx: RunContext[None],
    result: EmailDraft
) -> EmailDraft:
    """
    Implementa reflection: el agente evalúa su propia salida.
    
    En lugar de solo verificar reglas técnicas, este validator
    actúa como un editor humano que busca problemas de estilo,
    claridad y efectividad comunicativa.
    
    Criterios de calidad profesional:
    - Asunto específico y accionable (no clickbait)
    - Cuerpo conciso sin palabrería innecesaria
    - CTA inequívoca que dice exactamente qué hacer
    - Tono apropiado para comunicación corporativa (2-3/5)
    """
    issues = []
    
    # Criterio 1: Asunto sin sensacionalismo
    if any(word in result.subject.lower() for word in ["importante", "urgente", "atención"]):
        issues.append(
            "El asunto usa palabras sensacionalistas. "
            "Hazlo más específico y profesional."
        )
    
    # Criterio 2: Tono apropiado para contexto corporativo
    if result.tone_score not in [2, 3]:
        issues.append(
            f"El tono ({result.tone_score}/5) no es apropiado para comunicación corporativa. "
            "Usa un tono profesional equilibrado (2-3)."
        )
    
    # Criterio 3: CTA bien formada
    if not result.call_to_action.strip().endswith(("?", ".", "!")):
        issues.append("La llamada a la acción debe ser una oración completa.")
    
    if "por favor" in result.call_to_action.lower() and result.tone_score <= 2:
        issues.append(
            "'Por favor' suena demasiado formal para un tono nivel 2. "
            "Sé más directo."
        )
    
    # Criterio 4: Brevedad y claridad
    word_count = len(result.body.split())
    if word_count > 150:
        issues.append(
            f"El email tiene {word_count} palabras. "
            "Reduce a máximo 150 para mayor impacto."
        )
    
    # Si hay problemas, reintentar con feedback consolidado
    if issues:
        feedback = "\n".join(f"• {issue}" for issue in issues)
        raise ModelRetry(
            f"Reflection: El email necesita mejoras:\n{feedback}\n\n"
            "Reescribe el email corrigiendo estos aspectos."
        )
    
    return result

# Prueba el pattern de reflection
result = agent.run_sync(
    "Escribe un email para el equipo de IT informando que habrá mantenimiento "
    "del servidor el próximo sábado de 2am a 6am. Pide que guarden su trabajo antes."
)

email = result.output
print(f"Asunto: {email.subject}")
print(f"Tono: {email.tone_score}/5")
print(f"\nCuerpo:\n{email.body}")
print(f"\nCTA: {email.call_to_action}")
print(f"\nIntentos usados: {result.all_messages_count}")
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/reflection_pattern.py
```

**¿Qué está pasando aquí?**

El reflection pattern crea un ciclo de mejora continua:

```
┌─────────────────────────────────────────┐
│ 1. Agente genera borrador inicial      │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 2. Validator actúa como crítico        │
│    Detecta: asunto genérico, tono      │
│    casual, email largo                  │
└───────────────┬─────────────────────────┘
                │
                ▼ ModelRetry con feedback
┌─────────────────────────────────────────┐
│ 3. Agente lee el feedback y reescribe  │
│    "Ah, debo ser más específico..."    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 4. Validator revisa nuevamente         │
│    Detecta: ahora solo un problema     │
└───────────────┬─────────────────────────┘
                │
                ▼ ModelRetry con feedback
┌─────────────────────────────────────────┐
│ 5. Agente refina basándose en crítica  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 6. Validator: ✓ Todo correcto          │
│    Devuelve resultado al usuario       │
└─────────────────────────────────────────┘
```

**Diferencias clave:**

| Aspecto | Validator básico | Reflection pattern |
|---------|-----------------|-------------------|
| **Qué valida** | Reglas técnicas (longitud, formato) | Calidad subjetiva (tono, claridad) |
| **Feedback** | "Falta campo X" | "El tono no es apropiado porque..." |
| **Iteraciones** | Pocas (1-2) | Varias (3-5) |
| **Uso típico** | Validar datos estructurados | Mejorar contenido creativo |

**Caso de uso real:**

Imagina que necesitas generar 1000 emails de marketing. Sin reflection:
- 30% serían demasiado largos
- 20% tendrían asuntos clickbait
- 15% tendrían CTAs poco claras

Con reflection:
- 95% cumplen todos los criterios de calidad
- Los que no, se marcan para revisión humana
- Ahorraste horas de edición manual

**Cuándo usar reflection:**
- ✅ Generación de contenido (emails, informes, propuestas)
- ✅ Cuando la calidad es más importante que la velocidad
- ✅ Cuando los criterios de "bueno" son subjetivos
- ❌ APIs de baja latencia (cada retry añade tiempo)
- ❌ Tareas con validación objetiva (solo verificar formato)

---

## 3. Configuración avanzada de retries

No todos los errores son iguales. A veces quieres que el agente reintente más veces para ciertas operaciones críticas, y menos veces para operaciones simples.

**¿Por qué configurar retries diferenciados?**  
Imagina un agente de finanzas:
- Validar un presupuesto (crítico): 5 reintentos
- Consultar tipo de cambio (informativo): 1 reintento
- Si ambos tienen los mismos retries, desperdicias recursos

### 3.1. Retries diferenciados por tool

Vamos a construir un sistema donde cada tool tiene su propia configuración de reintentos según su criticidad.

**¿Qué vamos a construir?**  
Un asistente financiero con dos tools:
1. `validate_budget`: Crítica, 3 reintentos (verifica límites departamentales)
2. `get_exchange_rate`: Informativa, 1 reintento (consulta tipo de cambio)

**`01-agentes-avanzados/retries_por_tool.py`:**

```python
from pydantic_ai import Agent, RunContext, Tool, ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

model = AnthropicModel(
    "claude-haiku-4-5",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(
    model,
    retries=1,  # Retries por defecto para el agente
    instructions="Eres un asistente de finanzas que valida presupuestos."
)

@agent.tool(retries=3)  # Esta tool específica tiene 3 reintentos
async def validate_budget(
    ctx: RunContext[None],
    department: str,
    requested_amount: float
) -> dict:
    """
    Valida si un presupuesto solicitado está dentro del límite departamental.
    
    Esta es una operación CRÍTICA que afecta decisiones financieras,
    por eso usa 3 reintentos para maximizar la probabilidad de éxito.
    
    Args:
        department: Nombre del departamento solicitante
        requested_amount: Cantidad solicitada en EUR
    
    Returns:
        Dict con estado de aprobación y presupuesto restante
    """
    # Límites departamentales (en producción vendrían de DB)
    limits = {
        "marketing": 50000,
        "IT": 75000,
        "HR": 30000,
        "operations": 100000
    }
    
    dept_lower = department.lower()
    
    # Si el LLM pasa un departamento inválido, forzar retry
    if dept_lower not in limits:
        raise ModelRetry(
            f"Departamento '{department}' no reconocido. "
            f"Departamentos válidos: {', '.join(limits.keys())}"
        )
    
    limit = limits[dept_lower]
    approved = requested_amount <= limit
    
    return {
        "department": department,
        "requested_eur": requested_amount,
        "limit_eur": limit,
        "approved": approved,
        "remaining_budget_eur": limit - requested_amount if approved else 0
    }

@agent.tool  # Esta tool usa el retry por defecto del agente (1)
def get_exchange_rate(ctx: RunContext[None], currency: str) -> dict:
    """
    Obtiene el tipo de cambio de una moneda a EUR.
    
    Esta es una operación INFORMATIVA. Si falla, no es crítico.
    Usa el retry por defecto (1) para no desperdiciar recursos.
    
    Args:
        currency: Código de moneda (USD, GBP, JPY)
        
    Returns:
        Dict con tipo de cambio actual a EUR
    """
    # Tipos de cambio simulados (en producción, llamarías una API)
    rates = {"USD": 0.92, "GBP": 1.17, "JPY": 0.0063}
    
    if currency not in rates:
        raise ModelRetry(
            f"Moneda '{currency}' no soportada. "
            f"Disponibles: USD, GBP, JPY"
        )
    
    return {
        "currency": currency,
        "rate_to_eur": rates[currency]
    }

# Ejemplo que provoca retry intencional (typo en departamento)
result = agent.run_sync(
    "El departamento de Marketin solicita 45000 EUR. ¿Se aprueba? "
    "(Nota el typo intencional para forzar retry)"
)
print(result.output)
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/retries_por_tool.py
```

**¿Qué está pasando aquí?**

El agente detecta el typo "Marketin" y automáticamente:

1. **Primer intento**: Llama `validate_budget("Marketin", 45000)`
2. **Tool lanza ModelRetry**: "Departamento 'Marketin' no reconocido. Válidos: marketing, IT, HR, operations"
3. **Agente lee el error**: "Ah, los departamentos válidos son estos..."
4. **Segundo intento**: Llama `validate_budget("marketing", 45000)`
5. **Tool responde exitosamente**: `{"approved": True, "remaining_budget_eur": 5000}`

**Configuración de retries en detalle:**

```python
agent = Agent(model, retries=1)  # Default para todo el agente

@agent.tool(retries=3)  # Override para esta tool específica
async def critical_operation(...):
    pass

@agent.tool  # Usa el default del agente (1)
async def simple_operation(...):
    pass
```

**Estrategia de retries recomendada:**

| Tipo de tool | Retries | Justificación |
|--------------|---------|---------------|
| **Validación crítica** (aprobar pagos, modificar DB) | 3-5 | Fallar es costoso |
| **Cálculo complejo** (pronóstico, análisis) | 2-3 | Requiere precisión |
| **Consulta informativa** (clima, noticias) | 1 | No es crítico si falla |
| **Operación experimental** | 0 | Prefiero fallar rápido |

**Beneficios:**
- ✅ **Optimización de costes**: No reintentas innecesariamente
- ✅ **Latencia controlada**: Operaciones simples son más rápidas
- ✅ **Robustez**: Operaciones críticas tienen más oportunidades de éxito

---

### 3.2. Retries en validación de output

Además de retries para tools, puedes configurar retries específicos para cuando la validación del **output final** falla.

**¿Cuándo es útil?**  
Imagina que tu output tiene un schema complejo con muchas validaciones Pydantic. Quieres dar más oportunidades al LLM de generar algo válido sin incrementar los retries de las tools.

**¿Qué vamos a construir?**  
Un sistema de extracción de datos de facturas con validación estricta de formato.

**`01-agentes-avanzados/output_retries.py`:**

```python
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

class InvoiceData(BaseModel):
    """Datos de factura extraídos con validación estricta."""
    invoice_number: str = Field(pattern=r"^INV-\d{4}$")
    client_vat: str = Field(pattern=r"^[A-Z]{2}\d{9}$")
    total_eur: float = Field(gt=0)
    line_items: list[str] = Field(min_length=1)
    
    @field_validator('line_items')
    @classmethod
    def validate_line_items(cls, v):
        """Cada línea debe tener descripción mínima."""
        if any(len(item.strip()) < 5 for item in v):
            raise ValueError("Cada línea debe tener al menos 5 caracteres")
        return v

model = OpenAIChatModel(
    "gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=InvoiceData,
    retries=2,  # Retries para errores generales (tools, timeouts, etc.)
    output_retries=4,  # Retries específicos para validación Pydantic del output
    instructions=(
        "Extraes datos de facturas en formato estructurado. "
        "El número de factura DEBE ser formato INV-XXXX (4 dígitos). "
        "El CIF DEBE ser formato EU: 2 letras país + 9 dígitos."
    )
)

# Simulación de texto OCR de una factura escaneada
invoice_text = """
FACTURA
Nº: 2025-001
Cliente: Tecnologías Acme S.L.
CIF: ES12345678A
Total: 1,250.00 EUR

Líneas:
- Consultoría técnica: 800 EUR
- Soporte mensual: 450 EUR
"""

result = agent.run_sync(
    f"Extrae los datos de esta factura:\n\n{invoice_text}"
)

invoice = result.output
print(f"Factura: {invoice.invoice_number}")
print(f"CIF: {invoice.client_vat}")
print(f"Total: {invoice.total_eur:.2f} EUR")
print(f"Líneas: {len(invoice.line_items)}")
for item in invoice.line_items:
    print(f"  - {item}")
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/output_retries.py
```

**¿Qué está pasando aquí?**

El agente necesita transformar "2025-001" en "INV-2025" y "ES12345678A" en formato válido:

1. **Primer intento**:
   ```json
   {"invoice_number": "2025-001", ...}
   ```
   ❌ Validación Pydantic falla: patrón requiere "INV-XXXX"

2. **Segundo intento** (output_retry):
   ```json
   {"invoice_number": "INV-2025", "client_vat": "ES12345678A", ...}
   ```
   ❌ Validación falla: CIF debe ser 2 letras + 9 dígitos (son 10 chars)

3. **Tercer intento** (output_retry):
   ```json
   {"invoice_number": "INV-2025", "client_vat": "ES123456789", ...}
   ```
   ✅ Pasa todas las validaciones

**Diferencia entre `retries` y `output_retries`:**

```python
agent = Agent(
    model,
    retries=2,        # Para errores de tools, timeouts, excepciones
    output_retries=4  # Para ValidationError de Pydantic en el output
)
```

| Escenario | Usa `retries` | Usa `output_retries` |
|-----------|---------------|----------------------|
| Tool lanza ModelRetry | ✅ | ❌ |
| Tool lanza Exception | ✅ | ❌ |
| Timeout de red | ✅ | ❌ |
| Output no cumple schema Pydantic | ❌ | ✅ |
| Output falla @field_validator | ❌ | ✅ |

**Cuándo usar output_retries alto:**
- ✅ Schema complejo con muchas reglas (facturas, contratos)
- ✅ Patrones de regex estrictos
- ✅ Validaciones custom con @field_validator
- ❌ Schema simple (3-4 campos básicos)

**Pro tip**: Empieza con valores conservadores y ajusta según logs:
```python
# Ver cuántos intentos usó
print(f"Intentos usados: {result.all_messages_count}")

# Si siempre agota output_retries, incrementa o simplifica el schema
# Si siempre pasa en el primer intento, reduce para ahorrar latencia
```

---

## 4. Manejo avanzado de errores y fallbacks

En producción, los errores son inevitables: la API del LLM cae, se agota la cuota, el usuario envía input malicioso, etc. Un sistema empresarial debe manejar estos casos **sin crashear**.

**¿Por qué es crítico?**  
Imagina un chatbot de atención al cliente que muestra un stack trace cuando algo falla. Pérdida de confianza instantánea. Necesitas fallos "elegantes" que mantengan la profesionalidad.

### 4.1. Captura de errores con contexto

Vamos a construir un sistema de evaluación de riesgos que captura diferentes tipos de errores y devuelve respuestas apropiadas para cada caso.

**¿Qué vamos a lograr?**  
Un sistema robusto que diferencia entre:
- Agotamiento de reintentos (el agente intentó pero no pudo generar algo válido)
- Comportamiento inesperado del modelo (respuesta incoherente)
- Errores del sistema (red, API key inválida, etc.)

**`01-agentes-avanzados/error_handling.py`:**

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from config import settings

class RiskAssessment(BaseModel):
    """Evaluación de riesgo financiero."""
    risk_level: str  # high, medium, low
    confidence_score: float  # 0-100
    reasoning: str

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(
    model,
    output_type=RiskAssessment,
    retries=2,
    instructions="Evalúas riesgo financiero de operaciones empresariales."
)

def assess_operation(operation_desc: str) -> dict:
    """
    Evalúa una operación con manejo robusto de errores.
    
    Esta función es el patrón recomendado para APIs de producción:
    - Captura errores específicos antes que genéricos
    - Devuelve dict estructurado (nunca lanza excepciones al caller)
    - Incluye códigos de error para tracking
    - Mensajes user-friendly (sin stack traces)
    
    Returns:
        Dict con 'status' (success/warning/error) y datos/mensaje según el caso
    """
    try:
        result = agent.run_sync(
            f"Evalúa el riesgo de esta operación: {operation_desc}"
        )
        
        assessment = result.output
        
        # Validación post-output: lógica de negocio adicional
        if assessment.confidence_score < 50:
            return {
                "status": "warning",
                "message": "Confianza baja en la evaluación. Requiere revisión manual.",
                "assessment": assessment.model_dump()
            }
        
        return {
            "status": "success",
            "assessment": assessment.model_dump(),
            "tokens_used": result.usage().total_tokens if result.usage() else 0
        }
        
    except ModelRetry as e:
        # El agente agotó todos los reintentos sin éxito
        # Esto significa que el input era ambiguo o el schema muy restrictivo
        return {
            "status": "error",
            "error_type": "max_retries_exceeded",
            "message": (
                f"El agente no pudo generar una evaluación válida "
                f"tras {agent.retries + 1} intentos."
            ),
            "detail": str(e),
            "action": "Reformula la consulta con más detalles específicos"
        }
    
    except UnexpectedModelBehavior as e:
        # El modelo produjo una respuesta que no se pudo parsear
        # Raro pero posible con modelos pequeños o prompts mal diseñados
        return {
            "status": "error",
            "error_type": "unexpected_behavior",
            "message": "El modelo produjo una respuesta inesperada.",
            "detail": str(e),
            "action": "Contacta con soporte técnico con el código: ERR-UMB-001"
        }
    
    except Exception as e:
        # Error genérico: red, API key, rate limits, etc.
        # NUNCA expongas el mensaje de error real al usuario final
        return {
            "status": "error",
            "error_type": "system_error",
            "message": "Error del sistema. Contacta con soporte técnico.",
            "support_code": "ERR-ASSESS-001",
            "action": "Reintenta en 1 minuto. Si persiste, cita el código de soporte."
            # En producción, loguea aquí: logger.error(f"Assessment failed: {e}")
        }

# Pruebas que demuestran diferentes flujos de error
print("=== Caso 1: Operación normal ===")
r1 = assess_operation("Inversión de 50k EUR en bonos del estado alemán a 5 años")
print(f"Status: {r1['status']}")
if r1['status'] == 'success':
    print(f"Riesgo: {r1['assessment']['risk_level']}")
    print(f"Confianza: {r1['assessment']['confidence_score']}%")

print("\n=== Caso 2: Operación de alto riesgo ===")
r2 = assess_operation("Compra de 100k EUR en criptomoneda emergente sin regulación")
print(f"Status: {r2['status']}")
if r2['status'] == 'success':
    print(f"Riesgo: {r2['assessment']['risk_level']}")

print("\n=== Caso 3: Entrada ambigua (puede provocar retry) ===")
r3 = assess_operation("Hacer algo con dinero")
print(f"Status: {r3['status']}")
if r3['status'] == 'error':
    print(f"Tipo de error: {r3['error_type']}")
    print(f"Mensaje: {r3['message']}")
    print(f"Acción: {r3['action']}")
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/error_handling.py
```

**¿Qué acabas de ver?**

Este código implementa el patrón "never throw" para APIs de producción:

```python
# ❌ Mal: lanza excepciones que el caller debe manejar
def assess(op):
    return agent.run_sync(op).output

# ✅ Bien: siempre devuelve un dict estructurado
def assess(op):
    try:
        ...
    except SpecificError as e:
        return {"status": "error", "type": "...", ...}
```

**Anatomía de un error bien manejado:**

```json
{
  "status": "error",
  "error_type": "max_retries_exceeded",
  "message": "El agente no pudo generar una evaluación válida...",
  "action": "Reformula la consulta con más detalles",
  "support_code": "ERR-ASSESS-001"
}
```

**Componentes clave:**
1. **status**: success/warning/error (para decisiones programáticas)
2. **error_type**: Categoría del error (para logging/analytics)
3. **message**: Texto user-friendly (sin jerga técnica)
4. **action**: Qué puede hacer el usuario (empowerment)
5. **support_code**: ID único para que soporte pueda buscar en logs

**Best practices de manejo de errores:**

| ✅ Hacer | ❌ Evitar |
|----------|-----------|
| Capturar excepciones específicas primero | `except Exception: pass` |
| Devolver estructuras consistentes | Mezclar dicts y excepciones |
| Incluir códigos de error únicos | Mensajes genéricos tipo "Error" |
| Loguear detalles técnicos | Exponer stack traces al usuario |
| Sugerir acciones concretas | "Algo salió mal" sin más info |

**Niveles de gravedad:**

```python
# SUCCESS: Todo perfecto
{"status": "success", "assessment": {...}}

# WARNING: Funciona pero con advertencias
{"status": "warning", "message": "Confianza baja...", "assessment": {...}}

# ERROR: Falló, pero sabemos por qué
{"status": "error", "error_type": "max_retries", "action": "Reformula..."}
```

---

### 4.2. Agente con fallback a respuesta por defecto

A veces, no mostrar nada es **peor** que mostrar una respuesta segura pero genérica. Esto es especialmente cierto en experiencias de cara al cliente.

**¿Cuándo usar fallbacks?**
- ✅ Recomendaciones de productos (mejor algo genérico que nada)
- ✅ Respuestas de FAQ (tienes una respuesta "catchall")
- ❌ Transacciones financieras (fallar es más seguro que inventar)
- ❌ Diagnósticos médicos (nunca adivines)

**¿Qué vamos a construir?**  
Un sistema de recomendaciones que tiene un "plan B" si el agente falla.

**`01-agentes-avanzados/fallback_pattern.py`:**

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

class ProductRecommendation(BaseModel):
    """Recomendación de producto."""
    product_name: str
    reason: str
    confidence: float  # 0-1

model = OpenAIChatModel(
    "gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=ProductRecommendation,
    retries=2,
    instructions="Recomiendas productos basándote en necesidades del cliente."
)

def get_recommendation_with_fallback(customer_query: str) -> ProductRecommendation:
    """
    Obtiene recomendación con fallback a producto por defecto si falla.
    
    Patrón recomendado para sistemas donde mostrar algo genérico
    es mejor que no mostrar nada.
    
    El fallback es "honesto": indica explícitamente que es una
    recomendación por defecto debido a problemas técnicos.
    """
    try:
        result = agent.run_sync(customer_query, timeout=10.0)  # Timeout de 10s
        return result.output
    
    except Exception as e:
        # Si TODO falla, devuelve una recomendación segura
        print(f"⚠️ Agente falló: {e}")
        print("→ Usando recomendación por defecto")
        
        return ProductRecommendation(
            product_name="Pack de Inicio",
            reason=(
                "Recomendación por defecto debido a indisponibilidad temporal del servicio. "
                "Este pack cubre las necesidades básicas más comunes."
            ),
            confidence=0.5  # Honestidad: confianza media
        )

# Pruebas
print("=== Recomendación normal ===")
rec1 = get_recommendation_with_fallback("Necesito un portátil para diseño gráfico")
print(f"Producto: {rec1.product_name}")
print(f"Razón: {rec1.reason}")
print(f"Confianza: {rec1.confidence:.0%}")

# Para simular un fallo, podrías temporalmente:
# 1. Usar una API key inválida
# 2. Desconectar internet
# 3. Usar un timeout muy bajo: timeout=0.001
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/fallback_pattern.py
```

**¿Qué está pasando aquí?**

El patrón de fallback es simple pero poderoso:

```python
try:
    # Intenta lo ideal
    return agent.run_sync(query)
except:
    # Si falla, devuelve algo seguro
    return default_recommendation
```

**Diseño del fallback:**

Un buen fallback debe ser:

1. **Seguro**: Nunca causa daño (por eso el Pack de Inicio es genérico)
2. **Honesto**: Indica que es un fallback (`confidence=0.5`, mensaje explícito)
3. **Útil**: Mejor que un error (el usuario al menos tiene algo)
4. **Logueable**: El sistema registra que se usó el fallback para análisis

**Ejemplo de uso en producción:**

```python
# E-commerce: fallback a productos más vendidos
fallback = ProductRecommendation(
    product_name="Pack Bestseller",
    reason="Basado en productos más vendidos este mes",
    confidence=0.6
)

# Servicio de soporte: fallback a FAQ
fallback = FAQResponse(
    question="Pregunta general",
    answer="Visita nuestra sección de ayuda en help.ejemplo.com",
    confidence=0.4
)

# Newsletter: fallback a contenido genérico
fallback = EmailContent(
    subject="Novedades de este mes",
    body="Descubre las últimas actualizaciones...",
    confidence=0.5
)
```

**Cuándo NO usar fallback:**

```python
# ❌ NUNCA uses fallback en:

# Transacciones financieras
def transfer_money(...):
    try:
        return agent.run_sync(...)
    except:
        return {"status": "error"}  # NO inventes una transferencia

# Diagnósticos médicos
def diagnose(...):
    try:
        return agent.run_sync(...)
    except:
        return {"error": "Consulta a un médico"}  # NO adivines diagnósticos

# Operaciones destructivas
def delete_user(...):
    try:
        return agent.run_sync(...)
    except:
        raise  # Falla explícitamente, no asumas nada
```

**Regla de oro**: Usa fallback solo cuando el "plan B" es **objetivamente seguro** para el usuario.

---

## 5. Tools con dependencias y contexto

Hasta ahora, las tools han sido funciones autocontenidas. En producción, necesitas que accedan a recursos externos: bases de datos, configuración, servicios, etc.

**El sistema de dependencias** de PydanticAI te permite inyectar estos recursos de forma controlada y typesafe.

**¿Por qué es importante?**  
Separa las concerns: tu tool no "sabe" cómo conectarse a la DB, solo recibe una conexión ya configurada. Esto facilita testing (inyectas mocks) y mantenimiento (cambias la DB sin tocar las tools).

### 5.1. Tool con acceso a base de datos simulada

Vamos a construir un asistente de RRHH que consulta datos de empleados y presupuestos departamentales usando dependencias inyectadas.

**¿Qué vamos a lograr?**  
Un sistema donde:
- Las tools acceden a una "base de datos" a través de `ctx.deps`
- El agente registra todas las consultas para auditoría
- Puedes cambiar fácilmente entre DB real, mock o test

**`01-agentes-avanzados/tools_con_deps.py`:**

```python
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from config import settings

# Simulación de conexión a base de datos
# En producción, esto sería SQLAlchemy, psycopg2, etc.
@dataclass
class DatabaseConnection:
    """Simula una conexión a base de datos empresarial."""
    
    def query_employee(self, employee_id: int) -> dict | None:
        """Consulta datos de un empleado por ID."""
        employees = {
            101: {"name": "Ana García", "dept": "Engineering", "salary": 65000},
            102: {"name": "Carlos Ruiz", "dept": "Sales", "salary": 55000},
            103: {"name": "María López", "dept": "Marketing", "salary": 58000},
        }
        return employees.get(employee_id)
    
    def get_department_budget(self, dept: str) -> float:
        """Obtiene el presupuesto disponible de un departamento."""
        budgets = {
            "Engineering": 450000,
            "Sales": 320000,
            "Marketing": 180000,
        }
        return budgets.get(dept, 0)
    
    def log_query(self, query_type: str, user: str):
        """Registra las consultas para auditoría."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOG] {timestamp} | User: {user} | Query: {query_type}")

# Deps que se inyectarán en el agente
@dataclass
class HRDependencies:
    """Dependencias que el agente necesita para funcionar."""
    db: DatabaseConnection
    current_user: str

model = AnthropicModel(
    "claude-sonnet-4-5-20250929",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key)
)

agent = Agent(
    model,
    deps_type=HRDependencies,  # Declara qué tipo de deps espera
    instructions=(
        "Eres un asistente de RRHH con acceso a la base de datos de empleados. "
        "Responde consultas sobre empleados y presupuestos departamentales."
    )
)

@agent.tool
async def get_employee_info(
    ctx: RunContext[HRDependencies],  # RunContext ahora está tipado
    employee_id: int
) -> dict:
    """
    Obtiene información de un empleado por su ID.
    
    Esta tool accede a la base de datos a través de ctx.deps.db
    y registra la consulta para auditoría.
    
    Args:
        employee_id: ID numérico del empleado.
        
    Returns:
        Dict con nombre, departamento y salario.
    """
    # Acceso a la base de datos a través de las dependencias
    ctx.deps.db.log_query("employee_info", ctx.deps.current_user)
    
    employee = ctx.deps.db.query_employee(employee_id)
    if not employee:
        return {"error": f"Empleado {employee_id} no encontrado"}
    
    return {
        "employee_id": employee_id,
        "name": employee["name"],
        "department": employee["dept"],
        "salary_eur": employee["salary"]
    }

@agent.tool
async def check_department_budget(
    ctx: RunContext[HRDependencies],
    department: str
) -> dict:
    """
    Consulta el presupuesto disponible de un departamento.
    
    Args:
        department: Nombre del departamento.
        
    Returns:
        Dict con presupuesto total y disponible.
    """
    ctx.deps.db.log_query("dept_budget", ctx.deps.current_user)
    
    budget = ctx.deps.db.get_department_budget(department)
    if budget == 0:
        return {"error": f"Departamento '{department}' no encontrado"}
    
    return {
        "department": department,
        "total_budget_eur": budget,
        "available_eur": budget * 0.73  # Simular que se ha gastado el 27%
    }

# Uso del agente con dependencias inyectadas
db = DatabaseConnection()
deps = HRDependencies(db=db, current_user="admin@empresa.com")

result = agent.run_sync(
    "¿Cuál es el salario del empleado 102 y cuánto presupuesto queda en su departamento?",
    deps=deps  # Inyectamos las dependencias aquí
)
print(result.output)
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/tools_con_deps.py
```

**¿Qué acabas de ver?**

El patrón de dependency injection en acción:

```
┌─────────────────────────────────────────┐
│ 1. Preparas las dependencias            │
│    db = DatabaseConnection()            │
│    deps = HRDependencies(db, user)      │
└───────────────┬─────────────────────────┘
                │
                ▼ Inyectas via run_sync(deps=...)
┌─────────────────────────────────────────┐
│ 2. El agente recibe la consulta         │
│    "¿Salario del empleado 102?"         │
└───────────────┬─────────────────────────┘
                │
                ▼ Decide llamar la tool
┌─────────────────────────────────────────┐
│ 3. Tool accede a deps via ctx.deps      │
│    ctx.deps.db.query_employee(102)      │
│    ctx.deps.db.log_query(...)           │
└───────────────┬─────────────────────────┘
                │
                ▼ Devuelve datos
┌─────────────────────────────────────────┐
│ 4. Agente formula respuesta natural     │
│    "Carlos Ruiz gana 55,000 EUR..."     │
└─────────────────────────────────────────┘
```

**Ventajas del patrón:**

1. **Separación de concerns**: La tool no sabe cómo conectarse a la DB
2. **Testing fácil**: Inyectas un mock en lugar de la DB real
3. **Type safety**: `RunContext[HRDependencies]` te da autocomplete
4. **Seguridad**: Controlas qué tools tienen acceso a qué recursos
5. **Auditoría**: Todas las operaciones pueden loguearse centralizadamente

**Cómo lo usarías en producción:**

```python
# Producción: DB real
from sqlalchemy import create_engine

class ProductionDB:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
    
    def query_employee(self, employee_id):
        with self.engine.connect() as conn:
            result = conn.execute(
                "SELECT * FROM employees WHERE id = %s",
                employee_id
            )
            return result.fetchone()

# Testing: Mock
class MockDB:
    def query_employee(self, employee_id):
        return {"name": "Test User", "dept": "Test", "salary": 50000}

# Uso
db = ProductionDB("postgresql://...") if PROD else MockDB()
deps = HRDependencies(db=db, current_user=current_user)
agent.run_sync(query, deps=deps)
```

**Otras cosas que puedes inyectar:**

```python
@dataclass
class AppDependencies:
    db: DatabaseConnection       # Base de datos
    cache: RedisClient           # Cache
    api_keys: dict               # Credenciales externas
    config: AppConfig            # Configuración
    logger: logging.Logger       # Logging
    current_user: User           # Usuario autenticado
    request_id: str              # ID para tracing
```

Esto te da control total sobre qué recursos tiene el agente sin contaminar el código de las tools con detalles de infraestructura.

---

## 6. Instrucciones dinámicas avanzadas

Las instrucciones estáticas (strings fijos) son limitadas. En producción, necesitas que el comportamiento del agente cambie según contexto: rol del usuario, hora del día, configuración del tenant, etc.

**¿Por qué dinámicas?**  
Imagina un agente de atención al cliente en una empresa con múltiples marcas. Cada marca tiene:
- Tono diferente (formal vs casual)
- Productos diferentes
- Políticas diferentes

No quieres un agente por marca. Quieres un agente que adapta su comportamiento dinámicamente.

**¿Qué vamos a construir?**  
Un agente corporativo con control de acceso basado en roles (RBAC) que ajusta sus respuestas según el rol del usuario.

**`01-agentes-avanzados/dynamic_instructions_advanced.py`:**

```python
from dataclasses import dataclass
from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

@dataclass
class UserContext:
    """Contexto del usuario actual."""
    user_id: str
    role: str  # "admin", "manager", "employee"
    department: str
    tenure_years: int

model = OpenAIChatModel(
    "gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    deps_type=UserContext,
    instructions="Eres un asistente corporativo. Las instrucciones específicas se cargan dinámicamente."
)

@agent.instructions
def get_role_based_instructions(ctx: RunContext[UserContext]) -> str:
    """
    Genera instrucciones dinámicas basadas en el rol y contexto del usuario.
    
    Este es un ejemplo de RBAC (Role-Based Access Control) implementado
    a nivel de prompt. El agente sabe qué información puede compartir
    según el rol del usuario.
    """
    user = ctx.deps
    current_hour = datetime.now().hour
    
    # Instrucciones base (siempre incluidas)
    base = f"Usuario actual: {user.user_id} | Rol: {user.role} | Dept: {user.department}"
    
    # Permisos según rol
    if user.role == "admin":
        permissions = (
            "Tienes acceso completo a todos los datos y operaciones. "
            "Puedes proporcionar información sensible de cualquier departamento."
        )
    elif user.role == "manager":
        permissions = (
            f"Tienes acceso a datos de tu departamento ({user.department}). "
            "No puedes proporcionar información de otros departamentos sin autorización."
        )
    else:  # employee
        permissions = (
            "Tienes acceso limitado. Sólo puedes consultar información general "
            "y tus propios datos. No proporciones información de otros empleados."
        )
    
    # Ajuste por horario (seguridad: limita operaciones críticas fuera de horario)
    if current_hour < 9 or current_hour > 18:
        time_note = (
            "Es fuera del horario laboral. "
            "Limita las respuestas a información no crítica."
        )
    else:
        time_note = "Es horario laboral normal."
    
    # Ajuste por antigüedad (personalización: más detalle para nuevos)
    if user.tenure_years < 1:
        tone = (
            "Usa un tono más didáctico y explica los procesos internos "
            "cuando sea relevante."
        )
    else:
        tone = "Usa un tono profesional directo."
    
    return f"{base}\n\nPermisos: {permissions}\n\n{time_note}\n\n{tone}"

# Pruebas con diferentes contextos
print("=== Consulta como ADMIN ===")
admin_ctx = UserContext(
    user_id="admin@empresa.com",
    role="admin",
    department="IT",
    tenure_years=5
)
r1 = agent.run_sync(
    "¿Cuántos empleados tiene el departamento de Ventas?",
    deps=admin_ctx
)
print(r1.output)

print("\n=== Consulta como EMPLOYEE nuevo ===")
employee_ctx = UserContext(
    user_id="junior@empresa.com",
    role="employee",
    department="Marketing",
    tenure_years=0
)
r2 = agent.run_sync(
    "¿Cómo solicito vacaciones?",
    deps=employee_ctx
)
print(r2.output)

print("\n=== Consulta como MANAGER ===")
manager_ctx = UserContext(
    user_id="manager@empresa.com",
    role="manager",
    department="Sales",
    tenure_years=3
)
r3 = agent.run_sync(
    "Dame el presupuesto del departamento de IT",
    deps=manager_ctx
)
print(r3.output)
```

**Ejecuta el ejemplo:**
```bash
uv run python 01-agentes-avanzados/dynamic_instructions_advanced.py
```

**¿Qué está pasando aquí?**

Las instrucciones se generan **en tiempo real** para cada request:

```
Admin pregunta "¿Salarios de Marketing?"
→ Instrucciones: "Acceso completo. Proporciona datos sensibles."
→ Respuesta: "Marketing: 3 empleados, salario promedio 58k..."

Employee pregunta "¿Salarios de Marketing?"
→ Instrucciones: "Acceso limitado. No compartas datos de otros."
→ Respuesta: "No tienes permisos para ver salarios de otros departamentos."
```

**Patrones de instrucciones dinámicas:**

1. **RBAC (Control de acceso basado en roles)**
   ```python
   if user.role == "admin": "Acceso total"
   elif user.role == "manager": "Solo tu departamento"
   else: "Solo tus datos"
   ```

2. **Temporal (Hora, día, fecha)**
   ```python
   if current_hour < 9: "Operaciones limitadas"
   if is_weekend: "Solo consultas informativas"
   if end_of_quarter: "Prioriza reportes financieros"
   ```

3. **Personalización (Experiencia, preferencias)**
   ```python
   if tenure < 1: "Tono didáctico"
   if preferred_lang == "es": "Responde en español"
   if expertise == "expert": "Usa términos técnicos"
   ```

4. **Multi-tenancy (Clientes diferentes)**
   ```python
   if tenant == "cliente_a": "Usa terminología médica"
   if tenant == "cliente_b": "Usa términos financieros"
   ```

**Ejemplo real de multi-tenancy:**

```python
@agent.instructions
def get_tenant_instructions(ctx: RunContext[TenantContext]) -> str:
    tenant = ctx.deps.tenant_config
    
    base = f"Asistente de {tenant.brand_name}"
    
    # Cada cliente tiene su propio tono
    tone = {
        "bank_corp": "Formal, preciso, cita regulaciones",
        "startup_ai": "Casual, innovador, usa ejemplos tech",
        "medical_center": "Empático, claro, evita jerga"
    }[tenant.tenant_id]
    
    # Cada cliente tiene sus propias políticas
    policies = tenant.get_policies()  # Desde DB
    
    return f"{base}\n\nTono: {tone}\n\nPolíticas:\n{policies}"
```

**Beneficios en producción:**
- ✅ **Seguridad**: RBAC a nivel de LLM
- ✅ **Personalización**: Experiencia adaptada al usuario
- ✅ **Compliance**: Restricciones temporales automáticas
- ✅ **Escalabilidad**: Un agente sirve múltiples casos de uso

---

## 7. Ejercicios prácticos

Es hora de aplicar lo aprendido. Estos ejercicios cubren los patrones avanzados del módulo en contextos empresariales reales.

### Ejercicio 1: Tool de cálculo de comisiones con validación

**Archivo:** `01-agentes-avanzados/ej01_comisiones.py`  
**Tiempo estimado:** 25 minutos

**Contexto empresarial:**  
Eres el desarrollador de un sistema de compensaciones para el equipo de ventas. Necesitas un agente que calcule comisiones según el tier del vendedor, pero con validación estricta para evitar pagos erróneos.

**Objetivo:** Crea un agente que calcule comisiones de ventas con:
- Tool `calculate_commission(sales_amount: float, tier: str)` donde tier ∈ {bronze, silver, gold}
- Tasas de comisión: Bronze 3%, Silver 5%, Gold 8%
- Result validator que verifique que las comisiones no excedan 10,000 EUR
- Si exceden, debe lanzar `ModelRetry` sugiriendo revisar el tier

**Schema del output:**
```python
class CommissionResult(BaseModel):
    sales_amount_eur: float
    tier: str
    commission_rate: float  # Como decimal (0.03, 0.05, 0.08)
    commission_eur: float
    approved: bool
```

**Criterio de éxito:**
- ✅ La tool valida el tier con `Literal["bronze", "silver", "gold"]`
- ✅ El validator detecta comisiones > 10k y fuerza retry con feedback específico
- ✅ El agente responde con el `CommissionResult` estructurado
- ✅ Prueba con: 80k EUR Gold (debería aprobar), 150k EUR Gold (debería pedir revisar tier)

**Pista:** Usa `@agent.result_validator` para verificar `commission_eur > 10000`.

---

### Ejercicio 2: Reflection para generación de metadescripciones SEO

**Archivo:** `01-agentes-avanzados/ej02_seo_reflection.py`  
**Tiempo estimado:** 30 minutos

**Contexto empresarial:**  
Trabajas en una agencia de marketing digital que genera cientos de metadescripciones para sitios web de clientes. Necesitas automatizar la generación con garantía de calidad SEO.

**Objetivo:** Agente que genera metadescripciones con reflection automático:

**Schema del output:**
```python
class MetaDescription(BaseModel):
    text: str = Field(min_length=140, max_length=160)
    char_count: int
    keyword_density: float  # Veces que aparece keyword / total palabras
```

**Validaciones del reflection (en `@agent.result_validator`):**
1. Longitud: 140-160 caracteres (óptimo para Google)
2. Debe contener la keyword principal al menos 1 vez
3. No debe contener palabras genéricas prohibidas: "increíble", "mejor", "único", "revolucionario"
4. El tono debe ser profesional (no clickbait)

**Criterio de éxito:**
- ✅ Si la descripción no cumple, el validator da feedback específico sobre QUÉ falla
- ✅ El agente reescribe hasta cumplir todos los criterios o agotar 3 retries
- ✅ Prueba con keyword: "zapatos deportivos" y página: "tienda online de running"
- ✅ Imprime el número de intentos usados

**Pista:** Para contar palabras: `len(text.split())`. Para calcular keyword_density: `text.lower().count(keyword) / len(text.split())`.

---

### Ejercicio 3: Multi-tool con orquestación de inventario y pedidos

**Archivo:** `01-agentes-avanzados/ej03_multi_tool_warehouse.py`  
**Tiempo estimado:** 35 minutos

**Contexto empresarial:**  
Construyes un sistema de gestión de almacén para una empresa de e-commerce. El agente debe poder consultar stock, reservar unidades y estimar entregas, decidiendo automáticamente qué operaciones realizar según la consulta del usuario.

**Objetivo:** Sistema de gestión de almacén con 3 tools que el agente orquesta:

1. `check_stock(product_code: str) -> dict`
   - Consulta stock disponible y precio
   - Devuelve: {product_code, name, stock_units, price_eur, available}

2. `reserve_stock(product_code: str, quantity: int) -> dict`
   - Reserva unidades (solo si hay stock suficiente)
   - Devuelve: {reserved, quantity, remaining_stock}

3. `estimate_delivery(product_code: str, destination: str) -> dict`
   - Estima tiempo de entrega según destino
   - Destination ∈ {local, national, international}
   - Devuelve: {estimated_days, shipping_cost_eur}

**Base de datos mock:**
```python
WAREHOUSE = {
    "PROD-001": {"name": "Laptop Pro", "stock": 5, "price": 1299},
    "PROD-002": {"name": "Mouse", "stock": 0, "price": 29},
    "PROD-003": {"name": "Monitor", "stock": 15, "price": 449},
}
```

**Criterio de éxito:**
- ✅ El agente verifica stock ANTES de intentar reservar
- ✅ Si no hay stock, informa sin llamar `reserve_stock`
- ✅ Tras reserva exitosa, automáticamente estima entrega
- ✅ Prueba: "Quiero 3 Laptop Pro para envío nacional" (debería verificar → reservar → estimar)
- ✅ Prueba: "¿Hay Mouse disponible?" (debería solo verificar y decir que está agotado)

**Pista:** El agente decide automáticamente el flujo. Tu trabajo es implementar las tools correctamente y dejar que el LLM orqueste.

---

### Ejercicio 4: Tool con deps para acceso a API externa

**Archivo:** `01-agentes-avanzados/ej04_tool_api_deps.py`  
**Tiempo estimado:** 30 minutos

**Contexto empresarial:**  
Integras un agente con una API de tipos de cambio para un dashboard financiero. Necesitas diseñar la arquitectura para que el código sea testeable y el API key esté controlado.

**Objetivo:** Agente que consulta tipos de cambio usando dependency injection:

**Definir deps:**
```python
@dataclass
class CurrencyAPIDeps:
    api_key: str
    base_url: str
```

**Tool:**
```python
@agent.tool
async def get_exchange_rate(
    ctx: RunContext[CurrencyAPIDeps],
    from_currency: str,
    to_currency: str
) -> dict:
    """
    Obtiene el tipo de cambio entre dos divisas.
    
    Usa ctx.deps.api_key y ctx.deps.base_url para simular
    una llamada a API externa (no hace request real en este ejercicio).
    """
    # Simular respuesta de API
    rates = {
        ("EUR", "USD"): 1.09,
        ("USD", "EUR"): 0.92,
        ("EUR", "GBP"): 0.86,
        ("GBP", "EUR"): 1.17,
        # ... etc
    }
    rate = rates.get((from_currency, to_currency))
    if not rate:
        raise ModelRetry(f"Conversión {from_currency}->{to_currency} no disponible")
    
    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "api_used": ctx.deps.base_url  # Para demostrar que accede a deps
    }
```

**Criterio de éxito:**
- ✅ La tool recibe deps a través de `RunContext[CurrencyAPIDeps]`
- ✅ El agente responde preguntas tipo "¿Cuántos dólares son 100 euros?"
- ✅ El código muestra cómo podrías reemplazar la simulación con una llamada real
- ✅ Incluye un comentario explicando cómo harías testing con un mock

**Pista:** En producción harías: `response = requests.get(f"{ctx.deps.base_url}/rates", headers={"API-Key": ctx.deps.api_key})`.

---

### Ejercicio 5: Error handling completo con fallback

**Archivo:** `01-agentes-avanzados/ej05_error_handling_completo.py`  
**Tiempo estimado:** 35 minutos

**Contexto empresarial:**  
Desarrollas un chatbot de atención al cliente que debe NUNCA mostrar errores técnicos al usuario. Todo fallo debe manejarse elegantemente con mensajes profesionales y escalación cuando sea necesario.

**Objetivo:** Agente de atención al cliente con manejo robusto de errores:

**Schema del output:**
```python
class CustomerResponse(BaseModel):
    message: str  # Respuesta al cliente
    ticket_id: str | None = None  # ID de ticket si se creó
    escalated: bool  # Si se escaló a humano
```

**Implementar función wrapper:**
```python
def handle_customer_query(query: str) -> dict:
    """
    Maneja consulta con 3 niveles de error catching.
    
    Returns siempre un dict estructurado, nunca lanza excepciones.
    """
    try:
        result = agent.run_sync(query)
        return {
            "status": "success",
            "response": result.output.model_dump()
        }
    except ModelRetry as e:
        # Agotó retries: escalar a humano
        return {
            "status": "escalated",
            "response": CustomerResponse(
                message="He transferido tu consulta a un agente humano que te contactará pronto.",
                ticket_id=f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                escalated=True
            ).model_dump()
        }
    except UnexpectedModelBehavior as e:
        # Comportamiento raro del modelo
        return {
            "status": "error",
            "response": CustomerResponse(
                message="Disculpa, estoy experimentando dificultades técnicas. Un supervisor revisará tu caso.",
                ticket_id=f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                escalated=True
            ).model_dump(),
            "internal_error": "UMB",
            "support_code": "ERR-UMB-001"
        }
    except Exception as e:
        # Error genérico (red, API, etc.)
        return {
            "status": "system_error",
            "response": CustomerResponse(
                message="Lo sentimos, estamos experimentando problemas técnicos. Por favor, reintenta en unos minutos.",
                ticket_id=f"SYS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                escalated=False
            ).model_dump(),
            "support_code": "ERR-SYS-001"
        }
```

**Criterio de éxito:**
- ✅ Maneja 3 tipos de error diferentes con mensajes específicos
- ✅ Nunca expone errores técnicos al usuario (stack traces, nombres de excepciones)
- ✅ Incluye códigos de error únicos para que soporte pueda buscar en logs
- ✅ Genera ticket_id en caso de escalación
- ✅ Prueba: consulta normal, consulta que causa retry, consulta con API key inválida

---

### Ejercicio 6: Reflection + retries diferenciados

**Archivo:** `01-agentes-avanzados/ej06_reflection_retries.py`  
**Tiempo estimado:** 40 minutos

**Contexto empresarial:**  
Generas reportes ejecutivos automáticos para CEOs. Estos reportes deben ser impecables: tono profesional, métricas claras, recomendaciones accionables. Usarás reflection avanzado con configuración de retries optimizada.

**Objetivo:** Agente que genera reportes ejecutivos con reflection y retries configurados:

**Schema del output:**
```python
class ExecutiveReport(BaseModel):
    title: str = Field(max_length=80)
    summary: str = Field(min_length=100, max_length=250)  # Palabras, no chars
    key_metrics: list[str] = Field(min_length=3, max_length=5)
    recommendations: list[str] = Field(min_length=2, max_length=4)
```

**Configuración:**
```python
agent = Agent(
    model,
    output_type=ExecutiveReport,
    retries=2,         # Para errores de tools
    output_retries=4   # Para validación del output
)
```

**Result validator que verifica:**
1. **Title** < 80 caracteres
2. **Summary** entre 100-250 PALABRAS (no caracteres)
3. Al menos **3 key_metrics** específicas (no vagas tipo "mejora general")
4. Cada **recommendation** debe ser accionable (contener un verbo de acción: "implementar", "reducir", "contratar", etc.)

**Criterio de éxito:**
- ✅ El validator da feedback específico para cada criterio no cumplido
- ✅ El agente itera hasta cumplir todos o agotar los 4 output_retries
- ✅ Imprime el número total de intentos usados: `result.all_messages_count`
- ✅ Prueba con: "Genera reporte Q4 2024: ventas +15%, costes -8%, NPS 72"
- ✅ El reporte final debe pasar TODAS las validaciones

**Pista para validar verbos de acción:**
```python
action_verbs = {"implementar", "reducir", "aumentar", "contratar", "optimizar", 
                "desarrollar", "lanzar", "eliminar", "priorizar", "invertir"}
has_action = any(verb in rec.lower() for verb in action_verbs)
```

---

## 8. Troubleshooting y mejores prácticas

Esta sección recopila problemas comunes que encontrarás al desarrollar agentes avanzados y cómo resolverlos.

### Errores comunes y soluciones

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| **`ModelRetry` sin mensaje útil** | Lanzas `ModelRetry()` vacío | Siempre incluye mensaje específico: `ModelRetry("El campo X debe ser Y porque Z")` |
| **Retries infinitos / timeout** | Validator demasiado estricto | Añade logging en el validator para ver qué falla. Revisa si tus criterios son realistas |
| **Tool nunca se llama** | Docstring poco claro o confuso | El LLM decide basándose en el docstring. Hazlo más específico con ejemplos de cuándo usarla |
| **Tool muy lenta** | Operaciones síncronas bloqueantes (DB, API) | Usa `async def` y `await` correctamente para operaciones I/O |
| **ValidationError inesperado** | Schema del output demasiado restrictivo | Empieza simple y añade validaciones gradualmente. Prueba el schema manualmente primero |
| **Agente ignora instrucciones dinámicas** | `@agent.instructions` no devuelve string | Verifica que tu función retorna un `str`, no None o un objeto |
| **Deps no llegan a la tool** | Olvidaste pasar `deps` en `run_sync()` | Siempre: `agent.run_sync(query, deps=my_deps)` |
| **"Cannot pickle..." en tools async** | Usas objetos no serializables en deps | Usa dataclasses simples en deps, no objetos complejos con estado |

### Best practices de desarrollo

#### 1. Desarrollo iterativo de validators

No escribas validaciones complejas desde el principio. Empieza simple:

```python
# ❌ Mal: validador complejo desde el inicio
@agent.result_validator
async def validate(ctx, result):
    # 20 líneas de validaciones anidadas
    ...

# ✅ Bien: empieza simple, añade complejidad gradualmente
@agent.result_validator
async def validate_v1(ctx, result):
    # Solo valida longitud
    if len(result.text) > 280:
        raise ModelRetry("Máximo 280 caracteres")
    return result

# Luego añades más validaciones una por una
@agent.result_validator
async def validate_v2(ctx, result):
    if len(result.text) > 280:
        raise ModelRetry("Máximo 280 caracteres")
    
    # Nueva validación
    if not result.text[0].isupper():
        raise ModelRetry("La primera letra debe ser mayúscula")
    
    return result
```

#### 2. Logging estratégico

En producción necesitas entender qué está pasando. Añade logs en puntos clave:

```python
@agent.result_validator
async def validate(ctx, result):
    logger.info(f"Validando resultado: {result.model_dump()}")
    
    if result.score < 50:
        logger.warning(f"Score bajo: {result.score}, forzando retry")
        raise ModelRetry("Score debe ser ≥ 50")
    
    logger.info("Validación exitosa")
    return result

@agent.tool
async def critical_operation(ctx, amount):
    logger.info(f"[{ctx.deps.user_id}] Ejecutando operación crítica: {amount} EUR")
    
    try:
        result = process_payment(amount)
        logger.info(f"Operación exitosa: {result}")
        return result
    except Exception as e:
        logger.error(f"Falló operación: {e}", exc_info=True)
        raise
```

#### 3. Testing de tools

Las tools deben ser testeables independientemente del agente:

```python
# Tool
@agent.tool
async def calculate_tax(ctx: RunContext[None], amount: float) -> dict:
    """Calcula IVA 21%."""
    return {
        "amount": amount,
        "tax": round(amount * 0.21, 2),
        "total": round(amount * 1.21, 2)
    }

# Test (puedes usar pytest)
def test_calculate_tax():
    # Crea un contexto mock
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = None
    
    # Llama la tool directamente
    result = asyncio.run(calculate_tax(mock_ctx, 100.0))
    
    assert result["tax"] == 21.0
    assert result["total"] == 121.0
```

#### 4. Configuración de retries basada en criticidad

Establece una política clara:

```python
# Archivo: retry_policy.py
class RetryPolicy:
    """Política de reintentos según criticidad."""
    
    # Tools críticas (afectan dinero, datos sensibles)
    CRITICAL = 5
    
    # Tools importantes (afectan experiencia de usuario)
    HIGH = 3
    
    # Tools normales (consultas, cálculos)
    NORMAL = 2
    
    # Tools experimentales o informativas
    LOW = 1

# Uso
@agent.tool(retries=RetryPolicy.CRITICAL)
async def process_payment(...):
    pass

@agent.tool(retries=RetryPolicy.LOW)
async def get_weather(...):
    pass
```

#### 5. Docstrings efectivos para tools

El LLM decide cuándo usar una tool basándose en su docstring. Hazlos específicos:

```python
# ❌ Mal: docstring vago
@agent.tool
def get_data(ctx, id: int):
    """Gets data."""
    pass

# ✅ Bien: docstring descriptivo
@agent.tool
def get_employee_salary(ctx, employee_id: int) -> dict:
    """
    Obtiene el salario actual de un empleado por su ID.
    
    Úsala cuando el usuario pregunte sobre:
    - Cuánto gana un empleado
    - El salario de una persona
    - Compensación de un miembro del equipo
    
    NO la uses para:
    - Salario promedio del departamento (usa get_department_stats)
    - Historial de salarios (usa get_salary_history)
    
    Args:
        employee_id: ID numérico del empleado (ej: 101, 102).
        
    Returns:
        Dict con 'employee_id', 'name', 'salary_eur' y 'currency'.
    """
    pass
```

#### 6. Manejo de deps complejas

Si tus deps son complejas, crea una factory:

```python
# ❌ Mal: crear deps manualmente cada vez
db = create_db_connection()
cache = create_cache_client()
logger = setup_logger()
deps = AppDeps(db=db, cache=cache, logger=logger, user=user, ...)

# ✅ Bien: factory que centraliza la lógica
class DepsFactory:
    @staticmethod
    def create_for_user(user: User) -> AppDeps:
        return AppDeps(
            db=get_db_connection(),
            cache=get_cache_client(),
            logger=get_logger(),
            user=user,
            request_id=generate_request_id(),
            config=load_config()
        )

# Uso limpio
deps = DepsFactory.create_for_user(current_user)
result = agent.run_sync(query, deps=deps)
```

---

## 9. Caso de uso: Evolución del DataPulse AI

Ahora que dominas las técnicas avanzadas, vamos a evolucionar el DataPulse AI del Módulo 0 en un sistema robusto de producción.

### Mejoras implementadas en v2

**Comparación Módulo 0 vs Módulo 1:**

| Aspecto | Módulo 0 (Básico) | Módulo 1 (Avanzado) |
|---------|-------------------|---------------------|
| **Validación** | Solo Pydantic básico | Result validator + reflection |
| **Manejo de errores** | `try/except` simple | Captura diferenciada por tipo |
| **Calidad del output** | Sin garantías | Autocorrección automática |
| **Retries** | Configuración global | Diferenciados por criticidad |
| **Métricas** | Texto libre | Estructuradas y verificadas |

**`01-agentes-avanzados/datapulse_v2.py`:**

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

class BusinessInsight(BaseModel):
    """Respuesta estructurada con insight empresarial validado."""
    intencion: Literal["resumen", "comparativa", "forecast"]
    respuesta: str = Field(min_length=50, max_length=300)
    metricas_clave: list[str] = Field(min_length=2, max_length=4)
    nivel_confianza: float = Field(ge=0, le=1, description="0-1")
    requiere_accion: bool = Field(description="Si requiere acción del usuario")

model = OpenAIChatModel(
    "gpt-5",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model,
    output_type=BusinessInsight,
    retries=2,          # Para errores generales
    output_retries=3,   # Para validación del output
    instructions=(
        "Eres DataPulse AI, asistente de análisis empresarial para PYMEs. "
        "Proporcionas insights claros, breves y accionables sobre datos de negocio. "
        "Tu lenguaje es profesional pero accesible."
    )
)

@agent.result_validator
async def validate_business_insight_quality(
    ctx: RunContext[None],
    result: BusinessInsight
) -> BusinessInsight:
    """
    Valida que el insight cumpla estándares de calidad empresarial.
    
    Este validator implementa reflection: actúa como un editor de
    contenido que revisa el trabajo del agente y lo mejora iterativamente.
    """
    issues = []
    
    # Validación 1: Evitar lenguaje genérico tipo brochure
    generic_words = ["bien", "normal", "estable", "interesante"]
    if any(word in result.respuesta.lower() for word in generic_words):
        issues.append(
            "La respuesta contiene términos genéricos. "
            "Usa datos específicos y lenguaje ejecutivo."
        )
    
    # Validación 2: Métricas deben ser específicas y medibles
    for metric in result.metricas_clave:
        # Una métrica debe tener al menos 3 palabras (ej: "Crecimiento Q3: +15%")
        if len(metric.split()) < 3:
            issues.append(
                f"La métrica '{metric}' es demasiado vaga. "
                "Usa formato: 'Nombre métrica: valor unidad' (ej: 'Crecimiento ventas: +15%')."
            )
        
        # Debe contener algún número o porcentaje
        if not any(char.isdigit() for char in metric):
            issues.append(
                f"La métrica '{metric}' no incluye valores numéricos. "
                "Incluye el dato concreto."
            )
    
    # Validación 3: Coherencia entre confianza y acción requerida
    if result.nivel_confianza < 0.6 and result.requiere_accion:
        issues.append(
            f"Confianza baja ({result.nivel_confianza:.0%}) pero marcado como 'requiere acción'. "
            "Si la confianza es <60%, no debería requerir acción inmediata. "
            "Marca como no requiere acción o incrementa la confianza justificadamente."
        )
    
    # Si hay problemas, forzar reescritura con feedback consolidado
    if issues:
        feedback = "\n".join(f"• {issue}" for issue in issues)
        raise ModelRetry(
            f"El insight necesita mejoras:\n{feedback}\n\n"
            "Regenera el insight corrigiendo estos aspectos específicos."
        )
    
    return result

# Función wrapper con manejo de errores robusto
def get_business_insight(query: str) -> dict:
    """
    Obtiene insight con manejo completo de errores.
    
    Returns:
        Dict estructurado que siempre tiene 'status' y contenido relevante.
    """
    try:
        result = agent.run_sync(query)
        insight = result.output
        
        return {
            "status": "success",
            "insight": insight.model_dump(),
            "attempts": len(result.all_messages()),
            "tokens_used": result.usage().total_tokens if result.usage() else 0
        }
        
    except ModelRetry as e:
        return {
            "status": "error",
            "error_type": "max_retries",
            "message": "No se pudo generar un insight que cumpla los estándares de calidad.",
            "suggestion": "Reformula la pregunta con más contexto o datos específicos."
        }
    
    except Exception as e:
        return {
            "status": "system_error",
            "message": "Error del sistema. Contacta con soporte.",
            "support_code": "ERR-DP-001"
        }

# Pruebas del DataPulse AI v2
test_queries = [
    "Resúmeme las ventas de este trimestre",
    "¿Cómo van las ventas de este mes comparadas con el mes pasado?",
    "¿Qué tendencias esperas para el próximo trimestre?"
]

print("=" * 70)
print("DATAPULSE AI v2 - AGENTE AVANZADO")
print("=" * 70)

for query in test_queries:
    print(f"\n{'─' * 70}")
    print(f"📊 Consulta: {query}")
    print('─' * 70)
    
    response = get_business_insight(query)
    
    if response["status"] == "success":
        insight = response["insight"]
        
        print(f"\n✓ Intención: {insight['intencion'].upper()}")
        print(f"✓ Confianza: {insight['nivel_confianza']:.0%}")
        print(f"✓ Requiere acción: {'Sí' if insight['requiere_accion'] else 'No'}")
        
        print(f"\n📝 Respuesta:\n{insight['respuesta']}")
        
        print(f"\n📈 Métricas clave:")
        for metric in insight['metricas_clave']:
            print(f"  • {metric}")
        
        print(f"\n🔧 Intentos: {response['attempts']} | Tokens: {response['tokens_used']}")
    
    else:
        print(f"\n❌ Error: {response['message']}")
        if "suggestion" in response:
            print(f"💡 Sugerencia: {response['suggestion']}")

print(f"\n{'=' * 70}\n")
```

**Ejecuta la versión avanzada:**
```bash
uv run python 01-agentes-avanzados/datapulse_v2.py
```

**¿Qué mejoras notarás?**

1. **Autocorrección automática**: Si el agente genera métricas vagas tipo "Ventas buenas", el validator lo detecta y fuerza reescritura

2. **Métricas estructuradas**: Ya no acepta texto libre. Cada métrica debe tener formato específico con valores numéricos

3. **Coherencia lógica**: Verifica que la confianza y la necesidad de acción sean coherentes entre sí

4. **Manejo de errores profesional**: Nunca verás un stack trace, solo mensajes estructurados con códigos de soporte

5. **Observabilidad**: Puedes ver cuántos intentos usó y cuántos tokens consumió

**Comparación de outputs:**

```
Módulo 0:
{
  "respuesta": "Las ventas están bien este trimestre."
}

Módulo 1 (tras reflection):
{
  "respuesta": "Q3 registró 145k EUR en ventas (+12% vs Q2). El segmento Premium lideró con 35% del total.",
  "metricas_clave": [
    "Ventas Q3: 145,000 EUR",
    "Crecimiento interanual: +12%",
    "Participación Premium: 35%"
  ],
  "nivel_confianza": 0.85,
  "requiere_accion": false
}
```

---

## 10. Próximos pasos

**¡Felicidades por completar el Módulo 1!** Ahora sabes construir agentes robustos con validación, autocorrección, manejo de errores y dependencias.

**Habilidades adquiridas:**
- ✅ Tools complejas con validación Pydantic de parámetros
- ✅ Result validators que garantizan calidad del output
- ✅ Reflection patterns para mejora iterativa automática
- ✅ Configuración inteligente de retries por criticidad
- ✅ Orquestación de múltiples tools
- ✅ Dependency injection para integración con sistemas externos
- ✅ Manejo de errores profesional con fallbacks
- ✅ Instrucciones dinámicas basadas en contexto

**En el Módulo 2 aprenderás:**
- **Gestión de contexto conversacional**: Mantener estado entre múltiples mensajes
- **Streaming avanzado**: Respuestas progresivas para mejor UX
- **Integración con datos reales**: Conectar con CSV, bases de datos y APIs
- **Sesiones y persistencia**: Guardar y recuperar conversaciones

**En el Módulo 3 construirás:**
- **Agentes jerárquicos**: Agentes que delegan subtareas a otros agentes
- **Workflows complejos**: Encadenamiento de múltiples agentes
- **Integración con herramientas externas**: Búsqueda web, generación de gráficos
- **Despliegue en producción**: Docker, APIs REST, Gradio

Continúa con el [Módulo 2: Contexto y Validación](../02-contexto-validacion/README.md) cuando estés listo.

---

## Referencias

Documentación oficial que complementa este módulo:

- [PydanticAI Agents](https://ai.pydantic.dev/agents/) - Guía completa de agentes
- [PydanticAI Tools](https://ai.pydantic.dev/tools/) - Documentación de tools y parámetros
- [PydanticAI Output Validation](https://ai.pydantic.dev/output/) - Validación de salidas
- [PydanticAI Retries](https://ai.pydantic.dev/retries/) - Configuración de reintentos
- [Pydantic Validators](https://docs.pydantic.dev/latest/concepts/validators/) - Validators custom

---

**© 2025 Alfons Freixes | PydanticAI Course**