# Ejemplo 02: Normalizador de Envíos (Logística)

En e-commerce y logística, la calidad de los datos de dirección es crítica. Los usuarios cometen errores de formato, omiten códigos postales o mezclan instrucciones de entrega con la dirección física.

Este ejemplo muestra cómo crear un **pipeline de normalización robusto** que no solo valida, sino que limpia y estandariza los datos de entrada, procesando múltiples solicitudes en paralelo para maximizar el rendimiento.

## 📝 Código

Crea el archivo `00-introduccion/shipping_normalizer.py`:

```python
import asyncio
from typing import Annotated

from devtools import debug
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import settings


# 1. MODELO DE ENVÍO
class ShippingAddress(BaseModel):
    recipient_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Nombre y apellidos del destinatario. None si no se especifica.",
        ),
    ]
    street: str
    number: Annotated[str, Field(default="S/N", description="Número de calle")]
    city: str
    
    zip_code: Annotated[
        str, Field(pattern=r"^\d{5}$", description="Código postal de 5 dígitos (España)")
    ]
    
    province: str
    country: str = "España"
    
    notes: Annotated[
        str | None,
        Field(default=None, description="Instrucciones de entrega (interfono, piso, etc.)"),
    ]

    # Patrón de limpieza previo a la validación
    @field_validator("zip_code", mode="before")
    @classmethod
    def normalize_zip(cls, v: str) -> str:
        """Normaliza CP: elimina espacios y rellena con ceros si es necesario."""
        v = str(v).strip().replace(" ", "")
        return v.zfill(5)  # Ejemplo: transforma "8036" -> "08036"


# 2. CONFIGURACIÓN
model = OpenAIChatModel(
    "gpt-4o-mini",  # Tarea sencilla, priorizamos velocidad/bajo coste
    provider=OpenAIProvider(api_key=settings.openai_api_key),
)

agent = Agent(
    model,
    output_type=ShippingAddress,
    system_prompt=(
        "Eres un experto en logística y envíos en España. "
        "Normaliza la dirección proporcionada al formato estándar. "
        "Corrige errores ortográficos evidentes en nombres de ciudades. "
        "Separa las instrucciones de entrega (piso, puerta, 'dejar al portero') en el campo 'notes'. "
        "Si no se menciona el nombre del destinatario, deja recipient_name como null. "
        "NO inventes datos que no estén en el texto original."
    ),
)


# 3. FUNCIONES AUXILIARES
async def normalize_address(raw: str) -> ShippingAddress | None:
    """Normaliza una dirección individual manejando errores."""
    try:
        result = await agent.run(raw)
        return result.data
    except Exception as e:
        print(f"❌ Error procesando '{raw[:30]}...': {e}")
        return None


# 4. EJECUCIÓN (Batch Processing)
async def main():
    raw_inputs = [
        "Sr. Federico González, c/ mallorca 123, bcn, codigo 08036, dejar a la portera Maria",
        "Paseo de la Castellana num 50, planta 4 oficina B, Madrid 28046",
    ]

    print("🚚 Iniciando normalización de envíos (Paralelo)...\n")

    # Patrón Enterprise: Procesamiento paralelo con asyncio.gather
    # Esto reduce el tiempo total de ejecución drásticamente en lotes grandes.
    results = await asyncio.gather(*[normalize_address(raw) for raw in raw_inputs])

    for raw, address in zip(raw_inputs, results, strict=True):
        print(f"Input: '{raw}'")
        if address:
            debug(address)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())

```

🔍 Puntos Clave a Observar
1. Validadores de Limpieza (mode='before'): El validador normalize_zip se ejecuta antes de que Pydantic compruebe el regex. Esto nos permite sanear datos (ej: convertir el entero 8036 o el string " 8036 " al formato correcto "08036") antes de que falle la validación estricta. Es un patrón esencial para robustez.

2. Procesamiento Paralelo (asyncio.gather): En lugar de esperar a que termine una dirección para empezar la siguiente (secuencial), lanzamos todas las peticiones al LLM simultáneamente. En una API real, esto significa que procesar 10 direcciones tarda casi lo mismo que procesar 1.

3. Manejo de Optionalidad: El uso de str | None junto con default=None en recipient_name instruye al sistema para que sea honesto: si no hay nombre, devuelve None en lugar de inventar o poner "Desconocido".