# Ejemplo 01: Extractor de Entidades Legales (NER Estructurado)

En este ejemplo simularemos un caso de uso **LegalTech**. Transformaremos un texto legal no estructurado (un contrato o notificación) en una estructura de datos jerárquica y validada.

Este ejemplo demuestra un patrón avanzado: **Composición de Modelos**. En lugar de tener una lista plana de campos, utilizamos sub-modelos (`MonetaryAmount`, `Party`) para crear una representación rica de la información.

## 📝 Código

Crea el archivo `00-introduccion/legal_extractor.py`:

```python
import asyncio
from datetime import date
from typing import Annotated

from devtools import debug
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import settings


# 1. MODELADO DE DATOS (Composición y Annotated)

class Party(BaseModel):
    name: Annotated[str, Field(description="Nombre completo de la entidad o persona")]
    role: Annotated[
        str,  # Usamos str con descripción para permitir roles imprevistos (flexibilidad)
        Field(description="Rol en el contrato: provider, client, guarantor, witness, etc."),
    ]


class MonetaryAmount(BaseModel):
    """Sub-modelo para capturar contexto financiero completo."""
    value: float
    currency: Annotated[str, Field(description="Código ISO: USD, EUR, etc.")]
    period: Annotated[str, Field(description="Período: monthly, annual, one-time, etc.")]


class ContractExtraction(BaseModel):
    contract_type: Annotated[
        str,
        Field(
            description="Tipo de contrato: NDA, SLA, Services, Consulting, Employment, Licensing, Other"
        ),
    ]

    parties: Annotated[
        list[Party], Field(min_length=2, description="Partes involucradas en el acuerdo")
    ]

    effective_date: Annotated[
        date | None, Field(description="Fecha de inicio de efectos. None si no se menciona.")
    ]

    # Composición: Usamos el modelo MonetaryAmount aquí
    monetary_value: Annotated[
        MonetaryAmount | None,
        Field(description="Valor del contrato si se menciona, con moneda y período."),
    ]

    risk_flags: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Cláusulas potencialmente peligrosas: limitaciones de responsabilidad, renuncias, penalizaciones excesivas, etc.",
        ),
    ]


# 2. CONFIGURACIÓN
# Usamos gpt-4o-mini por eficiencia, ya que la tarea de extracción está muy guiada por el schema
model = OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(api_key=settings.openai_api_key))

agent = Agent(
    model,
    output_type=ContractExtraction,
    system_prompt=(
        "Eres un asistente legal senior especializado en auditoría de contratos. "
        "Extrae la información clave del texto proporcionado con precisión. "
        "Si no encuentras una fecha explícita, déjala como null. "
        "Normaliza los valores monetarios a float (ej: 10k -> 10000.0). "
        "Identifica cláusulas de riesgo como: limitaciones de responsabilidad, "
        "renuncias de derechos, penalizaciones, cláusulas de indemnidad."
    ),
)


# 3. LÓGICA DE NEGOCIO (Python Puro)
RISK_KEYWORDS = [
    "negligencia",
    "renuncia",
    "indemnidad",
    "limitación",
    "penalización",
    "exclusión",
    "responsabilidad",
]

def check_compliance_risks(extraction: ContractExtraction) -> list[str]:
    """
    Identifica alertas de compliance basadas en risk_flags.
    Este es un patrón clave: La IA estructura los datos, Python aplica las reglas deterministas.
    """
    alerts = []
    for flag in extraction.risk_flags:
        flag_lower = flag.lower()
        if any(kw in flag_lower for kw in RISK_KEYWORDS):
            alerts.append(flag)
    return alerts


# 4. EJECUCIÓN
async def main():
    legal_text = """
    ACUERDO DE SERVICIOS
    Entre TechSolutions Inc. (el Proveedor) y Global Corp S.A. (el Cliente).

    A partir del 15 de marzo de 2024, el Proveedor se compromete a dar soporte 24/7.
    El coste será de 5.000 USD mensuales facturados a 30 días.

    Cláusula 9: El Cliente renuncia a cualquier reclamación por daños indirectos, 
    incluso si fueron causados por negligencia grave del Proveedor.
    """

    print("📄 Analizando documento legal...\n")

    try:
        result = await agent.run(legal_text)
        extraction = result.data  # Acceso tipado al modelo ContractExtraction

        print("✅ Extracción Completada:")
        debug(extraction)

        # Validación de compliance post-extracción
        compliance_alerts = check_compliance_risks(extraction)
        
        if compliance_alerts:
            print(f"\n⚠️ ALERTAS DE COMPLIANCE ({len(compliance_alerts)}):")
            for alert in compliance_alerts:
                print(f"  • {alert}")

    except Exception as e:
        print(f"❌ Error en la extracción: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

🔍 Puntos Clave a Observar
1. Modelos Anidados (MonetaryAmount): En lugar de intentar meter toda la información financiera en un solo string o float, creamos un sub-modelo. Esto permite que el LLM estructure la moneda (USD) y la periodicidad (monthly) por separado, facilitando cálculos futuros (como anualizar costes).

2. Flexibilidad vs. Rigidez: Hemos cambiado Literal por str con una descripción detallada en campos como role. Literal es excelente para categorías cerradas, pero en documentos legales los roles pueden ser impredecibles ("avalista", "testigo", "subsidiaria"). str + description permite al LLM inferir el rol correcto sin romperse por validación.

3. Patrón Híbrido (AI + Code): Observa la función check_compliance_risks. No le pedimos al LLM que decida si el contrato es "peligroso" (subjetivo). Le pedimos que extraiga cláusulas (risk_flags), y luego usamos código Python determinista para buscar palabras clave. Esto combina la flexibilidad semántica de la IA con la fiabilidad del código tradicional.