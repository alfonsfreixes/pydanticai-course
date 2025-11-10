# DataPulse AI - Solución Propuesta (Módulo 0)

> **Solución de referencia** para el caso de uso empresarial DataPulse AI.  
> Archivo: `00-introduccion/caso_uso_01.py`

---

## 💻 Código completo de la solución

```python
"""
Módulo 0 - Caso de uso 01: DataPulse AI - Prototipo básico
Agente de análisis empresarial con clasificación de intención y respuesta ejecutiva.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
from typing import Literal, Any, Dict, List
from config import settings

# ============================================================================
# DATOS MOCK DE NEGOCIO
# ============================================================================

DATOS_NEGOCIO: Dict[str, Dict[str, Any]] = {
    "ventas_mensuales": {
        "q2_2025": [42000, 43000, 44000],
        "q3_2025": [45000, 48000, 52000]
    },
    "productos": {
        "premium": {"q2": 18000, "q3": 24000},
        "estandar": {"q2": 20000, "q3": 21000},
        "básico": {"q2": 6000, "q3": 7000}
    }
}

# ============================================================================
# MODELO DE SALIDA ESTRUCTURADA
# ============================================================================

class BusinessResponse(BaseModel):
    """Respuesta estructurada del agente de análisis empresarial."""
    
    intencion: Literal["resumen", "comparativa", "forecast"] = Field(
        description="Tipo de consulta identificada"
    )
    datos_calculados: dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas calculadas relevantes para la respuesta"
    )
    respuesta: str = Field(
        max_length=280,
        description="Respuesta ejecutiva concisa con cifras, insight y recomendación"
    )

# ============================================================================
# CONSTRUCCIÓN DINÁMICA DEL CONTEXTO
# ============================================================================

def construir_contexto_datos() -> str:
    """
    Construye dinámicamente el contexto de datos para las instrucciones.
    Ventajas: single source of truth, mantenibilidad, escalabilidad.
    """
    ventas_q2 = DATOS_NEGOCIO["ventas_mensuales"]["q2_2025"]
    ventas_q3 = DATOS_NEGOCIO["ventas_mensuales"]["q3_2025"]
    
    # Formatear ventas en miles (k)
    ventas_q2_str = ", ".join([f"{v//1000}k" for v in ventas_q2])
    ventas_q3_str = ", ".join([f"{v//1000}k" for v in ventas_q3])
    
    # Construir líneas de productos
    productos_lineas: List[str] = []
    for nombre, datos in DATOS_NEGOCIO["productos"].items():
        q2_k = datos["q2"] // 1000
        q3_k = datos["q3"] // 1000
        productos_lineas.append(f"{nombre.capitalize()} ({q2_k}k→{q3_k}k)")
    
    productos_str = ", ".join(productos_lineas)
    
    return f"""DATOS DISPONIBLES:
- Ventas Q2 2025: [{ventas_q2_str}] mensuales
- Ventas Q3 2025: [{ventas_q3_str}] mensuales  
- Productos Q2→Q3: {productos_str}"""

# ============================================================================
# CONFIGURACIÓN DEL AGENTE
# ============================================================================

model = OpenAIChatModel(
    model_name="gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

# Construir instrucciones con datos dinámicos
INSTRUCCIONES = f"""Eres un asistente ejecutivo de análisis empresarial.

{construir_contexto_datos()}

INSTRUCCIONES:
1. Clasifica la intención: resumen / comparativa / forecast
2. Calcula las métricas relevantes (totales, crecimientos, rankings)
3. Genera respuesta ejecutiva en 2-3 frases que incluya:
   ✓ Cifra principal calculada
   ✓ Insight clave del análisis
   ✓ Recomendación accionable
4. CRÍTICO: Máximo 280 caracteres (como un tweet extendido)
5. Usa lenguaje ejecutivo, números concretos, sin jerga técnica

EJEMPLO VÁLIDO:
"Q3 facturó 145k€ (+12% vs Q2). Premium lideró crecimiento con +33%. Recomendación: intensificar marketing en línea premium."
"""

agent = Agent(
    model=model,
    output_type=BusinessResponse,
    retries=2,
    instructions=INSTRUCCIONES
)

# ============================================================================
# EJECUCIÓN Y PRUEBAS
# ============================================================================

def ejecutar_consultas():
    """Ejecuta las tres consultas de ejemplo del caso de uso."""
    
    consultas = [
        "Resúmeme las ventas del último trimestre",
        "¿Qué producto creció más este trimestre comparado con el anterior?",
        "¿Qué esperas para el próximo mes basándote en las tendencias actuales?"
    ]
    
    print("\n" + "="*70)
    print("DATAPULSE AI - Asistente de Análisis Empresarial")
    print("="*70)
    print(f"\n📌 Datos de negocio cargados:")
    print(f"   • Q2 2025: {sum(DATOS_NEGOCIO['ventas_mensuales']['q2_2025']):,}€")
    print(f"   • Q3 2025: {sum(DATOS_NEGOCIO['ventas_mensuales']['q3_2025']):,}€")
    print(f"   • Productos: {len(DATOS_NEGOCIO['productos'])} líneas")
    
    for i, consulta in enumerate(consultas, 1):
        print(f"\n[CONSULTA {i}]")
        print(f"📋 {consulta}")
        print("-" * 70)
        
        try:
            result = agent.run_sync(consulta)
            output = result.output
            
            # Mostrar resultados estructurados
            print(f"🎯 Intención detectada: {output.intencion}")
            
            if output.datos_calculados:
                print(f"\n📊 Datos calculados:")
                for key, value in output.datos_calculados.items():
                    print(f"   • {key}: {value}")
            
            print(f"\n💼 Respuesta ejecutiva ({len(output.respuesta)} caracteres):")
            print(f"   {output.respuesta}")
            
        except Exception as e:
            print(f"❌ Error al procesar consulta: {e}")
            print(f"   Tip: El modelo puede estar generando respuestas demasiado largas.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    ejecutar_consultas()
```

---

## Resumen breve

DataPulse AI se apoya en `DATOS_NEGOCIO` para alimentar el prompt mediante `construir_contexto_datos()`, asegurando que cualquier cambio en los mocks actualice inmediatamente las instrucciones. El esquema `BusinessResponse` (con `Literal` y `max_length=280`) valida intención y longitud, mientras `retries=2` obliga al modelo a corregirse cuando viola el contrato. Toda la ejecución sucede con `OpenAIChatModel("gpt-5-mini")` y credenciales gestionadas por `settings`, por lo que basta correr `uv run python 00-introduccion/caso_uso_01.py` para reproducir la solución.

### Detalles relevantes

- La estructura de datos incluye ventas mensuales y desglose por producto, lo que permite al LLM derivar totales, crecimientos y rankings sin lógica adicional.
- El prompt combina rol, datos dinámicos y un ejemplo concreto, lo que guía el tono ejecutivo y mantiene las respuestas dentro de 280 caracteres.
- `datos_calculados` queda disponible para trazabilidad futura, aunque en esta ejecución el agente volcó los cálculos directamente en la respuesta ejecutiva.

## Respuesta real observada

```
======================================================================
DATAPULSE AI - Asistente de Análisis Empresarial
======================================================================

📌 Datos de negocio cargados:
   • Q2 2025: 129,000€
   • Q3 2025: 145,000€
   • Productos: 3 líneas

[CONSULTA 1]
📋 Resúmeme las ventas del último trimestre
----------------------------------------------------------------------
🎯 Intención detectada: resumen

💼 Respuesta ejecutiva (210 caracteres):
   Q3 facturó 145k€ (+12% vs Q2 129k€). Premium lideró con 24k€ (+33% vs Q2), 
   Estándar 21k€, Básico 7k€; recomendación: priorizar inversión y campañas en 
   Premium para consolidar el impulso y aumentar oferta/stock.

[CONSULTA 2]
📋 ¿Qué producto creció más este trimestre comparado con el anterior?
----------------------------------------------------------------------
🎯 Intención detectada: comparativa

💼 Respuesta ejecutiva (223 caracteres):
   Premium creció +6k (+33.3%) de 18k a 24k entre Q2 y Q3, liderando crecimiento. 
   Estándar subió +1k (+5%) y Básico +1k (+16.7%). Recomendación: priorizar 
   inversión en marketing y stock para Premium para consolidar el impulso.

[CONSULTA 3]
📋 ¿Qué esperas para el próximo mes basándote en las tendencias actuales?
----------------------------------------------------------------------
🎯 Intención detectada: forecast

💼 Respuesta ejecutiva (240 caracteres):
   Intención: forecast. Espero ~57k el próximo mes (+9.6% vs 52k, último mes). 
   Insight: Premium impulsa crecimiento (+33% Q2→Q3, 24k del último mes) mientras 
   Estándar y Básico crecen poco. Recomendación: priorizar inversión y stock en Premium.

======================================================================
```
