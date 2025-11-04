# 🧭 Caso de uso empresarial: **DataPulse AI – Asistente de análisis de datos para PYMEs**

> **Mini-proyecto transversal del curso PydanticAI**
> A lo largo del curso, construirás paso a paso un agente inteligente que analiza datos reales de negocio y genera insights ejecutivos.

---

## 🧩 Contexto

En 2025, las pequeñas y medianas empresas están adoptando **IA generativa aplicada a datos** para mejorar la toma de decisiones.
Sin embargo, la mayoría **no dispone de personal técnico** capaz de programar consultas o interpretar dashboards complejos.

**DataPulse AI** nace como una iniciativa interna de una consultora de datos.
Su objetivo: desarrollar un **asistente conversacional** que permita a los responsables de negocio **hacer preguntas en lenguaje natural sobre sus datos de ventas, clientes y productos** y obtener **respuestas con métricas calculadas**, lenguaje ejecutivo y recomendaciones accionables.

---

## 🎯 Objetivo general del proyecto

Construir un **agente de análisis empresarial** que:

1. Reciba consultas en lenguaje natural sobre datos (por ejemplo: "¿qué producto tuvo más crecimiento este trimestre?").
2. Interprete la intención y clasifique el tipo de consulta.
3. **Calcule métricas reales** a partir de datos estructurados.
4. Devuelva una respuesta profesional y validada en formato estructurado (JSON con intención, cálculos y recomendación).

El asistente deberá **evolucionar módulo a módulo** incorporando:

* Validación y tipado + datos mock inline (Módulo 0)
* Tools analíticas avanzadas (Módulo 1)
* Lectura de CSV y validación de fuentes (Módulo 2)
* Agentes encadenados o jerárquicos (Módulo 3)
* Conexión a APIs reales y persistencia (Módulo 4)
* Despliegue con Gradio / Docker (Módulo final)

---

## 🪜 Etapa 1 – Módulo 0 · Introducción

### 📘 Enunciado del ejercicio

Como punto de partida, queremos crear un **prototipo funcional** del agente DataPulse AI que:

1. **Reciba un mensaje de usuario** (consulta informal sobre datos de negocio).
2. **Clasifique la intención** del mensaje en una de tres categorías:
   * `"resumen"` – el usuario pide un resumen o interpretación general.
   * `"comparativa"` – el usuario compara períodos, productos o regiones.
   * `"forecast"` – el usuario pregunta por proyecciones o tendencias futuras.
3. **Calcule métricas reales** a partir de datos mock incluidos en el código.
4. **Genere una respuesta ejecutiva** (2-3 frases) con:
   - Números concretos del cálculo
   - Insight principal
   - Recomendación accionable

### 📊 Datos mock del negocio

El agente trabajará con estos datos ficticios de una PYME (inline en el código):

```python
DATOS_NEGOCIO = {
    "ventas_mensuales": {
        "q2_2025": [42000, 43000, 44000],  # abril, mayo, junio
        "q3_2025": [45000, 48000, 52000],  # julio, agosto, septiembre
    },
    "productos": {
        "Premium": {"q2": 18000, "q3": 24000},
        "Standard": {"q2": 20000, "q3": 21000},
        "Básico": {"q2": 6000, "q3": 7000},
    }
}
```

### 💡 Ejemplo de uso

**Input:**
```python
"¿Cómo van las ventas del último trimestre comparadas con el anterior?"
```

**Output esperado:**
```json
{
  "intencion": "comparativa",
  "datos_calculados": {
    "q2_total": 129000,
    "q3_total": 145000,
    "crecimiento_pct": 12.4,
    "top_producto": "Premium"
  },
  "respuesta": "Las ventas Q3 crecieron un 12.4% respecto a Q2 (145k€ vs 129k€), impulsadas por la línea Premium (+33%). Se recomienda ampliar inventario Premium y analizar punto de equilibrio para Standard."
}
```

### 🧠 Conceptos que se ponen en práctica

* Definición de agente con `Model(..., provider=Provider(...))`
* Validación estructurada de salida (`BaseModel`)
* **Paso de contexto** al agente mediante instrucciones con datos
* Instrucciones claras para cálculo y tono profesional
* Ejecución con `uv run python caso_uso_01.py`

---

## 🧑‍💻 Tu tarea

Crea un script llamado `00-introduccion/caso_uso_01.py` que:

### 1. Defina los datos mock

```python
DATOS_NEGOCIO = {
    "ventas_mensuales": {
        "q2_2025": [42000, 43000, 44000],
        "q3_2025": [45000, 48000, 52000],
    },
    "productos": {
        "Premium": {"q2": 18000, "q3": 24000},
        "Standard": {"q2": 20000, "q3": 21000},
        "Básico": {"q2": 6000, "q3": 7000},
    }
}
```

### 2. Defina el modelo de salida

```python
from pydantic import BaseModel, Field
from typing import Literal

class BusinessResponse(BaseModel):
    intencion: Literal["resumen", "comparativa", "forecast"]
    datos_calculados: dict = Field(
        description="Métricas calculadas (totales, porcentajes, top items)"
    )
    respuesta: str = Field(
        max_length=300,
        description="Respuesta ejecutiva con números, insight y recomendación"
    )
```

### 3. Configure el agente con instrucciones enriquecidas

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import settings

model = OpenAIChatModel(
    "gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

# Construir contexto con datos
contexto_datos = f"""
Datos disponibles del negocio:

Ventas trimestrales:
- Q2 2025: {DATOS_NEGOCIO['ventas_mensuales']['q2_2025']} (total: {sum(DATOS_NEGOCIO['ventas_mensuales']['q2_2025'])}€)
- Q3 2025: {DATOS_NEGOCIO['ventas_mensuales']['q3_2025']} (total: {sum(DATOS_NEGOCIO['ventas_mensuales']['q3_2025'])}€)

Productos (ventas por trimestre):
- Premium: Q2={DATOS_NEGOCIO['productos']['Premium']['q2']}€, Q3={DATOS_NEGOCIO['productos']['Premium']['q3']}€
- Standard: Q2={DATOS_NEGOCIO['productos']['Standard']['q2']}€, Q3={DATOS_NEGOCIO['productos']['Standard']['q3']}€
- Básico: Q2={DATOS_NEGOCIO['productos']['Básico']['q2']}€, Q3={DATOS_NEGOCIO['productos']['Básico']['q3']}€
"""

agent = Agent(
    model,
    output_type=BusinessResponse,
    instructions=f"""
Eres un asistente ejecutivo de análisis de datos empresariales.

{contexto_datos}

Instrucciones:
1. Clasifica la intención de la pregunta (resumen/comparativa/forecast)
2. Calcula métricas relevantes usando los datos disponibles:
   - Totales, porcentajes de crecimiento/caída
   - Identifica productos top
   - Compara períodos cuando sea relevante
3. Genera una respuesta profesional que:
   - Incluya números concretos del cálculo
   - Destaque el insight principal
   - Ofrezca una recomendación accionable
   - Máximo 2-3 frases, lenguaje ejecutivo
4. Incluye todos los cálculos en 'datos_calculados'
"""
)
```

### 4. Ejecute pruebas con tres consultas

```python
consultas = [
    "Resúmeme las ventas del último trimestre.",
    "¿Qué producto creció más este trimestre comparado con el anterior?",
    "¿Qué esperas para el próximo mes basándote en la tendencia?"
]

for consulta in consultas:
    print(f"\n{'='*60}")
    print(f"Consulta: {consulta}")
    print('='*60)
    
    result = agent.run_sync(consulta)
    
    print(f"\nIntención: {result.output.intencion}")
    print(f"\nDatos calculados:")
    for k, v in result.output.datos_calculados.items():
        print(f"  {k}: {v}")
    print(f"\nRespuesta:")
    print(f"  {result.output.respuesta}")
```

---

## 📈 Qué evaluaremos en esta etapa

| Aspecto | Descripción | Peso |
|---------|-------------|------|
| 💬 Claridad del prompt | Las instrucciones incluyen datos contextuales y son precisas | 20% |
| 🧱 Uso correcto del modelo y provider | Se usa el patrón `Model(..., provider=Provider(...))` | 15% |
| ✅ Validación del output | La respuesta cumple el modelo `BusinessResponse` con todos los campos | 20% |
| 🔢 Cálculos reales | Los `datos_calculados` contienen métricas correctas (no inventadas) | 25% |
| 🧠 Relevancia empresarial | La respuesta suena realista, profesional y accionable | 15% |
| 🧩 Legibilidad del código | Código limpio, con comentarios útiles, fácil de mantener | 5% |

---

## ✅ Criterios de aceptación

**El ejercicio está completo cuando:**

1. ✅ El agente clasifica correctamente la intención (resumen/comparativa/forecast)
2. ✅ Los `datos_calculados` contienen métricas reales calculadas de los datos mock
3. ✅ La respuesta incluye números concretos del cálculo (no genéricos)
4. ✅ El crecimiento de Premium es correcto: `(24000-18000)/18000 = 33.3%`
5. ✅ El crecimiento Q3 vs Q2 es correcto: `(145000-129000)/129000 = 12.4%`
6. ✅ La respuesta tiene tono ejecutivo (2-3 frases máximo)
7. ✅ Incluye una recomendación accionable

---

## 🎓 Ejemplo completo de salida esperada

### Consulta 1: "Resúmeme las ventas del último trimestre"

```json
{
  "intencion": "resumen",
  "datos_calculados": {
    "q3_total": 145000,
    "promedio_mensual": 48333,
    "mes_pico": "septiembre",
    "valor_pico": 52000
  },
  "respuesta": "Q3 cerró con 145k€ en ventas, con pico en septiembre (52k€). Tendencia alcista sostenida (+7% mensual promedio). Recomendación: replicar acciones de septiembre en Q4."
}
```

### Consulta 2: "¿Qué producto creció más?"

```json
{
  "intencion": "comparativa",
  "datos_calculados": {
    "premium_crecimiento_pct": 33.3,
    "standard_crecimiento_pct": 5.0,
    "basico_crecimiento_pct": 16.7,
    "top_producto": "Premium"
  },
  "respuesta": "Premium lidera con +33% de crecimiento Q3 vs Q2 (24k€ vs 18k€), seguido de Básico (+17%). Standard se estancó (+5%). Priorizar marketing en Premium y revisar estrategia de Standard."
}
```

### Consulta 3: "¿Qué esperas para el próximo mes?"

```json
{
  "intencion": "forecast",
  "datos_calculados": {
    "tendencia_q3": [45000, 48000, 52000],
    "crecimiento_promedio_mensual_pct": 7.5,
    "proyeccion_octubre": 56000
  },
  "respuesta": "Basado en tendencia Q3 (+7.5% mensual), octubre proyecta ~56k€. Si la estacionalidad se mantiene, Q4 podría superar 170k€. Validar con stock y capacidad operativa."
}
```

---

## 🚀 Próximos pasos

### **Módulo 1**
Evolucionarás este agente para:
- Implementar **tools** que calculen KPIs automáticamente
- Añadir validación estricta de rangos de fechas
- Detectar anomalías en los datos (outliers, valores negativos)

### **Módulo 2**
- Reemplazar datos mock por **lectura de CSV**
- Validar estructura de archivos con Pydantic
- Implementar caché de datos procesados

### **Módulo 3**
- Crear agentes especializados (ventas, productos, forecast)
- Implementar agente coordinador que delega a especialistas
- Añadir reflection para validar calidad de respuestas

### **Módulo 4**
- Conectar con APIs reales (Google Sheets, CRM, ERP)
- Implementar autenticación y manejo de errores robusto
- Persistencia de resultados en base de datos

### **Módulo Final**
- Desplegar con Gradio para interfaz web
- Dockerizar la aplicación
- Añadir observabilidad con Logfire

---

## 💡 Tips para el éxito

1. **Primero resuelve en Python puro**: Calcula las métricas manualmente antes de pedírselo al LLM
2. **Sé específico en las instrucciones**: "Calcula crecimiento con fórmula (nuevo-viejo)/viejo*100"
3. **Valida las respuestas**: Revisa que los números sean coherentes
4. **Itera el prompt**: Si el agente inventa datos, hazlo más explícito sobre usar solo los datos proporcionados

---

**¿Listo para empezar?** Crea tu archivo `caso_uso_01.py` y valida que puedes calcular métricas reales. ¡El resto del curso construye sobre esta base!