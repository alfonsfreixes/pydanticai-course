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

## 🎯 Decisiones clave de implementación

### 1. **Datos mock estructurados**

```python
DATOS_NEGOCIO: Dict[str, Dict[str, Any]] = {
    "ventas_mensuales": {
        "q2_2025": [42000, 43000, 44000],  # Datos mensuales
        "q3_2025": [45000, 48000, 52000]
    },
    "productos": {
        "premium": {"q2": 18000, "q3": 24000},
        "estandar": {"q2": 20000, "q3": 21000},
        "básico": {"q2": 6000, "q3": 7000}
    }
}
```

**Por qué esta estructura:**
- **Tipado fuerte**: Usamos `Dict[str, Dict[str, Any]]` para claridad y type hints
- **Datos granulares**: Ventas mensuales permiten análisis de tendencias
- **Comparabilidad**: Estructura Q2 vs Q3 facilita análisis comparativos
- **Escalabilidad**: Fácil añadir más trimestres, productos o métricas
- **Sin dependencias**: Funciona sin bases de datos o APIs externas

En producción, esta estructura se reemplazaría por queries a tu base de datos:
```python
# Producción
DATOS_NEGOCIO = obtener_datos_desde_db(trimestre="Q3", year=2025)
```

---

### 2. **Construcción dinámica del prompt (Single Source of Truth)**

```python
def construir_contexto_datos() -> str:
    """
    Construye dinámicamente el contexto de datos para las instrucciones.
    Ventajas: single source of truth, mantenibilidad, escalabilidad.
    """
    ventas_q2 = DATOS_NEGOCIO["ventas_mensuales"]["q2_2025"]
    ventas_q3 = DATOS_NEGOCIO["ventas_mensuales"]["q3_2025"]
    
    # Formatear ventas en miles (k) - más legible para el LLM
    ventas_q2_str = ", ".join([f"{v//1000}k" for v in ventas_q2])
    ventas_q3_str = ", ".join([f"{v//1000}k" for v in ventas_q3])
    
    # Construir líneas de productos con formato visual
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
```

**Esta es IMPORTANTE.** ¿Por qué?

**❌ Antipatrón (hardcodear datos en el prompt):**
```python
INSTRUCCIONES = """
Ventas Q2: 42k, 43k, 44k
Ventas Q3: 45k, 48k, 52k
...
"""
```
Si cambias los datos en `DATOS_NEGOCIO`, el prompt no se actualiza. Mantenimiento imposible.

**✅ Patrón correcto (construcción dinámica):**
```python
INSTRUCCIONES = f"""
{construir_contexto_datos()}
"""
```

**Beneficios:**
1. **Single source of truth**: Cambias datos en un lugar, todo se actualiza
2. **Mantenibilidad**: No hay duplicación de datos entre variables y prompts
3. **Preparado para producción**: Cuando conectes a una DB, solo cambias `DATOS_NEGOCIO`
4. **Testing**: Puedes mockear `DATOS_NEGOCIO` fácilmente en tests
5. **Formateo consistente**: El formato (k, →, etc.) se aplica automáticamente

---

### 3. **Modelo de salida con validación Pydantic**

```python
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
```

**Anatomía de cada campo:**

**`intencion: Literal["resumen", "comparativa", "forecast"]`**
- `Literal` restringe valores posibles (no puede ser "prediccion" o "analysis")
- El LLM **debe** elegir uno de estos tres valores exactos
- Si intenta devolver otro valor, Pydantic rechaza la respuesta automáticamente

**`datos_calculados: dict[str, Any]`**
- Opcional (por el `default_factory=dict`)
- Almacena métricas intermedias (totales, porcentajes, rankings)
- Útil para debugging y auditoría (ves QUÉ calculó el LLM)

**`respuesta: str = Field(max_length=280)`**
- **Límite**: 280 caracteres (como Twitter extendido)
- Sin este límite, el LLM genera respuestas de 500+ caracteres
- Combinado con `retries=2`, el agente reintenta si se pasa

**¿Por qué 280 caracteres?**
- Referencia familiar (Twitter)
- Fuerza concisión y claridad
- Ejecutivos no leen párrafos largos
- Facilita mostrar en dashboards/emails

---

### 4. **Configuración del modelo con patrón explícito**

```python
model = OpenAIChatModel(
    model_name="gpt-5-mini",
    provider=OpenAIProvider(api_key=settings.openai_api_key)
)

agent = Agent(
    model=model,
    output_type=BusinessResponse,
    retries=2,
    instructions=INSTRUCCIONES
)
```

**Decisiones de configuración:**

**`model_name="gpt-5-mini"`**
- Modelo eficiente y rápido
- Suficiente para análisis de datos estructurados
- Más económico que `gpt-5` completo

**`provider=OpenAIProvider(api_key=settings.openai_api_key)`**
- API key desde `config.py` (settings de Pydantic)
- No hardcodeada en el código
- Fácil cambiar a Anthropic si quieres: solo cambias el provider

**`output_type=BusinessResponse`**
- PydanticAI instruye automáticamente al LLM para devolver JSON con este schema
- Valida automáticamente la respuesta
- Si falla validación, el agente reintenta (hasta `retries` veces)

**`retries=2`**
- Permite 2 reintentos automáticos si:
  - La respuesta excede 280 caracteres
  - El campo `intencion` no es uno de los valores permitidos
  - Cualquier otra violación del schema
- Sin retries, muchas consultas fallarían en el primer intento

---

### 5. **Instrucciones con estructura y ejemplo concreto**

```python
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
```

**Estructura de las instrucciones:**

1. **Rol claro**: "Eres un asistente ejecutivo..." (establece el tono)
2. **Datos inyectados**: `{construir_contexto_datos()}` (información actualizada)
3. **Tareas numeradas**: Lista explícita de qué hacer
4. **Criterios de calidad**: Qué debe incluir cada respuesta
5. **Límite explícito**: "CRÍTICO: Máximo 280 caracteres"
6. **Ejemplo concreto**: Muestra el formato exacto esperado

**Por qué el ejemplo es crucial:**
Los LLMs aprenden mejor con ejemplos que con descripciones. El ejemplo muestra:
- Formato exacto ("Q3 facturó...")
- Longitud apropiada (bajo 280 chars)
- Estructura: cifra → insight → recomendación
- Tono ejecutivo (directo, con datos, accionable)

---

## 🚀 Ejecución

### Prerrequisitos

1. **Estructura del proyecto:**
   ```
   pydanticai-course/
   ├── .env                    # API keys (no subir a git)
   ├── config.py               # Configuración centralizada
   ├── 00-introduccion/
   │   └── caso_uso_01.py     # ← Este archivo
   └── ...
   ```

2. **Configurar `.env`** con tu API key:
   ```bash
   OPENAI_API_KEY="sk-..."
   ```

3. **Verificar que `config.py`** existe en la raíz:
   ```python
   from pydantic_settings import BaseSettings, SettingsConfigDict

   class Settings(BaseSettings):
       openai_api_key: str | None = None
       
       model_config = SettingsConfigDict(
           env_file=".env",
           case_sensitive=False,
       )

   settings = Settings()
   ```

### Ejecutar

```bash
# Desde la raíz del proyecto
uv run python 00-introduccion/caso_uso_01.py
```

---

## 📊 Salida real del programa

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

---

## 🔍 Análisis de los resultados

### ✅ Lo que funciona bien

**1. Clasificación de intención perfecta**
- Consulta 1: "Resúmeme..." → `resumen` ✓
- Consulta 2: "¿Qué producto creció más..." → `comparativa` ✓  
- Consulta 3: "¿Qué esperas..." → `forecast` ✓

El agente identifica correctamente la intención en los tres casos sin ambigüedad.

**2. Cumplimiento del límite de 280 caracteres**
- Consulta 1: 210 caracteres ✓
- Consulta 2: 223 caracteres ✓
- Consulta 3: 240 caracteres ✓

Todas las respuestas están bajo el límite. El constraint `max_length=280` funciona correctamente.

**3. Estructura respuesta ejecutiva**

Cada respuesta incluye los tres elementos solicitados:

- ✅ **Cifra principal**: "Q3 facturó 145k€", "Premium creció +6k", "Espero ~57k"
- ✅ **Insight clave**: "+12% vs Q2", "liderando crecimiento", "Premium impulsa crecimiento"
- ✅ **Recomendación**: "priorizar inversión en Premium" (aparece en las 3)

**4. Cálculos automáticos correctos**
- Crecimiento Q2→Q3: +12% (correcto: 129k → 145k)
- Crecimiento Premium: +33.3% (correcto: 18k → 24k)
- Proyección siguiente mes: ~57k basado en tendencia +9.6%

El LLM está **calculando métricas reales** a partir de los datos, no inventando números.

**5. Lenguaje ejecutivo y accionable**
- Usa cifras concretas (145k€, +33%, ~57k)
- Evita jerga técnica
- Termina con recomendaciones accionables ("priorizar inversión...")
- Tono directo y profesional

---

### 📝 Observaciones interesantes

**1. El agente identifica patrones sin que se le pida explícitamente**

En las tres consultas, el agente detecta que **Premium es el driver de crecimiento**:
- Consulta 1: "Premium lideró con 24k€ (+33%...)"
- Consulta 2: "Premium creció +6k (+33.3%)... liderando crecimiento"
- Consulta 3: "Premium impulsa crecimiento (+33% Q2→Q3...)"

Esto demuestra que el LLM está analizando los datos de forma **coherente entre consultas**.

**2. Coherencia en las recomendaciones**

Las tres respuestas convergen en la misma recomendación estratégica:
> "Priorizar inversión y campañas/stock en Premium"

Esto es exactamente lo que esperarías de un analista humano: identificar el producto estrella y recomendar doblar la apuesta.

**3. El campo `datos_calculados` está vacío**

En esta ejecución, el agente no rellenó el campo `datos_calculados`. Esto es porque:
- No lo pedimos explícitamente en las instrucciones
- El campo es opcional (`default_factory=dict`)
- El agente priorizó incluir los cálculos directamente en la respuesta

**¿Cómo hacer que lo use?**

Añadir a las instrucciones:
```python
2. Calcula las métricas relevantes y guárdalas en datos_calculados:
   - Para resumen: total_q3, total_q2, crecimiento_pct
   - Para comparativa: crecimiento_pct por producto
   - Para forecast: tendencia_mensual_pct, proyeccion_siguiente_mes
```

---

### 🎯 Validación de criterios de evaluación

| Criterio | Objetivo | Resultado | ✓/✗ |
|----------|----------|-----------|-----|
| **Claridad del prompt** | Instrucciones precisas + ejemplo | 5 secciones estructuradas + ejemplo de 280 chars | ✅ |
| **Patrón Model + Provider** | Configuración explícita | `OpenAIChatModel` + `OpenAIProvider(api_key=settings.openai_api_key)` | ✅ |
| **Validación del output** | Schema Pydantic completo | `BusinessResponse` con `Literal`, `max_length=280`, `default_factory` | ✅ |
| **Datos calculados** | Métricas relevantes | Cálculos correctos (+12%, +33.3%, ~57k) | ✅ |
| **Relevancia empresarial** | Lenguaje ejecutivo accionable | Cifras concretas, insights claros, recomendaciones prioritarias | ✅ |
| **Legibilidad del código** | Comentarios, estructura | 4 secciones comentadas, funciones con docstrings, type hints | ✅ |

**Puntuación: 6/6 criterios cumplidos** ✅

---

## 🔧 Troubleshooting

### Error: `No API key found`
**Causa:** El archivo `.env` no existe, está mal ubicado o tiene formato incorrecto.

**Solución:**
1. Verifica que `.env` está en la raíz del proyecto (mismo nivel que `config.py`)
2. Contenido de `.env`:
   ```bash
   OPENAI_API_KEY="sk-..."
   ```
3. Ejecuta siempre con `uv run python` (no `python` directamente)

---

### Error: `ValidationError: respuesta must have at most 280 characters`
**Causa:** El LLM generó una respuesta de más de 280 caracteres en el primer intento.

**¿Por qué pasa?**
- Los LLMs no siempre respetan límites solo con instrucciones textuales
- Necesitan validación automática para garantizar compliance

**Solución automática:**
- El agente reintentará automáticamente (hasta 2 veces con `retries=2`)
- Pydantic le dirá exactamente cuál fue el error
- El LLM ajustará la longitud en el siguiente intento

**Si persiste tras 2 reintentos:**
- El modelo puede necesitar instrucciones más enfáticas
- Considera añadir más ejemplos en las instrucciones
- Alternativamente, incrementa `retries=3`

---

### El agente no calcula correctamente los porcentajes
**Causa:** El LLM está "alucinando" números en lugar de calcularlos.

**Diagnóstico:**
1. Verifica que `construir_contexto_datos()` incluye los datos correctamente
2. Añade un `print(INSTRUCCIONES)` antes de crear el agente para ver qué prompt recibe
3. Los datos deben estar en formato legible: `[42k, 43k, 44k]` no `[42000, 43000, 44000]`

**Solución:**
- Los datos están correctamente formateados en la solución
- El LLM moderno (gpt-5-mini) puede hacer aritmética básica correctamente
- Si ves cálculos incorrectos, puede ser un problema de contexto truncado (poco probable con estos datos)

---

### La respuesta es demasiado genérica
**Ejemplo:** "Las ventas están bien" en lugar de "Q3 facturó 145k€..."

**Causa:** El LLM no está interpretando correctamente las instrucciones o los datos.

**Diagnóstico:**
1. Verifica el ejemplo en las instrucciones (debe ser concreto con números)
2. Asegúrate que `{construir_contexto_datos()}` se está inyectando correctamente
3. Revisa que el formato de `DATOS_NEGOCIO` no tiene errores

**Solución en la implementación:**
- ✅ Ejemplo concreto incluido: "Q3 facturó 145k€ (+12% vs Q2)..."
- ✅ Instrucciones enfatizan: "números concretos, sin jerga técnica"
- ✅ Datos bien estructurados y formateados

---

### `datos_calculados` está vacío
**Observado en la salida real:** El campo existe pero no tiene valores.

**¿Es un problema?**  
No necesariamente. Es opcional (`default_factory=dict`).

**¿Por qué está vacío?**
- Las instrucciones no exigen explícitamente rellenar este campo
- El LLM priorizó incluir los cálculos en la respuesta textual
- El agente cumple todos los requisitos mínimos sin usarlo

**¿Cómo hacer que lo use?**

Modifica las instrucciones para ser más explícito:
```python
2. Calcula las métricas relevantes y guárdalas en datos_calculados con estas claves:
   - Para resumen: "total_q3", "total_q2", "crecimiento_pct"
   - Para comparativa: "{producto}_crecimiento_pct" para cada producto
   - Para forecast: "tendencia_mensual_pct", "proyeccion_siguiente_mes"
```

O haz el campo obligatorio:
```python
datos_calculados: dict[str, Any] = Field(
    ...,  # ← Remueve default_factory, ahora es obligatorio
    min_length=1,  # ← Al menos una métrica
    description="Métricas calculadas relevantes"
)
```

---

## 💡 Lecciones aprendidas

### 1. **Los LLMs necesitan validación, no solo instrucciones**

**Descubrimiento:**  
Decir "máximo 280 caracteres" en el prompt NO es suficiente. El LLM lo intentará, pero no lo garantiza.

**Solución:**  
`Field(max_length=280)` + `retries=2` → Validación automática con oportunidades de corregir.

---

### 2. **Single source of truth es crítico**

**Antipatrón:**
```python
DATOS = {...}
PROMPT = "Ventas Q2: 42k, 43k, 44k"  # ← Duplicación
```

Si cambias `DATOS`, el prompt no se actualiza. Bug garantizado.

**Solución:**
```python
DATOS = {...}
PROMPT = f"{construir_contexto_datos()}"  # ← Construcción dinámica
```

---

### 3. **Los ejemplos concretos son más efectivos que las descripciones**

**Menos efectivo:**
```
Genera una respuesta concisa y profesional con cifras relevantes.
```

**Más efectivo:**
```
EJEMPLO VÁLIDO:
"Q3 facturó 145k€ (+12% vs Q2). Premium lideró crecimiento con +33%..."
```

El LLM aprende el formato, tono y estructura directamente del ejemplo.

---

### 4. **Los datos mock deben ser realistas**

**Por qué importa:**
- Datos realistas te preparan para producción
- Detectas problemas de escala (¿qué pasa con 1000 productos?)
- Entiendes limitaciones del LLM con datos complejos

**En esta solución:**
- Ventas mensuales reales con crecimiento gradual (42k → 43k → 44k)
- Productos con crecimientos diferenciados (Premium +33%, Estándar +5%)
- Estructura escalable a más trimestres, regiones, métricas

---

### 5. **La iteración es parte del proceso**

**Primera versión típica:**
- Respuestas de 500+ caracteres
- Porcentajes incorrectos
- Lenguaje demasiado técnico

**Después de ajustes:**
- `max_length=280` + `retries=2`
- Datos en formato "k" más legible
- Ejemplo concreto en instrucciones

**Moraleja:** No esperes perfección en el primer intento. Itera basándote en los resultados.

---

## 🎓 Conceptos clave del Módulo 0

Al completar este caso de uso has practicado:

- ✅ **Configuración de agente**: `Agent(model, output_type, retries, instructions)`
- ✅ **Validación estructurada**: `BaseModel`, `Field()`, `Literal`, constraints
- ✅ **Patrón Model + Provider**: Configuración explícita de credenciales
- ✅ **Construcción dinámica de prompts**: Single source of truth
- ✅ **Gestión de reintentos**: Recuperación automática de errores de validación
- ✅ **Output profesional**: Lenguaje ejecutivo, conciso, accionable

---

## 🚀 Próximos pasos

Esta es la **versión básica** del Módulo 0. En el **Módulo 1** evolucionarás este agente con:

### Tools avanzadas
```python
@agent.tool
async def calcular_metricas(ctx, periodo: str) -> dict:
    """Calcula métricas con trazabilidad completa."""
    # Acceso a datos reales via ctx.deps
    return {"total": 145000, "crecimiento": 12.4}
```

### Result validators con reflection
```python
@agent.result_validator
async def validate_quality(ctx, result):
    if result.nivel_confianza < 0.6:
        raise ModelRetry("La confianza es muy baja. Recalcula...")
    return result
```

### Dependency injection
```python
@dataclass
class DatabaseDeps:
    db: DatabaseConnection
    user: str

agent = Agent(model, deps_type=DatabaseDeps)
# Las tools acceden a db via ctx.deps.db
```

### Manejo de errores robusto
```python
try:
    result = agent.run_sync(query)
except ModelRetry:
    # Agotó reintentos
    return fallback_response()
except Exception as e:
    # Error del sistema
    log_error(e)
    return {"error": "ERR-001", "message": "Contacta soporte"}
```

---

## 📚 Referencias

- [PydanticAI Documentation](https://ai.pydantic.dev)
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [OpenAI Models](https://platform.openai.com/docs/models/)

---

**© 2025 Alfons Freixes | PydanticAI Course**