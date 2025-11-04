# 🧭 Caso de uso empresarial: **DataPulse AI – Asistente de análisis de datos para PYMEs**

> **Mini-proyecto transversal del curso PydanticAI**
> A lo largo del curso, construirás paso a paso un agente inteligente que ayuda a empresas a entender y resumir sus datos internos de negocio.

---

## 🧩 Contexto

En 2025, las pequeñas y medianas empresas están adoptando herramientas de **IA generativa aplicada a datos** para mejorar la toma de decisiones.
Sin embargo, la mayoría **no dispone de personal técnico** capaz de programar consultas o interpretar dashboards complejos.

**DataPulse AI** nace como una iniciativa interna de una consultora de datos.
Su objetivo: desarrollar un **asistente conversacional** que permita a los responsables de negocio **hacer preguntas en lenguaje natural sobre sus datos de ventas, clientes y productos** y obtener respuestas precisas, breves y con lenguaje ejecutivo.

---

## 🎯 Objetivo general del proyecto

Construir un **agente de análisis empresarial** que:

1. Reciba consultas en lenguaje natural sobre datos (por ejemplo: "¿qué producto tuvo más crecimiento este trimestre?").
2. Interprete la intención y clasifique el tipo de consulta.
3. Ejecute funciones o tools según la necesidad (resumen, forecast, ranking, etc.).
4. Devuelva una respuesta profesional y validada en formato estructurado (por ejemplo JSON con título, insight y recomendación).

El asistente deberá **evolucionar módulo a módulo** incorporando:

* Validación y tipado (Módulo 1)
* Tools analíticas simples (Módulo 2)
* Agentes encadenados o jerárquicos (Módulo 3)
* Conexión a fuentes reales (CSV o API) (Módulo 4)
* Persistencia y despliegue con Gradio / Docker (Módulo final)

---

## 🪜 Etapa 1 – Módulo 0 · Introducción

### 📘 Enunciado del ejercicio

Como punto de partida, queremos crear un **prototipo básico** del agente DataPulse AI que:

1. **Reciba un mensaje de usuario** (consulta informal sobre datos de negocio).
2. **Clasifique la intención** del mensaje en una de tres categorías:
   * `"resumen"` – el usuario pide un resumen o interpretación general.
   * `"comparativa"` – el usuario compara períodos, productos o regiones.
   * `"forecast"` – el usuario pregunta por proyecciones o tendencias futuras.
3. **Calcule métricas relevantes** (totales, crecimientos, rankings) y las almacene estructuradamente.
4. **Genere una respuesta ejecutiva breve (2-3 frases, máximo 280 caracteres)** con tono profesional.

### 💡 Ejemplo de uso
```python
>>> "¿Cómo van las ventas del último trimestre comparadas con el anterior?"
{
  "intencion": "comparativa",
  "datos_calculados": {
    "total_q2": 129000,
    "total_q3": 145000,
    "crecimiento_pct": 12.4
  },
  "respuesta": "Q3 facturó 145k€ (+12.4% vs Q2). Premium lideró el crecimiento. Recomendación: mantener estrategia de precios premium."
}
```

### 🧠 Conceptos que se ponen en práctica

* Definición de agente básico con `Model(..., provider=Provider(...))`
* Validación estructurada con `BaseModel` y `Field()` constraints
* Instrucciones claras con ejemplos inline
* Gestión de reintentos con `retries`
* Ejecución con `uv run python caso_uso_01.py`

---

## 🧑‍💻 Tu tarea

Crea un script llamado `00-introduccion/caso_uso_01.py` que:

### 1. Define el modelo de salida estructurada

Crea una clase `BusinessResponse` con tres campos:
- **`intencion`**: clasificación del tipo de consulta usando `Literal["resumen", "comparativa", "forecast"]`
- **`datos_calculados`**: diccionario flexible que almacene métricas relevantes (totales, porcentajes, rankings)
- **`respuesta`**: texto ejecutivo con **máximo 280 caracteres**

💡 Usa `Field()` de Pydantic para añadir constraints de validación (especialmente `max_length` en `respuesta`).

### 2. Configura el agente con el patrón correcto

Usa OpenAI (gpt-5-mini) o Anthropic siguiendo el patrón explícito:
```python
model = Model("gpt-5-mini", provider=Provider(api_key=settings.openai_api_key))
agent = Agent(
    model, 
    output_type=BusinessResponse,
    retries=2,  # Importante: permite reintentos si falla validación
    instructions="..."
)
```

### 3. Proporciona datos de negocio en las instrucciones

Incluye estos datos **directamente en el prompt** (inline):
- Ventas Q2 2025: [42k, 43k, 44k] mensuales
- Ventas Q3 2025: [45k, 48k, 52k] mensuales
- Productos Q2→Q3: Premium (18k→24k), Estándar (20k→21k), Básico (6k→7k)

**Importante**: NO calcules porcentajes manualmente. Deja que el agente haga los cálculos.

### 4. Escribe instrucciones claras y específicas

Tus instrucciones deben:
- Especificar las tres tareas: clasificar, calcular, responder
- Incluir límites explícitos (280 caracteres máximo)
- Proporcionar un **ejemplo válido** del formato esperado
- Usar lenguaje directo y ejecutivo

### 5. Ejecuta pruebas con tres consultas

Implementa una función que ejecute estas tres consultas:
- "Resúmeme las ventas del último trimestre"
- "¿Qué producto creció más este trimestre comparado con el anterior?"
- "¿Qué esperas para el próximo mes basándote en las tendencias actuales?"

Imprime los resultados mostrando:
- ✓ Intención detectada
- ✓ Datos calculados (si los hay)
- ✓ Respuesta ejecutiva con longitud

---

## 💡 Consejos técnicos

### Sobre validación
- **Longitud de respuesta**: `max_length=280` es el límite de un tweet extendido - referencia familiar y profesional
- **Reintentos**: `retries=2` da dos oportunidades al modelo si genera respuestas demasiado largas en el primer intento

### Sobre las instrucciones
- **Datos inline**: Incluye los datos directamente en el texto de las instrucciones, no como variables Python
- **Ejemplo concreto**: Muestra exactamente el formato que esperas (estructura y longitud)
- **Límites explícitos**: Di "MÁXIMO 280 caracteres" en mayúsculas para enfatizar

### Sobre los cálculos
- **Deja calcular al LLM**: No precalcules porcentajes. Los modelos modernos pueden hacer aritmética simple
- **Esto es más realista**: En producción recibirás datos en bruto desde bases de datos o APIs

### Manejo de errores
- Envuelve la ejecución en `try/except` para capturar errores de validación
- Muestra mensajes informativos que ayuden a diagnosticar problemas

---

## 📈 Qué evaluaremos en esta etapa

| Aspecto | Descripción | Peso |
|---------|-------------|------|
| 💬 Claridad del prompt | Las instrucciones del agente son precisas, concisas y con ejemplo | 15% |
| 🧱 Uso correcto del modelo y provider | Se usa el patrón `Model(..., provider=Provider(...))` correctamente | 15% |
| ✅ Validación del output | La respuesta cumple el modelo `BusinessResponse` con todos los constraints | 25% |
| 📊 Datos calculados | El campo `datos_calculados` contiene métricas relevantes y correctas | 15% |
| 🧠 Relevancia empresarial | La respuesta suena realista, profesional y accionable | 20% |
| 🧩 Legibilidad del código | Código limpio, bien estructurado, con comentarios útiles | 10% |

---

## 🚀 Próximos pasos

### En el Módulo 1 evolucionarás este agente para:
- Implementar **tools personalizadas** para cálculos complejos (garantizando precisión y trazabilidad)
- Añadir **validadores custom** de Pydantic para reglas de negocio específicas
- Usar **reflection** para que el agente autocorrija respuestas incoherentes o inconsistentes

### En el Módulo 2 aprenderás a:
- Gestionar **contexto conversacional** y memoria del agente entre múltiples interacciones
- Conectar con **fuentes de datos reales** (archivos CSV, bases de datos SQL, APIs REST)
- Implementar **streaming** para respuestas progresivas en interfaces de usuario

### En el Módulo 3 construirás:
- **Agentes jerárquicos** que delegan subtareas a agentes especializados
- **Workflows complejos** con encadenamiento de múltiples agentes
- **Integración con herramientas** externas (búsqueda web, generación de gráficos)

### En el Módulo final desplegarás:
- Interfaz web con **Gradio** o **Streamlit**
- **Containerización** con Docker para portabilidad
- **Monitoreo** con Pydantic Logfire para observabilidad en producción

---

## 📚 Conceptos clave aprendidos

Al completar esta etapa habrás practicado:

✅ **Arquitectura básica** de un agente con PydanticAI  
✅ **Validación estructurada** con Pydantic (tipos, constraints, defaults)  
✅ **Patrón Model + Provider** para configuración explícita de LLMs  
✅ **Prompt engineering** efectivo (claridad, ejemplos, límites)  
✅ **Manejo de errores** de validación con reintentos  
✅ **Output profesional** adaptado a audiencia ejecutiva  

Estos fundamentos son la base sobre la que construirás agentes cada vez más sofisticados en los siguientes módulos.

---

<div style="text-align:center; margin-top:40px; font-size:0.9em; color:#64748b;">
  💡 <strong>Consejo pedagógico</strong>: Si tu agente genera respuestas demasiado largas, revisa que:
  <br>1. El campo <code>respuesta</code> tenga <code>max_length=280</code>
  <br>2. Las instrucciones mencionen explícitamente el límite de 280 caracteres
  <br>3. Hayas configurado <code>retries=2</code> para dar oportunidades de corrección
  <br>4. Incluyas un ejemplo concreto del formato esperado
</div>