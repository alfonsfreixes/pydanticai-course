# Ejemplo 02: Unit Testing & TestModel (Zero-Cost Testing)

En producción, no podemos depender de llamar al LLM para verificar nuestro código. Sería lento, costoso y no determinista.

La estrategia correcta para testear Agentes de IA sigue una **Pirámide de Testing**:

1.  **Tests Unitarios (Lógica Pura):** Probamos el `Service Layer` directamente. Es Python puro, instantáneo.
2.  **Tests de Integración (Wiring):** Usamos `TestModel` de PydanticAI para verificar que el agente sabe llamar a las herramientas, sin contactar a OpenAI.
3.  **Tests E2E (Opcional):** Solo en entornos de staging, llamando al LLM real.

## ⚙️ Configuración Previa

Antes de escribir el test, necesitamos instalar las librerías de testing en nuestro grupo de desarrollo. Las versiones modernas son necesarias para soportar las últimas características asíncronas.

1. **Instala las dependencias:**

```bash
uv add --dev pytest pytest-asyncio
````

Esto actualizará tu `pyproject.toml` automáticamente. El grupo `dev` quedará similar a esto:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.14.6",
]
```

2.  **Configura el modo asíncrono:**

Añade esto al final de tu `pyproject.toml` para que `pytest` maneje `asyncio` automáticamente:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["01-core-patterns"] 
```

## 📝 Código

Crea el archivo `01-core-patterns/test_crm_agent.py`.

```python
import pytest
from pydantic_ai.models.test import TestModel

# Importamos los componentes de nuestro agente CRM
# Asegúrate de que crm_agent.py existe en el mismo directorio
from crm_agent import CRMDeps, Customer, CustomerService, agent


# --- CONSTANTES DE TEST ---
TEST_EMAIL = "test@test.com"
TEST_NAME = "Test User"
CHURNED_EMAIL = "sad@guy.com"
CHURNED_NAME = "Churned Guy"


# --- 1. MOCK SERVICE (Simulación para Test) ---
class MockCustomerService(CustomerService):
    """
    Versión controlada del servicio para tests.
    Datos predecibles sin depender de DBs externas.
    """

    def __init__(self):
        self._db = {
            99: Customer(id=99, name=TEST_NAME, email=TEST_EMAIL, status="active"),
            100: Customer(id=100, name=CHURNED_NAME, email=CHURNED_EMAIL, status="churned"),
        }


# --- 2. NIVEL 1: TESTS UNITARIOS DEL SERVICIO (Sin LLM) ---
# Aquí probamos la lógica de negocio pura. Si esto falla, el problema es tuyo, no de la IA.

@pytest.mark.asyncio
async def test_service_get_by_email_found():
    """Verifica que el servicio recupera datos correctamente."""
    service = MockCustomerService()
    customer = await service.get_by_email(TEST_EMAIL)

    assert customer is not None
    assert customer.name == TEST_NAME
    assert customer.status == "active"


@pytest.mark.asyncio
async def test_service_get_by_email_not_found():
    """Verifica que el servicio retorna None para emails inexistentes."""
    service = MockCustomerService()
    customer = await service.get_by_email("noexiste@fake.com")

    assert customer is None


@pytest.mark.asyncio
async def test_service_search_fuzzy():
    """Verifica la lógica de búsqueda difusa."""
    service = MockCustomerService()
    results = await service.search_by_name("Guy")

    assert len(results) == 1
    assert results[0].email == CHURNED_EMAIL


@pytest.mark.asyncio
async def test_service_search_no_results():
    """Verifica búsqueda sin coincidencias."""
    service = MockCustomerService()
    results = await service.search_by_name("ZZZ_NoExiste")

    assert len(results) == 0


# --- 3. NIVEL 2: TESTS DE INTEGRACIÓN CON TestModel ---
# Aquí probamos que el Agente está bien "cableado" y sabe usar las tools.
# TestModel simula ser un LLM, pero es gratuito y local.

@pytest.mark.asyncio
async def test_agent_tool_execution():
    """Verificamos que el agente ejecuta sin errores."""
    service = MockCustomerService()
    deps = CRMDeps(crm=service, request_id="test-req-agent")

    # Opción A: TestModel genérico (devuelve texto plano simulado)
    with agent.override(model=TestModel()):
        result = await agent.run("Hola", deps=deps)
        assert result.data is not None


@pytest.mark.asyncio
async def test_agent_specific_tool_call():
    """Forzamos a TestModel a invocar la herramienta."""
    service = MockCustomerService()
    deps = CRMDeps(crm=service, request_id="test-req-tool")

    # Opción B: TestModel configurado para llamar tools
    test_model = TestModel(call_tools=["lookup_customer"])

    with agent.override(model=test_model):
        result = await agent.run("Busca a test@test.com", deps=deps)
        
        # Verificamos que no hubo excepciones en la ejecución de la tool
        assert result is not None
```

## 🏃‍♂️ Cómo ejecutar el test

Ahora que tenemos las dependencias instaladas y configuradas, ejecutamos:

```bash
uv run pytest 01-core-patterns/test_crm_agent.py -v
```

Deberías ver una salida verde indicando que los tests pasaron en milisegundos.

## 🔍 Análisis de la Estrategia

1.  **Service Layer Tests:** Son los más importantes. Cubren el 90% de tu código (validaciones, cálculos, accesos a datos). Son rápidos y deterministas.
2.  **`TestModel`:** Es la herramienta oficial de PydanticAI para testing. Reemplaza al modelo real (`OpenAIChatModel`) por uno simulado.
      * **Beneficio:** No necesitas API Key. No gastas dinero. Funciona offline.
      * **Uso:** `with agent.override(model=TestModel()): ...` reemplaza temporalmente el cerebro del agente.
3.  **Seguridad:** Al testear el servicio directamente, garantizas que tu lógica es sólida antes de añadir la capa de incertidumbre que supone un LLM.