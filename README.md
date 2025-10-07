# automclient

## Detección de elementos de UI en Windows

Para enumerar los controles accionables (botones, cajas de texto, etc.) de una ventana cuando el cliente se ejecuta sobre Windows, se puede aprovechar la API de UI Automation expuesta por el sistema operativo. Desde Python existen varios *wrappers* listos para usar, por ejemplo [`uiautomation`](https://github.com/yinkaisheng/Python-UIAutomation-for-Windows) o [`pywinauto`](https://pywinauto.github.io/).

El siguiente ejemplo usa el paquete `uiautomation` para localizar todos los elementos invocables dentro de la ventana actualmente en primer plano:

```python
import uiautomation as auto

# Obtener la ventana enfocada en este momento
root = auto.GetFocusedControl().GetTopLevelControl()
print(f"Ventana activa: {root.Name}")

# Recorrer el árbol UIA buscando elementos con el patrón Invoke (clic) o Toggle
for control, depth in auto.WalkTree(root, includeTop=True):
    patterns = control.GetSupportedPatterns()
    if auto.PatternId.InvokePattern in patterns or auto.PatternId.TogglePattern in patterns:
        rect = control.BoundingRectangle
        print(
            " " * depth * 2
            + f"- {control.ControlTypeName} '{control.Name}'"
            + f" @ ({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"
        )
```

Este código imprime el nombre, tipo y rectángulo de cada control con patrón `Invoke` (botones, enlaces) o `Toggle` (checkboxes, switches). Se pueden añadir otras condiciones (por ejemplo `ValuePattern` para entradas de texto) según el tipo de elemento deseado.

Para ejecutar el script es necesario:

1. Instalar el paquete con `pip install uiautomation`.
2. Ejecutar el script con privilegios suficientes para inspeccionar otras aplicaciones (cuando sea necesario).
3. Opcionalmente, usar la herramienta `Inspect.exe` del *Windows SDK* para visualizar los identificadores UIA y ajustar los filtros.

Con este enfoque se puede generar un inventario de elementos interactivos en la pantalla y acompañar la captura con metadatos antes de enviarla al servidor.

## Capturas enriquecidas desde el cliente

El flujo `client_take_screenshot` ahora adjunta automáticamente los controles detectados mediante UI Automation al mensaje que se envía al servidor. Cada captura incluye:

- `image`: la captura de pantalla en Base64 (PNG).
- `width` y `height`: dimensiones de la imagen capturada.
- `uiElements`: lista de controles accionables visibles (botones, casillas, pestañas, etc.) con su nombre accesible, tipo, `AutomationId`, `RuntimeId`, `ClassName`, `FrameworkId`, `ProcessId` y rectángulo delimitador en coordenadas de pantalla.

La recolección de metadatos solo se activa cuando el cliente se ejecuta en Windows y tiene disponible la librería `uiautomation`. En otras plataformas o si la dependencia no está instalada, el listado de elementos se envía vacío sin interferir con el envío del pantallazo.
