import json
import os


def _rect_to_dict(rect):
    """Convierte un rectángulo de UI Automation en un diccionario serializable."""
    if rect is None:
        return None

    # El objeto Rect de uiautomation expone atributos left/top/right/bottom.
    for attr in ("left", "top", "right", "bottom"):
        if not hasattr(rect, attr):
            break
    else:
        left = getattr(rect, "left")
        top = getattr(rect, "top")
        right = getattr(rect, "right")
        bottom = getattr(rect, "bottom")
        if None in (left, top, right, bottom):
            return None
        width = max(0, right - left)
        height = max(0, bottom - top)
        return {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "width": int(width),
            "height": int(height),
        }

    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        left, top, right, bottom = rect
        if None in (left, top, right, bottom):
            return None
        width = max(0, right - left)
        height = max(0, bottom - top)
        return {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "width": int(width),
            "height": int(height),
        }

    return None


def collect_actionable_ui_controls(max_depth=5):
    """Obtiene controles accionables disponibles mediante UI Automation (solo Windows)."""
    actionable_controls = []

    try:
        import platform
        if platform.system().lower() != "windows":
            return actionable_controls
        import uiautomation as auto
    except Exception:
        # Si la plataforma no es Windows o falta la librería, devolvemos vacío.
        return actionable_controls

    try:
        root = auto.GetRootControl()
    except Exception:
        return actionable_controls

    actionable_types = {
        "Button",
        "ButtonControl",
        "CheckBox",
        "CheckBoxControl",
        "ComboBox",
        "ComboBoxControl",
        "Edit",
        "EditControl",
        "Hyperlink",
        "HyperlinkControl",
        "ListItem",
        "ListItemControl",
        "MenuItem",
        "MenuItemControl",
        "RadioButton",
        "RadioButtonControl",
        "SplitButton",
        "SplitButtonControl",
        "TabItem",
        "TabItemControl",
        "TreeItem",
        "TreeItemControl",
        "ToggleButton",
        "ToggleButtonControl",
    }

    seen_runtime_ids = set()

    def normalize_type(control_type_name):
        if not control_type_name:
            return ""
        if control_type_name.endswith("Control"):
            return control_type_name[:-7]
        return control_type_name

    def walk(control, depth):
        if depth > max_depth:
            return
        try:
            children = control.GetChildren()
        except Exception:
            children = []

        for child in children:
            runtime_id = None
            try:
                runtime_id = tuple(getattr(child, "RuntimeId"))
            except Exception:
                pass

            if runtime_id and runtime_id in seen_runtime_ids:
                continue
            if runtime_id:
                seen_runtime_ids.add(runtime_id)

            try:
                control_type = getattr(child, "ControlTypeName", "")
            except Exception:
                control_type = ""

            normalized_type = normalize_type(control_type)

            is_enabled = False
            try:
                is_enabled = bool(getattr(child, "IsEnabled"))
            except Exception:
                pass

            is_offscreen = False
            try:
                is_offscreen = bool(getattr(child, "IsOffscreen"))
            except Exception:
                pass

            rect_dict = None
            try:
                rect_dict = _rect_to_dict(getattr(child, "BoundingRectangle", None))
            except Exception:
                rect_dict = None

            if (
                rect_dict
                and not is_offscreen
                and is_enabled
                and (control_type in actionable_types or normalized_type in actionable_types)
            ):
                element_info = {
                    "name": getattr(child, "Name", None),
                    "automationId": getattr(child, "AutomationId", None),
                    "controlType": control_type or normalized_type,
                    "localizedControlType": getattr(child, "LocalizedControlType", None),
                    "boundingRectangle": rect_dict,
                    "className": getattr(child, "ClassName", None),
                    "frameworkId": getattr(child, "FrameworkId", None),
                    "processId": getattr(child, "ProcessId", None),
                    "runtimeId": list(runtime_id) if runtime_id else None,
                }
                actionable_controls.append(element_info)

            walk(child, depth + 1)

    walk(root, 0)
    return actionable_controls


def load_config():
    """Carga la configuración desde el archivo config.json si existe."""
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        return {
            "token": config.get('token'),
            "server": config.get('server')
        }
    return {}


def save_config(token=None, server=None):
    """Guarda los valores de configuración proporcionados en el archivo config.json."""
    config = {}
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

    if token is not None:
        config['token'] = token
    if server is not None:
        config['server'] = server

    with open('config.json', 'w') as f:
        json.dump(config, f)

def log(msg, sio, token):
    """Logea un mensaje y lo envía al servidor mediante socketio."""
    sio.emit('msg_to_web', {"msg": msg, "token": token})


def _capture_raw_screenshot():
    import pyautogui

    return pyautogui.screenshot()


def _encode_image_to_base64(image):
    import base64
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def take_screenshot():
    """Toma una captura de pantalla y la devuelve en formato base64."""
    screenshot = _capture_raw_screenshot()
    return _encode_image_to_base64(screenshot)


def capture_screenshot_with_ui_metadata():
    """Captura un pantallazo y adjunta la información de controles accionables."""
    screenshot = _capture_raw_screenshot()
    screenshot_base64 = _encode_image_to_base64(screenshot)

    try:
        ui_elements = collect_actionable_ui_controls()
    except Exception:
        ui_elements = []

    width = getattr(screenshot, "width", None)
    height = getattr(screenshot, "height", None)
    if width is None or height is None:
        width, height = screenshot.size

    return {
        "image": screenshot_base64,
        "width": int(width),
        "height": int(height),
        "uiElements": ui_elements,
    }

def save_uploaded_file(filename, content):
    """Guarda un archivo subido en el directorio 'autom_files'."""
    directory_path = 'autom_files'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'wb') as file:
        file.write(content)

def execute_code(code, globals_dict, locals_dict):
    """Ejecuta el código proporcionado en los espacios de nombres globales y locales."""
    try:
        exec(code, globals_dict, locals_dict)
    except Exception as e:
        print(f"Error al ejecutar el código: {e}")

def get_scaled_rect(rectangulo, screenshot):
    """Calcula las coordenadas escaladas de un rectángulo dado en la captura de pantalla."""
    tamano_original = {"width": rectangulo["dimensionesOriginales"]["width"], "height": rectangulo["dimensionesOriginales"]["height"]}
    left_original = rectangulo["left"]  
    top_original = rectangulo["top"]
    width_original = rectangulo["width"]
    height_original = rectangulo["height"]

    left_escalado = left_original * ((screenshot.size[0]) / (tamano_original["width"]))  
    top_escalado = top_original * (screenshot.size[1] / tamano_original["height"])
    width_escalado = width_original * (screenshot.size[0] / tamano_original["width"])
    height_escalado = height_original * (screenshot.size[1] / tamano_original["height"])

    return left_escalado, top_escalado, width_escalado, height_escalado
