import json
import os

def load_config():
    """Carga la configuración desde el archivo config.json si existe."""
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('token')
    return None

def save_config(token):
    """Guarda el token en el archivo config.json."""
    with open('config.json', 'w') as f:
        json.dump({"token": token}, f)

def log(msg, sio, token):
    """Logea un mensaje y lo envía al servidor mediante socketio."""
    sio.emit('msg_to_web', {"msg": msg, "token": token})

def take_screenshot():
    """Toma una captura de pantalla y la devuelve en formato base64."""
    import pyautogui
    import base64
    from io import BytesIO
    
    screenshot = pyautogui.screenshot()
    buffer = BytesIO()
    screenshot.save(buffer, format='PNG')
    screenshot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return screenshot_base64

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
