import socketio
import threading
from utils import load_config, save_config
from PyQt5.QtWidgets import QApplication
from contextlib import suppress
import os

window = None
sio = socketio.Client()
stop_signal = {"stop": False}
token = load_config()  # Carga el token desde la configuración
sio.token = token
class CodeRunner(threading.Thread):
    def __init__(self, code,jobid, stop_signal):
        super().__init__()
        self.code = code
        self.jobid = jobid
        self.stop_signal = stop_signal

    def run(self):
        local_namespace = {"stop_signal": self.stop_signal}
        import pygame
        from sounds import generate_r2d2_hello_sound,generate_r2d2_yeah_sound,generate_r2d2_help_sound
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        output="{}"
        try:
            from actions import dblclick_text,hover_simple,wait_exists, wait_exists_text,is_text_visible,click_text,rightclick,mkdir,up,down,right,left, press,press2, sleep, click, dblclick, wait_exists, exists, leer, escribir_texto, teclear, chatGPT, agregar_a_excel, email
            sio.token = load_config()
            # Añadir las funciones importadas al namespace local
            local_namespace.update({
                "up": up,
                "down": down,
                "right": right,
                "left": left,   
                "press": press,
                "press2":press2,
                "sleep": sleep,
                "click": click,
                "dblclick": dblclick,
                "hover_simple":hover_simple,
                "rightclick":rightclick,
                "click_text":click_text,
                "dblclick_text":dblclick_text,
                "is_text_visible":is_text_visible,
                "wait_exists": wait_exists,
                "wait_exists_text": wait_exists_text,
                "exists": exists,
                "leer": leer,
                "escribir_texto": escribir_texto,
                "teclear": teclear,
                "chatGPT": chatGPT,
                "agregar_a_excel": agregar_a_excel,
                "email": email,
                "token":sio.token,
                "mkdir":mkdir,
                "sio": sio,
                "window":sio.window
            })
            
            
         
            from io import StringIO
            from contextlib import redirect_stdout
            f = StringIO()
            with redirect_stdout(f):
                exec(self.code, globals(), local_namespace)
            output = f.getvalue()
            print("output",output)
            sio.emit('client_ok', {"status": 'ok', "token": sio.token, "output": output,"jobid":self.jobid})
            sio.window.start()
        except Exception as e:
            print(f"Error: {e}")
            sio.emit('test_code_to_web', {"status": f'Error:  {e}', "token": sio.token})
            sio.emit('client_error', {"status": 'failed', "token": sio.token,"output": output,"jobid":self.jobid})
            

    def stop(self):
        self.stop_signal["stop"] = True
        sio.emit('test_code_to_web', {"status": f'Stop', "token": sio.token})

    def gogogo(self):
        self.stop_signal["stop"] = False

def log(msg):
    sio.emit('msg_to_web', {"msg": msg, "token": sio.token})

@sio.event
def connect():
    sio.emit('register_client', {"token": sio.token})
    try:
        sio.window.button.setText("Desconectar")
    except:
        pass

@sio.event
def disconnect():
    sio.window.button.setText("Conectar")


@sio.on('client_take_screenshot')
def client_take_screenshot(code):
    import pyautogui
    import base64
    from io import BytesIO

    sio.window.lookup()

    try:
        screenshot = pyautogui.screenshot()
        buffer = BytesIO()
        screenshot.save(buffer, format='PNG')
        screenshot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        sio.emit('screenshot_from_client', {"image": screenshot_base64, "token": sio.token})
        sio.window.conn()
        print("client_take_screenshot", sio.token)
    except Exception as e:
        sio.window.error()
        print(e)

@sio.on('upload_file_to_client')
def upload_file_to_client(data):
    import os
    filename = data['filename']
    content = data['content']
    directory_path = 'autom_files'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_path = os.path.join(directory_path, filename)

    with open(file_path, 'wb') as file:
        file.write(content)

@sio.on('ping_to_client')
def ping_to_client(data):
    sio.emit('pong', {"status": "ok", "token": sio.token})

@sio.on('test_code_to_client')
def test_code_to_client(data):
    code = data["code"]
    jobid = data["jobid"]
    print(data)
    try:
        print("-------------")
        sio.window.lookup()
#        code += "\nimport sys\nsys.exit()\n"
        sio.emit('test_code_to_web_start', {"token": sio.token})
        sio.window.label.setText("Ejecutando")
        local_namespace = {"stop_signal": stop_signal}
        try:
            sio.window.press()
            sio.window.runner(code,jobid)
        except Exception as e:
            sio.window.error()
            print(f"Error al ejecutar el código: {e}")
        sio.emit('test_code_to_web', {"status": "ok", "token": sio.token})
    except Exception as e:
        print(e)
        sio.window.error()
        sio.emit('test_code_to_web', {"status": f'Error:  {e}', "token": sio.token})
