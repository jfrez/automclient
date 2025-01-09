from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QInputDialog, QSizePolicy
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap


import os
import json
import time
from communication import sio, CodeRunner, stop_signal

import pygame
from sounds import generate_r2d2_hello_sound,generate_r2d2_yeah_sound,generate_r2d2_help_sound
pygame.mixer.init(frequency=44100, size=-16, channels=1)

server = 'https://autom.be'
#server = 'http://127.0.0.1:8000'
window=None
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setWindowTitle('Robot AUTOM')
        self.button = QPushButton('Conectar')
        self.continuar = QPushButton('Continuar')
        self.paused = False

        self.stop = QPushButton('Detener')
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setMinimumHeight(50)
        self.setMinimumWidth(200)

        #help
        self.text_label = QLabel("")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        spacer = QSpacerItem(0, 30, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.layout.addWidget(self.button)

        self.layout.addWidget(self.label)
        self.layout.addItem(spacer)  # Añadir el espaciador

        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.continuar)
        self.continuar.clicked.connect(self.on_continue_clicked)

        self.layout.addWidget(self.stop)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.app=None
        sio.window = self
        self.button.clicked.connect(self.conectar)
        self.stop.clicked.connect(self.stoper)
        self.button.setText("Conectar/Desconectar")
        self.code_runner = CodeRunner("",0, stop_signal)
        self.setWindowIcon(QIcon('logo.ico'))
        sio.token = self.obtener_token()
        QTimer.singleShot(10, self.conectar)
        
    def show_continue_button(self):
    
        self.continuar.show()
        self.paused=True

    def hide_continue_button(self):
        self.paused=False
        self.continuar.hide()
        self.text_label.setText("")
        self.image_label.clear()

    def on_continue_clicked(self):
        self.paused= False
        self.hide_continue_button()

    def closeEvent(self, event):
        # Aquí puedes añadir cualquier limpieza adicional si es necesario
        self.app.quit()
        
    def exit(self):
        QApplication.instance().quit()
    def setApp(self,app):
        self.app=app

    def runner(self, code,jobid):
        self.start()
        self.code_runner = CodeRunner(code, jobid,stop_signal)
        self.code_runner.gogogo()
        self.code_runner.start()

    def stoper(self):
        if not stop_signal["stop"]:
            self.code_runner.stop()
            self.error()
            sio.emit('test_code_to_web', {"status": f'Stop', "token": sio.token})
        self.error()

    def conectar(self):
        sio.token = self.obtener_token()
        if sio.token:
            try:
                self.start()
                sio.disconnect()
                print("Connecting: ",server)
                sio.connect(server, headers={"Authorization": sio.token})
                self.button.clicked.disconnect(self.conectar)
                self.button.clicked.connect(self.desconectar)
                self.conn()
                self.continuar.hide()
                sio.window=self
            except Exception as e:
                print(e)
                self.error()
        else:
            self.label.setText("Token no proporcionado.")

    def obtener_token(self):
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                sio.token = config.get('token')
                if sio.token:
                    return sio.token
        sio.token, ok = QInputDialog.getText(self, 'Input Dialog', 'Ingrese su token:')
        if ok and sio.token:
            with open('config.json', 'w') as f:
                json.dump({"token": sio.token}, f)
            return sio.token
        return None

    def desconectar(self):
        sio.disconnect()
        self.button.clicked.disconnect(self.desconectar)
        self.button.clicked.connect(self.conectar)
        self.button.setText("Conectar")
        self.label.setText("desconectado")
        self.pixmap = QPixmap('images/logo_disc.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))

    def click(self):
        self.pixmap = QPixmap('images/click.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))
    
    def lookup(self):
        self.pixmap = QPixmap('images/lookup.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))
    
    def press(self):
        self.pixmap = QPixmap('images/key.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))

    def conn(self):
        self.pixmap = QPixmap('images/start.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))
        self.button.setText("Desconectar")
        sound = generate_r2d2_hello_sound()
        sound.play()
        QTimer.singleShot(1000, self.start)

    
    def error(self):
        self.pixmap = QPixmap('images/error.png')
        self.label.setPixmap(self.pixmap.scaled(50, 50))
        QTimer.singleShot(1000, self.start)


    def start(self):
   
            self.pixmap = QPixmap('images/logo.png')

            self.label.setPixmap(self.pixmap.scaled(50, 50))
    def help_text(self,msg,data):
        msg += " " + data
        self.text_label.setText(msg)
        time.sleep(10)
        self.text_label.setText("")

    def text(self,msg):
        self.text_label.setText(msg)
        time.sleep(1)
        self.text_label.setText("")

    def help_image(self,msg, data):
        # Crear un nuevo QLabel para mostrar el mensaje
        self.text_label.setText(msg)
        import base64
        from PyQt5.QtCore import QByteArray
        from PyQt5.QtGui import QPixmap

        image_data = base64.b64decode(data)
        byte_array = QByteArray(image_data)
        pixmap = QPixmap()
        pixmap.loadFromData(byte_array)
        self.image_label.setPixmap(pixmap.scaled(200, 100, Qt.KeepAspectRatio))
        time.sleep(10)
        self.image_label.clear()  
        self.text_label.setText("")  
        
    def end_help(self):
        window.text_label.setText("")
        window.image_label.clear()

