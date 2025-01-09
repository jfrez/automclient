import sys
import os
from PyQt5.QtWidgets import QApplication
from ui import MainWindow
from PyQt5.QtCore import QTimer

# Añadir el directorio de librerías al PATH
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

libs_path = os.path.join(base_path, 'libraries')
os.environ['PATH'] = libs_path + os.pathsep + os.environ['PATH']

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.setApp(app)
    sys.exit(app.exec_())
# pyinstaller  --onedir --noconsole --windowed --icon=logo.ico --add-data="../images/*;images/" --noconfirm --name="Autom" "..\main.py"