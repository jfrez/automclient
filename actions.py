import pyautogui
import time
import platform
import pyperclip
from communication import stop_signal, log,sio,token
from utils import take_screenshot, get_scaled_rect
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import cv2
import os
import io
import json
import pygame
from sounds import generate_r2d2_help_sound
pygame.mixer.init(frequency=44100, size=-16, channels=1)

import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'tesseract/tesseract.exe'



def up():
    if stop_signal['stop'] :
        return
    pyautogui.press("up")
    time.sleep(0.05)
def down():
    if stop_signal['stop'] :
        return
    sio.window.press()
    pyautogui.press("down")
    time.sleep(0.05)
def left():
    if stop_signal['stop'] :
        return
    sio.window.press()
    pyautogui.press("left")
    time.sleep(0.05)
def right():
    if stop_signal['stop'] :
        return
    sio.window.press()
    pyautogui.press("right")
    time.sleep(0.05)

def press(key):
    if stop_signal['stop']:
        return
    sio.window.press()
    pyautogui.press(key)
    time.sleep(0.05)

def press2(l):
    if stop_signal['stop'] :
        return
    sio.window.press()
    pyautogui.press([l])
    time.sleep(0.05)

def mkdir(ruta):
    if stop_signal['stop'] :
        return
    sio.window.press()
    directory_path = 'autom_files'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    os.mkdir(directory_path+'/'+ruta)

def sleep(s):
    if stop_signal['stop']:
        return
    time.sleep(s)
    time.sleep(0.5)


def help(tipo,data):
    sound = generate_r2d2_help_sound()
    sound.play()
    if tipo == "click_text":
        sio.window.help_text("Ayuda: click",data)
    if tipo == "click":
        sio.window.help_image("Ayuda: click",data)
    if tipo == "dblclick":
        sio.window.help_image("Ayuda: dblclick",data)
    if tipo == "dblclick_text":
        sio.window.help_text("Ayuda: dblclick",data)
    if tipo == "hover_simple":
        sio.window.help_image("Ayuda: hover",data)  
    if tipo == "rightclick":
        sio.window.help_image("Help: right click",data)        
    if tipo == "is_text_visible":
        sio.window.help_text("No encontrado ",data)
    if tipo == "wait_exists_text":
        sio.window.help_text("Buscando.. ",data)
    if tipo == "wait_exists":
        sio.window.help_image("Buscando.. ",data)



def find_text(text, screenshot):
    # Convertir la imagen de PIL a un array de numpy
    image = np.array(screenshot)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Configuración de Tesseract
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)
    
    # Dividir la frase en palabras
    words = text.lower().split()
    
    # Recorrer los resultados del OCR para encontrar la frase completa
    encontrado = False
    coords = (0, 0)
    for i in range(len(data['text'])):
        # Buscar la primera palabra de la frase
        if data['text'][i].lower() == words[0]:
            
            # Comprobar si las palabras siguientes forman la frase completa
            match = True
            for j in range(1, len(words)):
                print(data['text'][i + j].lower(),words[0])
                if i + j >= len(data['text']) or data['text'][i + j].lower() != words[j]:
                    print("NOOO")
                    match = False
                    break
            if match:
                # Si se encuentra la frase completa, obtener las coordenadas de la primera palabra
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                coords = (x + w // 2, y + h // 2)  # Coordenadas del centro del área encontrada
                encontrado = True
                break

    return (encontrado, coords)





def compare_image(image_to_find, screenshot):
    # Convertir screenshot a formato OpenCV
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Decodificar y convertir image_to_find a formato OpenCV
    base64_image = image_to_find.split(',')[1]
    image_to_find_aux = Image.open(BytesIO(base64.b64decode(base64_image)))
    image_to_find_aux = cv2.cvtColor(np.array(image_to_find_aux), cv2.COLOR_RGB2BGR)

 # Imprimir las resoluciones iniciales
    screenshot_height, screenshot_width = screenshot.shape[:2]
    image_to_find_height, image_to_find_width = image_to_find_aux.shape[:2]



    # Inicializar SIFT detector
    sift = cv2.SIFT_create()

    # Detectar y computar características clave y descriptores
    kp1, des1 = sift.detectAndCompute(screenshot, None)
    kp2, des2 = sift.detectAndCompute(image_to_find_aux, None)

    # Verificar si se encontraron descriptores
    if des1 is None or des2 is None:
        return (False, (0, 0))

    # Crear el objeto FLANN para emparejamiento
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Ordenar las coincidencias por distancia y mostrar las 10 menores
    matches = sorted(matches, key=lambda x: x[0].distance)


    # Calcular el promedio de las 10 distancias más cercanas
    top_10_distances = [match[0].distance for match in matches[:10]]
    average_distance = np.mean(top_10_distances)
    # Si el promedio es menor a 100, usar las 10 más cercanas
    good_matches = []
    if average_distance < 100:
        good_matches = [match[0] for match in matches[:10]]
    else:
        # Aplicar el ratio test de Lowe para encontrar las buenas coincidencias
        for m, n in matches:
            if m.distance < 10:
                good_matches.append(m)

    # Definir un umbral para considerar la imagen como encontrada
    threshold = 10  # Ajustar según sea necesario
    
    if len(good_matches) >= threshold:
        # Obtener los puntos de correspondencia en ambas imágenes
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # Calcular la homografía
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w, _ = image_to_find_aux.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Calcular la ubicación promedio de las esquinas transformadas
            mean_coords = np.mean(dst, axis=0).flatten()
            return (True, (int(mean_coords[0]), int(mean_coords[1])))

    return (False, (0, 0))



def compare_image2(image_to_find, screenshot, threshold=0.9):
    base64_image = image_to_find.split(',')[1]
    image_to_find_aux = Image.open(BytesIO(base64.b64decode(base64_image)))
    image_to_find_cv = cv2.cvtColor(np.array(image_to_find_aux), cv2.COLOR_RGB2BGR)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    main_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(image_to_find_cv, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    best_match = None
    best_val = -1

    # Iterate over scales
    for scale in np.linspace(0.5, 2, 10)[::-1]:
        # Resize the main image according to the scale
        resized_main = cv2.resize(main_gray, (int(main_gray.shape[1] * scale), int(main_gray.shape[0] * scale)))
        r = main_gray.shape[1] / float(resized_main.shape[1])

        if resized_main.shape[0] < h or resized_main.shape[1] < w:
            break

        # Perform template matching
        result = cv2.matchTemplate(resized_main, template_gray, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        
        if maxVal >= threshold and maxVal > best_val:
            best_val = maxVal
            best_match = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    print(maxVal)
    if best_match:
        return True, best_match
    else:
        return False, (0, 0)

def click_text(text):
    
    if stop_signal['stop']:
        return

    sio.window.click()
    screenshot = pyautogui.screenshot()
   
    encontrado,maxLoc = find_text(text,screenshot)

    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        
        if stop_signal['stop'] :
            return
        help("click_text",text)
        log("Texto no encontrado")
        screenshot = pyautogui.screenshot()
        encontrado,maxLoc = find_text(text,screenshot)
 
        if not sio.window.paused:
            sio.window.hide_continue_button()
            return True
    
    (startX, startY) = maxLoc
    pyautogui.moveTo(x=startX, y=startY, duration=0.5)
    time.sleep(0.25)
    pyautogui.click(x=startX, y=startY)
    time.sleep(0.5)
    return True

def hover_text(text):
    if stop_signal['stop']:
        return

    sio.window.click()
    screenshot = pyautogui.screenshot()

    encontrado,maxLoc = find_text(text,screenshot)
    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        help("click_text",text)
        log("Texto no encontrado")
        encontrado,maxLoc = find_text(text,screenshot)
        if not sio.window.paused:
            sio.window.hide_continue_button()
            return True
    
    (startX, startY) = maxLoc
    pyautogui.moveTo(x=startX, y=startY, duration=0.5)
    time.sleep(0.25)
    return True


def dblclick_text(text):
    if stop_signal['stop']:
        return

    sio.window.click()
    screenshot = pyautogui.screenshot()

    encontrado,maxLoc = find_text(text,screenshot)
    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        help("dblclick_text",text)
        log("Texto no encontrado")
        encontrado,maxLoc = find_text(text,screenshot)
        if not sio.window.paused:
            sio.window.hide_continue_button()
            return True 
        

    (startX, startY) = maxLoc
    pyautogui.moveTo(x=startX, y=startY, duration=0.5)
    time.sleep(0.25)
    pyautogui.doubleClick(x=startX, y=startY)
    time.sleep(0.5)
    return True

def is_text_visible(text):
    if stop_signal['stop']:
        return False
    val =wait_exists_text(text,3)
    sio.window.text("Buscando... "+text)
    if not val:
        help("is_text_visible",text)
    return val

def wait_exists_text(text,counter=-1):
    if stop_signal['stop']:
        return False
    sio.window.lookup()
    sio.window.text("")
    print("BUSCANDO")
    screenshot = pyautogui.screenshot()
    encontrado, _ = find_text(text, screenshot)
    print(encontrado)
    while not encontrado:
        if counter == 0:
            return False
        if stop_signal['stop']:
            return False  
        sio.window.text("")
        screenshot = pyautogui.screenshot()
        encontrado, _ = find_text(text, screenshot)
        sio.window.text("Buscando... "+text)
        if encontrado:
            return True
        counter -= 1
        time.sleep(0.5)
    return True
        

def click(image_to_find, xp, yp):
    if stop_signal['stop']:
        return
    sio.window.lookup()
    screenshot = pyautogui.screenshot()
    
    encontrado,maxLoc = compare_image2(image_to_find,screenshot)
    print(encontrado)
    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        if stop_signal['stop']:
            return
        base64_image = image_to_find.split(',')[1]
        help("click",base64_image)
        log("Boton no encontrado")
        screenshot = pyautogui.screenshot()
        encontrado,maxLoc = compare_image2(image_to_find,screenshot)
        if not sio.window.paused:
            sio.window.hide_continue_button()
            return True 
        
    sio.window.click()
    (startX, startY) = maxLoc
    screenshot_np = np.array(screenshot)
    # Obtener el tamaño de la pantalla
    screen_width, screen_height  = pyautogui.size()
    base64_image = image_to_find.split(',')[1]
    image_to_find_aux = Image.open(BytesIO(base64.b64decode(base64_image)))
    img_width, img_height = image_to_find_aux.size
    offset_x = xp * img_width
    offset_y = yp * img_height
    new_startX = startX + offset_x
    new_startY = startY + offset_y
    # Ajustar las coordenadas según la escala de la pantalla
    adjusted_x = (new_startX / screenshot_np.shape[1]) * screen_width
    adjusted_y = (new_startY / screenshot_np.shape[0]) * screen_height

    adjusted_x = (offset_x / screenshot_np.shape[1]) * screen_width
    adjusted_y = (offset_y / screenshot_np.shape[0]) * screen_height 

    pyautogui.moveTo(x=new_startX, y=new_startY, duration=0.5)

    time.sleep(0.25)
    pyautogui.click(x=new_startX, y=new_startY)
    time.sleep(0.5)
    return True


def hover_simple(image_to_find, xp, yp):
    if stop_signal['stop']:
        return

    sio.window.click()
    screenshot = pyautogui.screenshot()
    

    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        if stop_signal['stop']:
            return
        base64_image = image_to_find.split(',')[1]
        help("hover_simple",base64_image)
        log("Boton no encontrado")
        screenshot = pyautogui.screenshot()
        encontrado,maxLoc = compare_image2(image_to_find,screenshot)
    if not sio.window.paused:
            sio.window.hide_continue_button()
            return True 


    (startX, startY) = maxLoc
    screenshot_np = np.array(screenshot)
    # Obtener el tamaño de la pantalla
    screen_width, screen_height  = pyautogui.size()
    base64_image = image_to_find.split(',')[1]
    image_to_find_aux = Image.open(BytesIO(base64.b64decode(base64_image)))
    img_width, img_height = image_to_find_aux.size
    offset_x = xp * img_width
    offset_y = yp * img_height
    new_startX = startX + offset_x
    new_startY = startY + offset_y
    # Ajustar las coordenadas según la escala de la pantalla
    adjusted_x = (new_startX / screenshot_np.shape[1]) * screen_width
    adjusted_y = (new_startY / screenshot_np.shape[0]) * screen_height

    adjusted_x = (offset_x / screenshot_np.shape[1]) * screen_width
    adjusted_y = (offset_y / screenshot_np.shape[0]) * screen_height 

    pyautogui.moveTo(x=new_startX, y=new_startY, duration=0.5)

    time.sleep(0.25)
    return True


def rightclick(image_to_find, xp, yp):
    if stop_signal['stop']:
        return

    sio.window.click()
    screenshot = pyautogui.screenshot()
    
    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        if stop_signal['stop']:
            return
        base64_image = image_to_find.split(',')[1]
        help("rightclick",base64_image)
        log("Boton no encontrado")
        screenshot = pyautogui.screenshot()
        encontrado,maxLoc = compare_image2(image_to_find,screenshot)
    if not sio.window.paused:
            sio.window.hide_continue_button()
            return True 
    
    (startX, startY) = maxLoc
    screenshot_np = np.array(screenshot)
    # Obtener el tamaño de la pantalla
    screen_width, screen_height  = pyautogui.size()
    base64_image = image_to_find.split(',')[1]
    image_to_find_aux = Image.open(BytesIO(base64.b64decode(base64_image)))
    img_width, img_height = image_to_find_aux.size
    offset_x = xp * img_width
    offset_y = yp * img_height
    new_startX = startX + offset_x
    new_startY = startY + offset_y
    # Ajustar las coordenadas según la escala de la pantalla
    adjusted_x = (new_startX / screenshot_np.shape[1]) * screen_width
    adjusted_y = (new_startY / screenshot_np.shape[0]) * screen_height

    adjusted_x = (offset_x / screenshot_np.shape[1]) * screen_width
    adjusted_y = (offset_y / screenshot_np.shape[0]) * screen_height 

    pyautogui.moveTo(x=new_startX, y=new_startY, duration=0.5)

    time.sleep(0.25)
    pyautogui.click(x=new_startX, y=new_startY,button='right')
    time.sleep(0.5)
    return True


def dblclick(image_to_find, xp, yp):
    
    if stop_signal['stop']:
        return
    
    sio.window.click()
    screenshot = pyautogui.screenshot()
    encontrado, maxLoc = compare_image2(image_to_find, screenshot)

    if not encontrado:
        sio.window.show_continue_button()
    while not encontrado:
        if stop_signal['stop']:
            return
        base64_image = image_to_find.split(',')[1]
        help("dblclick",base64_image)
        screenshot = pyautogui.screenshot()
        encontrado,maxLoc = compare_image2(image_to_find,screenshot)
    if not sio.window.paused:
            sio.window.hide_continue_button()
            return True 


    (startX, startY) = maxLoc
    screenshot_np = np.array(screenshot)
    # Obtener el tamaño de la pantalla
    screen_width, screen_height = pyautogui.size()

    # Ajustar las coordenadas según la escala de la pantalla
    adjusted_x = (startX / screenshot_np.shape[1]) * screen_width
    adjusted_y = (startY / screenshot_np.shape[0]) * screen_height

    # Movimiento suave del mouse
    pyautogui.moveTo(x=adjusted_x + xp, y=adjusted_y + yp, duration=0.5)
    
    time.sleep(0.25)
    pyautogui.doubleClick(x=adjusted_x + xp, y=adjusted_y + yp)
    time.sleep(0.5)
    return True

def wait_exists(image_to_find, counter=-1):
    if stop_signal['stop']:
        return False
    sio.window.lookup()
    sio.window.text("Buscando...")
    base64_image = image_to_find.split(',')[1]
    screenshot = pyautogui.screenshot()
    encontrado, maxLoc = compare_image2(image_to_find, screenshot)
    while not encontrado:
        if counter == 0:
            return False
        if stop_signal['stop']:
            return False

        screenshot = pyautogui.screenshot()
        encontrado, maxLoc = compare_image2(image_to_find, screenshot)
        if encontrado:
            return True
        counter -= 1
        time.sleep(1)
    




def exists(image_to_find):
    return wait_exists(image_to_find,1)

def alttab(times):
    if stop_signal['stop'] :
        return
    window.press()
    import platform
    if platform.system() == 'Darwin':
        pyautogui.keyDown('command')
        for i in range(times):
            pyautogui.press(['tab'])
        pyautogui.keyUp('command')
    else:
        with pyautogui.hold('alt'):  
            for i in range(times):
                pyautogui.press(['tab'])
    time.sleep(0.25)
    time.sleep(0.5)

def copiar():
    if stop_signal['stop'] :
        return
    window.press()
    import platform
    if platform.system() == 'Darwin':
        with pyautogui.hold('command'):
            pyautogui.press(['c'])
    else:
        with pyautogui.hold('ctrl'):
            pyautogui.press(['c'])
    time.sleep(0.25)
    time.sleep(0.5)


def pegar():
    if stop_signal['stop'] :
        return
    window.press()
    import platform
    if platform.system() == 'Darwin':
        with pyautogui.hold('command'):
            pyautogui.press(['v'])
    else:
        with pyautogui.hold('ctrl'):
            pyautogui.press(['v'])
    time.sleep(0.25)
    time.sleep(0.5)



def copiarClip(output):
    if stop_signal['stop'] :
        return
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))
    time.sleep(0.5)


def leer(json_rectangulo):
    if stop_signal['stop']:
        return ""
    
    # Parsear el JSON string a un diccionario de Python
    rectangulo = json.loads(json_rectangulo)
    
    # Obtener los valores del JSON
    left = rectangulo['left']
    top = rectangulo['top']
    width = rectangulo['width']
    height = rectangulo['height']
    rwidth = rectangulo['rwidth']
    rheight = rectangulo['rheight']
    dimensiones_originales = rectangulo['dimensionesOriginales']
    original_width = dimensiones_originales['width']
    original_height = dimensiones_originales['height']
    # Obtener la resolución actual de la pantalla
    current_width, current_height = pyautogui.size()

    # Calcular las coordenadas escaladas
    left_escalado = left*(current_width/original_width)*(original_width/rwidth)
    top_escalado = top*(current_height/original_height)*(original_height/rheight)
    width_escalado = int(width * current_width / rwidth)
    height_escalado = int(height * current_height / rheight)
    pyautogui.moveTo(x=left_escalado , y=top_escalado , duration=0.5)
    pyautogui.moveTo(x=left_escalado+width_escalado , y=top_escalado+height_escalado , duration=0.5)

    time.sleep(0.5)
    sio.window.lookup()
    screenshot = pyautogui.screenshot()
    imagen_recortada = screenshot.crop((left_escalado, top_escalado, left_escalado + width_escalado, top_escalado + height_escalado))
    imagen_recortada = np.array(imagen_recortada, dtype=np.uint8)
    
    gray = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2GRAY)

    # Aplicar OCR utilizando Tesseract
    texto_extraido = pytesseract.image_to_string(gray, lang='spa')
    time.sleep(0.25)
    return texto_extraido

def escribir_texto(texto):

    if stop_signal['stop']:
        return
    teclear(texto)

def teclear(texto):
    texto = str(texto)
    if stop_signal['stop']:
        return
    sio.window.press()
    pyperclip.copy(texto)
    if platform.system() == "Darwin":
        pyautogui.hotkey("command", "v", interval=0.1)
    else:
        pyautogui.hotkey("ctrl", "v", interval=0.1)
    time.sleep(0.25)
    time.sleep(0.5)

def escribiren(img,texto):
    if stop_signal['stop'] :
        return
    window.press()
    click(img)
    escribir_texto(texto)
    time.sleep(0.25)
    time.sleep(0.5)

def type(text: str):
    
    if stop_signal['stop'] :
        return  
    pyperclip.copy(text)
    if platform.system() == "Darwin":
        pyautogui.hotkey("command", "v",interval=0.1)
    else:
        pyautogui.hotkey("ctrl", "v",interval=0.1)
    time.sleep(0.25)
    window.press()
    time.sleep(0.5)

def chatGPT(key, prompt, data):
    
    if stop_signal['stop']:
        return ""
    log("chatGPT")
    sio.window.lookup()
    from openai import OpenAI

    client = OpenAI(api_key=key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt + " DATA: " + data,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def agregar_a_excel(string, nombre_archivo):
    
    if stop_signal['stop']:
        return
    log("excel")
    import pandas as pd

    try:
        df = pd.read_excel("autom_files/" + nombre_archivo)
    except FileNotFoundError:
        df = pd.DataFrame()

    nueva_fila = pd.DataFrame({'Log': [string]})
    df = pd.concat([df, nueva_fila], ignore_index=True)
    df.to_excel("autom_files/" + nombre_archivo, index=False)

def email(to, subject, content):
    
    import smtplib
    import email.utils
    from email.mime.text import MIMEText
    log("email")
    msg = MIMEText(content)
    msg['To'] = email.utils.formataddr(('Recipient Name', to))
    msg['From'] = email.utils.formataddr(('Autom.cl', 'sender@autom.cl'))
    msg['Subject'] = subject
    server = smtplib.SMTP()
    server.connect('mail.autom.cl', 25)
    server.login('sender', 'wololo')
    server.set_debuglevel(True)
    try:
        server.sendmail('sender@autom.cl', [to], msg.as_string())
        sio.window.start()
    finally:
        server.quit()
