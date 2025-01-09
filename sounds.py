import numpy as np
import pygame
import random
def generate_r2d2_help_sound(duration=1, sample_rate=44100):
    """
    Genera un sonido similar a los efectos de R2-D2 usando frecuencias.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sound_wave = np.zeros_like(t)

     # Crear una mezcla de frecuencias para el sonido tipo R2-D2
# Definir las frecuencias
    frequencies = [
        (200, 0.2),  # "H"
        (200, 0.2),  # "e"
        (200, 0.2),  # "l"
        (random.randint(500, 1500), 0.2),  # "l"
        (150, 0.2),  # "o"
    ]
    
    
    start_idx = 0
    for freq, duration_part in frequencies:
        segment_length = int(duration_part * sample_rate)
        end_idx = start_idx + segment_length
        if end_idx > len(t):
            end_idx = len(t)
        freq_variation = freq + 50 * np.sin(2 * np.pi * 2 * t[start_idx:end_idx])  # Variación de frecuencia
        sound_wave[start_idx:end_idx] = np.sin(2 * np.pi * freq_variation * t[start_idx:end_idx])
        start_idx = end_idx


    # Normalizar el sonido
    sound_wave = (sound_wave / np.max(np.abs(sound_wave)) * 32767).astype(np.int16)

    # Crear un array estéreo duplicando el canal mono en dos canales
    stereo_sound_wave = np.stack((sound_wave, sound_wave), axis=-1)

    sound = pygame.sndarray.make_sound(stereo_sound_wave)

    return sound

def generate_r2d2_hello_sound(duration=1, sample_rate=44100):
    """
    Genera un sonido similar a los efectos de R2-D2 usando frecuencias.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sound_wave = np.zeros_like(t)

    # Crear una mezcla de frecuencias para el sonido tipo R2-D2
# Definir las frecuencias
    frequencies = [
        (500, 0.2),  # "H"
        (500, 0.2),  # "e"
        (300, 0.2),  # "l"
        (random.randint(500, 1500), 0.2),  # "l"
        (250, 0.2),  # "o"
    ]
    
    start_idx = 0
    for freq, duration_part in frequencies:
        segment_length = int(duration_part * sample_rate)
        end_idx = start_idx + segment_length
        if end_idx > len(t):
            end_idx = len(t)
        freq_variation = freq + 200 * np.sin(2 * np.pi * 2 * t[start_idx:end_idx])  # Variación de frecuencia
        sound_wave[start_idx:end_idx] = np.sin(2 * np.pi * freq_variation * t[start_idx:end_idx])
        start_idx = end_idx
    

    # Normalizar el sonido
    sound_wave = (sound_wave / np.max(np.abs(sound_wave)) * 32767).astype(np.int16)

    # Crear un array estéreo duplicando el canal mono en dos canales
    stereo_sound_wave = np.stack((sound_wave, sound_wave), axis=-1)

    sound = pygame.sndarray.make_sound(stereo_sound_wave)

    return sound

def generate_r2d2_yeah_sound(duration=1, sample_rate=44100):
    """
    Genera un sonido similar a los efectos de R2-D2 usando frecuencias.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sound_wave = np.zeros_like(t)

    # Crear una mezcla de frecuencias para el sonido tipo R2-D2
# Definir las frecuencias
    frequencies = [
        (2000, 0.2),  # "H"
        (2000, 0.2),  # "e"
        (2000, 0.2),  # "l"
        (random.randint(1000, 1500), 0.2),  # "l"
        (2250, 0.2),  # "o"
    ]
    
    start_idx = 0
    for freq, duration_part in frequencies:
        segment_length = int(duration_part * sample_rate)
        end_idx = start_idx + segment_length
        if end_idx > len(t):
            end_idx = len(t)
        freq_variation = freq + 200 * np.sin(2 * np.pi * 2 * t[start_idx:end_idx])  # Variación de frecuencia
        sound_wave[start_idx:end_idx] = np.sin(2 * np.pi * freq_variation * t[start_idx:end_idx])
        start_idx = end_idx
    

    # Normalizar el sonido
    sound_wave = (sound_wave / np.max(np.abs(sound_wave)) * 32767).astype(np.int16)

    # Crear un array estéreo duplicando el canal mono en dos canales
    stereo_sound_wave = np.stack((sound_wave, sound_wave), axis=-1)

    sound = pygame.sndarray.make_sound(stereo_sound_wave)

    return sound