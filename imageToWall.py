from PIL import Image, ImageSequence
import serial
import time


IMAGEWIDTH = 192
IMAGEHIGHT = 112

# Pfad zur GIF-Datei
gif_path = "/home/neura/Desktop/50step01.gif"

# Serielle Verbindung einrichten

time.sleep(2)  # Warte auf die Initialisierung der Verbindung

# GIF-Datei öffnen
#img = Image.open(gif_path)

# Pin-Nummern entsprechend der Reihenfolge von unten nach oben
pins = [14, 12, 13, 15, 2, 4, 16, 17, 5, 18, 19, 21, 22, 23]  # Von unten nach oben

# Funktion zum Senden von Bild-Daten an ESP32
def send_image_to_esp32(image, pin_index):
    ser = serial.Serial('/dev/ttyUSB0', 115200)
    data = image.tobytes()
    ser.write(f"{pin_index},{len(data)}\n".encode())
    time.sleep(0.1)  # Kurze Pause
    ser.write(data)
    time.sleep(0.1)  # Warten auf Verarbeitung
    ser.close()

# Verarbeitung jedes Frames der GIF-Datei
#for frame in ImageSequence.Iterator(img):
#    frame = frame.convert("RGB")  # Konvertieren, falls nicht bereits RGB
#    frame = frame.resize((192, 112), Image.ANTIALIAS)  # Skalieren auf Matrixgröße
#
#    # Teile das Bild und sende es, beginnend mit dem untersten Streifen
#    for i, pin in enumerate(pins):
#        strip = frame.crop((0, 112 - 8 * (i + 1), 192, 112 - 8 * i))
#        send_image_to_esp32(strip, i)
#    
#    time.sleep(0.5)  # Wartezeit zwischen den Frames, anpassbar je nach gewünschter Abspielgeschwindigkeit

# Serielle Verbindung schließen
#

def pipline_entry(image):
    #Image is 640x480x4 (rgba)
    frame = image
    for i, pin in enumerate(pins):
        strip = frame.crop((0, IMAGEWIDTH - 8 * (i + 1), IMAGEWIDTH, IMAGEHIGHT - 8 * i))
        send_image_to_esp32(strip, i)
    return True

if __name__ == "__main__":
    pass