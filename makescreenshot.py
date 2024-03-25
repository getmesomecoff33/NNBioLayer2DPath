import pyautogui

def take_screenshot(filename):
    try:
        # Screenshot aufnehmen
        screenshot = pyautogui.screenshot()

        # Bild speichern
        screenshot.save(filename)
        print("Screenshot erfolgreich gespeichert als:", filename)
    except Exception as e:
        print("Fehler beim Speichern des Screenshots:", e)

if __name__ == "__main__":
    filename = input("this.png")
    take_screenshot(filename)