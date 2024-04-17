import os
import json
import torch
#import pyautogui
import torchvision

from PIL import Image

with open("config.json") as file:
    config = json.load(file)
    IMAGE_DIR = config["ImageDir"]
    IMAGE_NAME = config["ImageName"]
    RAWDATA = config["RawData"]
    DATAPATH = config["DataPath"]

TEST_IMAGE_PATH = IMAGE_DIR + IMAGE_NAME

SIMULATIONMODE = True
IMAGETRANSFORMER = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28,28))])
#
#def take_screenshot(filename):
#    try:
#        # Screenshot aufnehmen
#        screenshot = pyautogui.screenshot()
#
#        # Bild speichern
#        screenshot.save(filename)
#        print("Screenshot erfolgreich gespeichert als:", filename)
#    except Exception as e:
#        print("Fehler beim Speichern des Screenshots:", e)
#
def invert_image(image):
    invertedImage = 1-image
    imgWhithoutNoise = torch.nn.functional.relu(invertedImage-0.01)
    imgWhithoutNoise[imgWhithoutNoise!=0] = 1
    return imgWhithoutNoise

def load_image_from_path(image_path):
    image = Image.open(image_path).convert("L")    
    img_normalized = IMAGETRANSFORMER(image).float()
    img_inverted = invert_image(img_normalized)
    #img_normalized = img_normalized.to(DEVICE)
    return img_inverted

def load_image():
    if SIMULATIONMODE:
        return load_image_from_path (image_path=TEST_IMAGE_PATH)

    pass

if __name__ == "__main__":
    filename = input("this.png")
    #take_screenshot(filename)