import tkinter as tk
import threading

from PIL import Image, ImageDraw

from net import eval_image
from net  import render_image_path
from net import yield_training_loop
from makescreenshot import load_image

step = 1
reward_correct = 1.00001
reward_wrong = 0.999

class PaintApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.lamb = 0.0005
        self.modelEngine = yield_training_loop(self.lamb)
        self.model = None
        self.classes = None

        self.title("Paint App")
        self.geometry("800x600")

        # Set up canvas
        self.canvas = tk.Canvas(self, bg="white", width=800, height=500)
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

        # Button to save the image
        save_button = tk.Button(self, text="What is that?", command=self.start_thread)
        save_button.pack(side=tk.LEFT, padx=10)

        # Button to clear the canvas
        clear_button = tk.Button(self, text="Clear Pad", command=self.clear_canvas)
        clear_button.pack(side=tk.RIGHT, padx=10)

        self.image = Image.new("RGB", (800, 500), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", width=11)
            self.draw.rectangle([(x-5, y-5),(x+5, y+5)],fill="black")
        self.last_x, self.last_y = x, y

    def start_thread(self):
        threading.Thread(target=self.save_image).start()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (800, 500), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def save_image(self):
        # Logic goes Here
        global step
        image = self.image.resize((28,28))
        if self.model == None:
            return
        if self.classes == None:
            return
        image_nomalized = load_image(image)
        image.save('./results/mnist/input.png',"PNG")
        predictedLabel, certainty = eval_image(self.model, img_normalized=image_nomalized)
        render_image_path(self.model,image_nomalized, step)
        step += 1
        if certainty >1:
            certainty = 1
        print("I'am {} percent sure this is a {}".format(certainty*100,self.classes[predictedLabel]))
        #TODO
        #Wait for button input
        #Button Rewards
        #correct = is_correct button pressed
        correct = False # Set to buttom press
        if not  correct:
            self.lamb = self.lamb*reward_wrong
            self.modelEngine.send(self.lamb)
            outputTuple = next(self.modelEngine)
            self.model = outputTuple[0]
            self.classes = outputTuple[1]
        

def after_loo(app):
    outputTuple = next(app.modelEngine)
    app.model = outputTuple[0]
    app.classes = outputTuple[1]
    print("Hi")


if __name__ == "__main__":
    app = PaintApp()
    # Required for Startup
    app.after(1000, after_loo, app)
    app.mainloop()
