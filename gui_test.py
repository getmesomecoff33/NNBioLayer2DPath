import tkinter as tk

from PIL import Image, ImageDraw

from net import eval_image
from net  import render_image_path
from net import yield_training_loop
from makescreenshot import load_image


class PaintApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.modelEngine = yield_training_loop()
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
        save_button = tk.Button(self, text="Bild speichern", command=self.save_image)
        save_button.pack(side=tk.LEFT, padx=10)

        # Button to clear the canvas
        clear_button = tk.Button(self, text="Leinwand löschen", command=self.clear_canvas)
        clear_button.pack(side=tk.RIGHT, padx=10)

        self.image = Image.new("RGB", (800, 500), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black", width=5)
        self.last_x, self.last_y = x, y



    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (800, 500), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def save_image(self):
        # Logic goes Here
        step = 1
        correct = False # Set to buttom press
        image = self.image
        if self.model == None:
            return
        if self.classes == None:
            return
        image_nomalized = load_image(image)
        predictedLabel, certainty = eval_image(self.model, img_normalized=image_nomalized)
        image = render_image_path(self.model,image_nomalized, step)
        step += 1
        print(self.classes[predictedLabel])
        if not  correct:
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
