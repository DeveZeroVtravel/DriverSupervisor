import tkinter as tk
from PIL import Image, ImageTk

# Create the main window
root = tk.Tk()
root.title("Transparent PNG Overlay")
root.geometry("1024x600")

# Load the PNG image with transparency
image = Image.open("DriverSupervisor/BG.png")  # Replace with your PNG file
image = image.resize((1024, 600), Image.Resampling.LANCZOS)  # Resize if needed
tk_image = ImageTk.PhotoImage(image)

# Create a Canvas
canvas = tk.Canvas(root, width=1024, height=600)
canvas.pack()

# Display the transparent image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

root.mainloop()
