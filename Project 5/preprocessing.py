import os
from PIL import Image

folder = "assets"

for fname in os.listdir(folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, fname)

        img = Image.open(path)
        img = img.convert("L")
        img = img.resize((512, 512))

        img.save(path)
