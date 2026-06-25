from PIL import Image
from pathlib import Path 
import glob

folder = Path("bigrun")
outname = "ReScalMovie10fps.gif"
fps = 10   # frames per second

pngs = sorted(folder.glob("DUN*.png"))

frames = []
for f in pngs:
    im = Image.open(f)

    # --- NEW CROP LINE ---
    # Crop rectangle: (left, top, right, bottom)
    # Upper-left 1000 × 600 → from (0,0) to (1000,600)
    im = im.crop((0, 0, 700, 800))

    frames.append(im)

duration = int(1000 / fps)  # ms per frame

frames[0].save(
    outname,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
)

print(f"Saved {outname} with {len(frames)} frames.")

