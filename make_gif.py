from pathlib import Path
from PIL import Image

folder = Path("fft_summary_tight_presentation")
outname = "modelFFT_presentation.gif"
fps = 5   # frames per second

files = sorted(folder.glob("*.png"))

frames = [Image.open(f) for f in files]

duration = int(1000 / fps)  # ms per frame

frames[0].save(
    outname,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
)

print(f"Saved {outname} with {len(frames)} frames.")

