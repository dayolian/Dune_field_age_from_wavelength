import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# === USER SETTINGS ===
folder = Path("tiles")  # <-- change this
pixel_size = 30.0  # meters per pixel (or whatever units you want)
nbins = 100        # number of radial bins for 1D spectrum

# Optional: which file extensions to use
extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def load_image_as_float(path):
    """Load image and return as float64 2D array (grayscale)."""
    img = Image.open(path).convert("L")  # convert to grayscale
    arr = np.asarray(img, dtype=np.float64)
    return arr


def power_spectrum_2d(img):
    """
    Compute 2D power spectrum |FFT(img)|^2.
    img is 2D array (float).
    """
    img = img - np.mean(img)  # remove DC/mean
    # Optional: Hann window to reduce edge effects
    ny, nx = img.shape
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    window = wy * wx
    imgw = img * window

    F = np.fft.fft2(imgw)
    F = np.fft.fftshift(F)
    P = (F * F.conj()).real  # |F|^2
    return P


def radial_average(P, pixel_size=1.0, nbins=60):
    """
    Radially average 2D power spectrum P(kx,ky) into 1D power(k).
    Returns wavelength (same units as pixel_size) and power_1d.
    """
    ny, nx = P.shape

    # Spatial frequencies in cycles per unit distance
    ky = np.fft.fftfreq(ny, d=pixel_size)
    kx = np.fft.fftfreq(nx, d=pixel_size)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)

    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2).ravel()
    P_flat = P.ravel()

    # Ignore k=0 (infinite wavelength)
    mask = k > 0
    k = k[mask]
    P_flat = P_flat[mask]

    # Log-spaced bins in k
    kmin, kmax = k.min(), k.max()
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    which_bin = np.digitize(k, bins) - 1
    power_1d = np.zeros(nbins)

    for i in range(nbins):
        m = which_bin == i
        if np.any(m):
            power_1d[i] = P_flat[m].mean()
        else:
            power_1d[i] = np.nan

    # Convert wavenumber to wavelength: lambda = 1 / k
    wavelength = 1.0 / k_centers  # same units as pixel_size

    # Sort by wavelength ascending
    idx = np.argsort(wavelength)
    return wavelength[idx], power_1d[idx]


def main():
    # --- Collect files ---
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    if not files:
        raise RuntimeError(f"No image files found in {folder}")

    # --- First pass: find minimum common shape ---
    ny_min, nx_min = None, None
    for path in sorted(files):
        img = load_image_as_float(path)
        ny, nx = img.shape
        if ny_min is None:
            ny_min, nx_min = ny, nx
        else:
            ny_min = min(ny_min, ny)
            nx_min = min(nx_min, nx)

    print(f"Using common cropped FFT size: ({ny_min}, {nx_min})")

    # --- Second pass: compute FFTs, crop, and accumulate ---
    global_P = np.zeros((ny_min, nx_min), dtype=float)

    for i, path in enumerate(sorted(files)):
        img = load_image_as_float(path)
        P = power_spectrum_2d(img)   # full FFT power

        ny, nx = P.shape
        # center-crop to (ny_min, nx_min)
        y0 = (ny - ny_min) // 2
        x0 = (nx - nx_min) // 2
        P_crop = P[y0:y0+ny_min, x0:x0+nx_min]

        global_P += P_crop

        print(f"Processed {i+1}/{len(files)}: {path.name}")

    # --- Radial average to get 1D spectrum (power vs wavelength) ---
    wavelength, power_1d = radial_average(global_P, pixel_size=pixel_size, nbins=nbins)

    # --- Plot 2D FFT and 1D spectrum together ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 2D FFT image
    P_log = np.log10(global_P + 1)
    im = ax1.imshow(P_log, cmap="inferno", origin='lower')
    ax1.set_title("Summed 2D FFT Power Spectrum (log scale)")
    ax1.set_xlabel("kx")
    ax1.set_ylabel("ky")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="log10 Power")

    # 1D power vs wavelength
    ax2.loglog(wavelength, power_1d, marker="o")
    ax2.set_xlabel("Wavelength (units of pixel_size)")
    ax2.set_ylabel("Power")
    ax2.set_title("Global Power vs Wavelength")
    ax2.grid(True, which='both', ls=':')
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

