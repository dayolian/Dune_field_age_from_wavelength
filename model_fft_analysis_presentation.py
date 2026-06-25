import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import csv
import re

# ----------------- USER SETTINGS -----------------
image_folder = Path("testminifft")   # folder with PNGs
output_folder = Path("testminifft")   # where to save figures
pixel_size = 1.0                      # model units per pixel
nbins_radial = 300                     # bins for wavelength spectrum (log-spaced)
nbins_theta = 72                      # bins for orientation spectrum
crop_width = 600                      # pixels (x)
crop_height = 400                     # pixels (y)
csv_output = "dominant_wavelengths_tight.csv"
# -------------------------------------------------


def load_and_crop(path):
    """
    Load PNG as grayscale float array and crop to upper-left
    crop_height x crop_width region.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64)
    #cropped = arr[0:crop_height, 0:crop_width]
    cropped = arr
    return cropped


def fft_power_2d(img):
    """
    Compute 2D power spectrum |FFT(img)|^2 with Hann window and mean removed.
    """
    img = img - np.mean(img)
    ny, nx = img.shape

    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    imgw = img * wy * wx

    F = np.fft.fft2(imgw)
    F = np.fft.fftshift(F)
    P = (F * F.conj()).real
    return P


def radial_spectrum(P, pixel_size=1.0, nbins=50):
    """
    Radially average 2D power spectrum into 1D power vs wavelength.

    Radial bins are logarithmically spaced in wavenumber k, which corresponds
    to logarithmically spaced bins in wavelength lambda = 1/k.
    """
    ny, nx = P.shape

    ky = np.fft.fftfreq(ny, d=pixel_size)
    kx = np.fft.fftfreq(nx, d=pixel_size)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)

    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2).ravel()
    P_flat = P.ravel()

    # Exclude k=0 (infinite wavelength)
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

    # Convert to wavelength: lambda = 1/k
    wavelength = 1.0 / k_centers  # same units as pixel_size
    idx = np.argsort(wavelength)
    return wavelength[idx], power_1d[idx]


def angular_spectrum(P, nbins=72):
    """
    Angular spectrum: average power as a function of orientation (0–180°).
    Returns theta_centers (degrees) and power_theta.
    """
    ny, nx = P.shape

    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)

    KX, KY = np.meshgrid(kx, ky)

    theta = np.degrees(np.arctan2(KY, KX))
    theta = np.mod(theta, 180.0).ravel()
    P_flat = P.ravel()

    # Exclude the DC point
    mask = (KX != 0) | (KY != 0)
    theta = theta[mask.ravel()]
    P_flat = P_flat[mask.ravel()]

    bins = np.linspace(0.0, 180.0, nbins + 1)
    theta_centers = 0.5 * (bins[:-1] + bins[1:])
    which_bin = np.digitize(theta, bins) - 1

    power_theta = np.zeros(nbins)
    for i in range(nbins):
        m = which_bin == i
        if np.any(m):
            power_theta[i] = P_flat[m].mean()
        else:
            power_theta[i] = np.nan

    return theta_centers, power_theta


def parse_time_from_name(name):
    """
    Extract model time (in thousands of t0) from filename like 'DUN00010_t0.png'.
    Returns int or None if not found.
    """
    m = re.search(r"DUN(\d+)_t0", name)
    if m:
        return int(m.group(1))
    return None


def make_summary_figure(
    img,
    P,
    wavelength,
    power_radial,
    theta_deg,
    power_theta,
    filename,
    time_k,
    outpath,
    fft_vmin,
    fft_vmax,
    rad_ylim,
    ang_ylim,
):
    """
    Make 3-panel figure and save to outpath, using fixed axis limits.
    Panels (left to right): image, 2D FFT, radial spectrum.
    Note: theta_deg, power_theta, ang_ylim are unused (kept for API compatibility).
    """
    fig, (ax_img, ax_fft, ax_rad) = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Cropped image ---
    ax_img.imshow(img, cmap="gray", origin="upper")
    ax_img.set_title("Cropped model image")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # --- 2D FFT (log power) with fixed color scale ---
    P_log = np.log10(P + 1.0)
    im = ax_fft.imshow(
        P_log,
        cmap="inferno",
        origin="lower",
        vmin=fft_vmin,
        vmax=fft_vmax,
    )
    ax_fft.set_title("2D FFT power (log10)")
    ax_fft.set_xticks([])
    ax_fft.set_yticks([])
    fig.colorbar(im, ax=ax_fft, fraction=0.046, pad=0.04)

    # --- Wavelength vs power (radial spectrum) ---
    ax_rad.loglog(wavelength, power_radial, marker="o")
    ax_rad.set_xlabel("Wavelength (model units)")
    ax_rad.set_ylabel("Power")
    ax_rad.set_title("Radial spectrum (power vs wavelength)")
    ax_rad.grid(True, which="both", ls=":")
    ax_rad.set_xlim(wavelength.min(), wavelength.max())
    ax_rad.set_ylim(rad_ylim[0], rad_ylim[1])

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)



def main():
    if not image_folder.exists():
        raise RuntimeError(f"Image folder not found: {image_folder}")

    files = sorted(image_folder.glob("*.png"))
    if not files:
        raise RuntimeError(f"No PNG files found in {image_folder}")

    n = len(files)

    # Storage for per-image results
    times_k = []
    dom_lambdas = []
    radial_spectra = []
    angular_spectra = []

    wavelength_ref = None
    theta_ref = None

    # Global ranges for consistent axes
    fft_log_min = np.inf
    fft_log_max = -np.inf
    rad_min_pos = np.inf   # minimum positive radial power
    rad_max = -np.inf
    ang_min = np.inf
    ang_max = -np.inf

    # ---------- PASS 1: compute spectra + global ranges ----------
    for i, path in enumerate(files, start=1):
        print(f"[Pass 1] Processing {i}/{n}: {path.name}")

        img = load_and_crop(path)
        P = fft_power_2d(img)

        # FFT log range
        P_log = np.log10(P + 1.0)
        valid_fft = np.isfinite(P_log)
        if np.any(valid_fft):
            fft_log_min = min(fft_log_min, P_log[valid_fft].min())
            fft_log_max = max(fft_log_max, P_log[valid_fft].max())

        # Radial spectrum
        wavelength, power_radial = radial_spectrum(P, pixel_size=pixel_size,
                                                   nbins=nbins_radial)
        radial_spectra.append(power_radial)
        if wavelength_ref is None:
            wavelength_ref = wavelength

        # Radial power range (positive only for log y-axis)
        valid_rad = np.isfinite(power_radial) & (power_radial > 0)
        if np.any(valid_rad):
            rad_min_pos = min(rad_min_pos, power_radial[valid_rad].min())
            rad_max = max(rad_max, power_radial[valid_rad].max())

        # Angular spectrum
        theta_deg, power_theta = angular_spectrum(P, nbins=nbins_theta)
        angular_spectra.append(power_theta)
        if theta_ref is None:
            theta_ref = theta_deg

        valid_ang = np.isfinite(power_theta)
        if np.any(valid_ang):
            ang_min = min(ang_min, power_theta[valid_ang].min())
            ang_max = max(ang_max, power_theta[valid_ang].max())

        # Dominant wavelength
        if np.all(np.isnan(power_radial)) or not np.any(valid_rad):
            dom_lambda = np.nan
        else:
            dom_idx = np.nanargmax(power_radial)
            dom_lambda = wavelength[dom_idx]
        dom_lambdas.append(dom_lambda)

        # Time from filename
        time_k = parse_time_from_name(path.name)
        times_k.append(time_k)

    # Clean up ranges
    if not np.isfinite(fft_log_min) or not np.isfinite(fft_log_max):
        fft_log_min, fft_log_max = 0.0, 1.0  # fallback

    if not np.isfinite(rad_min_pos) or not np.isfinite(rad_max):
        rad_min_pos, rad_max = 1e-6, 1.0  # fallback

    if not np.isfinite(ang_min) or not np.isfinite(ang_max):
        ang_min, ang_max = 0.0, 1.0  # fallback

    rad_ylim = (rad_min_pos, rad_max)
    ang_ylim = (ang_min, ang_max)

    print("\nGlobal axis ranges:")
    print(f"  FFT log10 power: [{fft_log_min:.2f}, {fft_log_max:.2f}]")
    print(f"  Radial power (log y): [{rad_ylim[0]:.3e}, {rad_ylim[1]:.3e}]")
    print(f"  Angular power (linear y): [{ang_ylim[0]:.3e}, {ang_ylim[1]:.3e}]\n")

    # ---------- PASS 2: regenerate figures with fixed axes ----------
    for i, path in enumerate(files, start=1):
        print(f"[Pass 2] Making figure {i}/{n}: {path.name}")

        img = load_and_crop(path)
        P = fft_power_2d(img)

        power_radial = radial_spectra[i - 1]
        power_theta = angular_spectra[i - 1]
        time_k = times_k[i - 1]

        outfig = output_folder / f"summary_{path.stem}.png"
        make_summary_figure(
            img,
            P,
            wavelength_ref,
            power_radial,
            theta_ref,
            power_theta,
            path.name,
            time_k,
            outfig,
            fft_vmin=fft_log_min,
            fft_vmax=fft_log_max,
            rad_ylim=rad_ylim,
            ang_ylim=ang_ylim,
        )

    # ---------- CSV output ----------
    rows = []
    for path, time_k, dom_lambda in zip(files, times_k, dom_lambdas):
        rows.append([path.name, time_k, dom_lambda])

    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "time_k", "dominant_wavelength"])
        writer.writerows(rows)

    print(f"\nSaved {len(files)} summary figures to {output_folder}")
    print(f"Saved CSV: {csv_output}")


if __name__ == "__main__":
    main()

