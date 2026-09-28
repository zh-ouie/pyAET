"""Edit settings.py, then run this file."""
from preprocessing import denoise_projections
from settings import CONFIG
from _stage_io import read_stage, write_stage

def main():
    d = read_stage("02_noise.npz")
    projections = denoise_projections(d.pop("summed"), d["alpha"], d["sigma"], backend=CONFIG.bm3d_backend)
    write_stage("03_denoising.npz", d, projections=projections)

if __name__ == "__main__":
    main()
