"""Edit settings.py, then run this file."""
from preprocessing import subtract_background
from settings import CONFIG
from _stage_io import read_stage, write_stage

def main():
    d = read_stage("03_denoising.npz")
    r = subtract_background(d["projections"], disk=CONFIG.background_disk, otsu_scale=CONFIG.otsu_scale)
    write_stage("04_background.npz", d, projections=r.projections, masks=r.masks,
                original_masks=r.original_masks, initial_original_masks=r.initial_original_masks,
                initial_shifts=r.initial_shifts, final_shifts=r.final_shifts)

if __name__ == "__main__":
    main()
