"""Edit settings.py, then run this file."""
from preprocessing import load_emd, prepare_acquisition
from settings import INPUT_FILE, CENTERS_FILE, CONFIG
from _stage_io import write_stage
import numpy as np

def main():
    raw, angles = load_emd(INPUT_FILE)
    centers = np.load(CENTERS_FILE, allow_pickle=False) if CENTERS_FILE else None
    result = prepare_acquisition(raw, angles, CONFIG, centers)
    write_stage("01_acquisition.npz", {}, frames=result.frames,
                projections=result.projections, angles=result.angles, centers=result.centers)

if __name__ == "__main__":
    main()
