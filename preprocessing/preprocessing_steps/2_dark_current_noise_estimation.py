"""Edit settings.py, then run this file."""
from preprocessing import estimate_projection_noise
from _stage_io import read_stage, write_stage

def main():
    d = read_stage("01_acquisition.npz")
    n = estimate_projection_noise(d.pop("frames"), d["angles"])
    write_stage("02_noise.npz", d, summed=n.summed, dark=n.dark, alpha=n.alpha,
                sigma=n.sigma, dark_fit_parameters=n.dark_fit["parameters"],
                dark_fit_values=n.dark_fit["fitted"], dark_fit_resnorm=n.dark_fit["resnorm"])

if __name__ == "__main__":
    main()
