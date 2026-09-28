"""Edit settings.py, then run this file."""
from preprocessing import prepare_reconstruction_input
from settings import CONFIG
from _stage_io import read_stage, write_stage

def main():
    d = read_stage("05_alignment.npz")
    proj, angle, masks, support = prepare_reconstruction_input(d["projections"], d["angles"], d["masks"], CONFIG)
    noise = {k: d[k] for k in ("dark", "alpha", "sigma", "dark_fit_parameters", "dark_fit_values",
                              "dark_fit_resnorm", "commonline_angle", "commonline_scores")}
    write_stage("06_export.npz", noise, proj=proj, angle=angle, masks=masks,
                support=support, crop_centers=d["centers"])

if __name__ == "__main__":
    main()
