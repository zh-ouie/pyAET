"""Edit settings.py, then run this file."""
from preprocessing import normalize_and_align
from settings import CONFIG
from _stage_io import read_stage, write_stage

def main():
    d = read_stage("04_background.npz")
    r = normalize_and_align(d["projections"], angle_range=CONFIG.commonline_range,
                            apply_rotation=CONFIG.apply_commonline_rotation,
                            workers=CONFIG.commonline_workers)
    write_stage("05_alignment.npz", d, projections=r.projections,
                commonline_angle=r.commonline_angle, commonline_scores=r.commonline_scores,
                rotation_applied=r.rotation_applied)

if __name__ == "__main__":
    main()
