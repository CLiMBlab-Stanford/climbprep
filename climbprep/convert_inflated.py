#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path

def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def find_left_pial_dir(out_root: Path, sub: str) -> Path:
    """
    Find the directory that contains {SUB}_hemi-L_pial.surf.gii.
    Returns the parent directory (DST).
    """
    pattern = f"{sub}*hemi-L_pial.surf.gii"
    # Search within .../derivatives/preprocess/main/<sub>/**/anat/
    search_root = out_root / sub
    if not search_root.exists():
        fail(f"Subject folder not found: {search_root}")

    matches = list(search_root.rglob(pattern))
    if not matches:
        fail(
            f"Could not find '{pattern}' under {search_root}.\n"
            "Make sure fMRIPrep completed surfaces for this subject.\n"
            "Searched recursively for: **/anat/" + pattern
        )
    # If multiple found, take the first (usually ses-1/anat)
    return matches[0].parent

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert FreeSurfer lh.inflated to GIFTI in the same dir as *_hemi-L_pial.surf.gii."
    )
    ap.add_argument("project", help='Project name (e.g., "moth")')
    ap.add_argument("subject", help='Subject code (e.g., "sub-UTS02")')
    ap.add_argument("-i", "--fmriprep-image", default="/juice6/u/nlp/climblab/apptainer/images/fmriprep-25.2.2.simg")
    ap.add_argument("-f", "--fs-license", default="/juice6/u/nlp/climblab/freesurfer/license.txt")
    ap.add_argument("-r", "--bids-root", default="/juice6/u/nlp/climblab/BIDS")

    args = ap.parse_args(argv)

    project = args.project
    sub = args.subject

    img = Path(args.fmriprep_image)
    if not img.exists():
        fail(f"Singularity image not found: {img}\n"
             "Set FMRIPREP_IMG env var if your image lives elsewhere.")

    fs_license = Path(args.fs_license)
    if not fs_license.exists():
        fail(f"FreeSurfer license not found: {fs_license}\n"
             "Set FS_LICENSE env var to your license file path.")

    bids_root = Path(args.bids_root)
    out_root = bids_root / project / "derivatives" / "preprocess" / "main"

    if not out_root.exists():
        fail(f"Output root not found: {out_root}\n"
             "Set BIDS_ROOT env var if your BIDS root differs.")

    # 1) Determine DST from the existing pial GIFTI
    dst_dir = find_left_pial_dir(out_root, sub)

    #convert L and R hemisphere
    for hemi_codes in [['lh','L'],['rh','R']]:
        # 2) Build the path to the FS inflated surface produced by fMRIPrep
        matches = list((out_root / "sourcedata" / "freesurfer" / sub / "surf").rglob("**/" + hemi_codes[0] + ".inflated"))
        h_inflated = matches[0]
        
        if not h_inflated.exists():
            fail(f"Missing FreeSurfer lh.inflated: {h_inflated}\n"
                 "Did fMRIPrep run FreeSurfer (no --fs-no-reconall)?")
    
        # 3) Output filename in the same directory as the left pial GIFTI
        out_gii = dst_dir / f"{sub}_hemi-{hemi_codes[1]}_inflated.surf.gii"
    
        # 4) Prepare Singularity exec call (mirrors your bash)
        # Bind the OUT root, destination directory, and the license directory
        binds = ",".join({
            str(out_root),
            str(dst_dir),
            str(fs_license.parent),
        })
        env = os.environ.copy()
        # Export FS_LICENSE into container via SINGULARITYENV_*
        env["SINGULARITYENV_FS_LICENSE"] = str(fs_license)
    
        cmd = [
            "singularity", "exec", "--cleanenv",
            "-B", binds,
            str(img),
            "mris_convert",
            str(h_inflated),
            str(out_gii),
        ]
    
        print("Running:", " ".join(cmd))
        try:
            subprocess.check_call(cmd, env=env)
        except subprocess.CalledProcessError as e:
            fail(f"mris_convert failed with exit code {e.returncode}")
    
        if out_gii.exists():
            print(f"Done: {out_gii}")
        else:
            fail("mris_convert reported success but output file not found.")

if __name__ == "__main__":
    main()
