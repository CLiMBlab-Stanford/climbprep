#!/usr/bin/env python3
import os
import glob
import re
import sys
import json
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.resources import files, as_file

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import climbprep.convert_inflated

from nilearn import surface

from climbprep.constants import *
from climbprep.util import *

# ------------------------------- container helpers --------------------------------

def stderr(msg: str):
    sys.stderr.write(msg)

def sh(cmd: str):
    stderr(cmd + "\n\n")
    rc = os.system(cmd)
    assert rc == 0, f"Command failed (exit {rc}): {cmd}"

def _shlex(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"

def _bind_flags(binds: str) -> str:
    flags = ""
    if binds:
        for b in re.split(r'[,\s]+', binds.strip()):
            if b:
                flags += f" -B {b}"
    return flags

def cx(container: str, home_dir: str, binds: str, inner: str):
    """Run a bash snippet inside the QuNex container with clean env & writable HOME."""
    cmd = f"singularity exec --cleanenv -H {_shlex(home_dir)}{_bind_flags(binds)} {_shlex(container)} bash -lc {_shlex(inner)}"
    sh(cmd)

def cx_out(container: str, home_dir: str, binds: str, inner: str) -> str:
    """Run inside container and capture stdout (raises on nonzero)."""
    cmd = ["singularity", "exec", "--cleanenv", "-H", home_dir]
    cmd += _bind_flags(binds).split()
    cmd += [container, "bash", "-lc", inner]
    cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return cp.stdout

# -----------------------------------------------------------------------------------

def same_tr(files):
    trs = []
    for f in files:
        img = nib.load(f)
        z = img.header.get_zooms()
        if len(z) < 4:
            raise AssertionError(f"Not a 4D NIfTI: {f}")
        trs.append(float(z[3]))
    u = sorted(set(round(x, 6) for x in trs))
    return (len(u) == 1, float(u[0]) if u else None, u, trs)

def add_surfaces_to_spec(spec_path, out_root, anat_path, participant, ses_str_anat, container, home_dir, binds):
    """Copy GIFTI surfaces from fMRIPrep anat and add to spec."""
    for surf in ('pial', 'white', 'midthickness', 'inflated'):
        for hemi in ('L', 'R'):
            suffix = '.shape.gii' if surf == 'sulc' else '.surf.gii'
            file = list(Path(anat_path).glob(f'sub-{participant}*_hemi-{hemi}_{surf}{suffix}'))[0]
            src = str(file)
            print(src)
            #src = os.path.join(anat_path, f'sub-{participant}{ses_str_anat}_hemi-{hemi}_{surf}{suffix}')
            
            dst = os.path.join(out_root, os.path.basename(src))
            shutil.copy(src, dst)
            
            #copy surface files to a generic name so that we can have a generic scene template file
            shutil.copy(src, out_root + '/' + f'hemi-{hemi}_{surf}{suffix}')
                        
            which_hemi = 'LEFT' if hemi == 'L' else 'RIGHT'
            cx(container, home_dir, binds,
               f"wb_command -add-to-spec-file {_shlex(spec_path)} CORTEX_{which_hemi} {_shlex(dst)}")

def map_ics_to_surface_and_merge_dscalar(ic_4d_path, out_root, anat_path, participant, ses_str_anat,
                                         container, home_dir, binds):
    """
    Map a 4D MELODIC IC volume to L/R surfaces in one call per hemisphere (no splitting),
    smooth within cortical ROIs, create a multi-map dscalar, and set map names.
    """
    def _q(s):  # mini shell-escape
        return "'" + s.replace("'", "'\\''") + "'"

    # Surfaces / ROIs from fMRIPrep anat
    def find_surface(anat_path, pattern):
        matches = glob.glob(os.path.join(anat_path, pattern))
        if len(matches) == 0:
            raise FileNotFoundError(f"No matches for {pattern} in {anat_path}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple matches for {pattern} in {anat_path}: {matches}")
        return matches[0]
    
    LWHITE = find_surface(anat_path, f"sub-{participant}*_hemi-L_white.surf.gii")
    LPIAL  = find_surface(anat_path, f"sub-{participant}*_hemi-L_pial.surf.gii")
    LMID   = find_surface(anat_path, f"sub-{participant}*_hemi-L_midthickness.surf.gii")
    RWHITE = find_surface(anat_path, f"sub-{participant}*_hemi-R_white.surf.gii")
    RPIAL  = find_surface(anat_path, f"sub-{participant}*_hemi-R_pial.surf.gii")
    RMID   = find_surface(anat_path, f"sub-{participant}*_hemi-R_midthickness.surf.gii")

    # One mapping per hemisphere -> multi-map metrics (one map per IC)
    LMET = os.path.join(out_root, "melodic_IC.L.func.gii")
    RMET = os.path.join(out_root, "melodic_IC.R.func.gii")

    # Map 4D volume -> multi-map metric
    cx(container, home_dir, binds,
       f"wb_command -volume-to-surface-mapping {_q(ic_4d_path)} {_q(LMID)} {_q(LMET)} "
       f"-ribbon-constrained {_q(LWHITE)} {_q(LPIAL)}")
    cx(container, home_dir, binds,
       f"wb_command -volume-to-surface-mapping {_q(ic_4d_path)} {_q(RMID)} {_q(RMET)} "
       f"-ribbon-constrained {_q(RWHITE)} {_q(RPIAL)}")

    #cx(container, home_dir, binds,
    #   f"wb_command -volume-to-surface-mapping {_q(ic_4d_path)} {_q(LMID)} {_q(LMET)} "
    #   f"-enclosing")
    #cx(container, home_dir, binds,
    #   f"wb_command -volume-to-surface-mapping {_q(ic_4d_path)} {_q(RMID)} {_q(RMET)} "
    #   f"-enclosing")

    # Create a single multi-map dscalar from the two multi-map metrics
    DS = os.path.join(out_root, "melodic_IC_all.dscalar.nii")
    cx(container, home_dir, binds,
       f"wb_command -cifti-create-dense-scalar {_q(DS)} -left-metric {_q(LMET)} -right-metric {_q(RMET)}")

    return DS

def collect_t1w_bolds(fmriprep_sub_root: Path, regex_filter: str, target_session: str | None):
    """Find all *_space-T1w_desc-preproc_bold.nii.gz under subject (optionally filtering sessions and regex)."""
    patt = "**/func/*_space-T1w_desc-preproc_bold.nii.gz"
    all_vols = sorted(fmriprep_sub_root.glob(patt))
    if target_session:
        all_vols = [p for p in all_vols if f"/ses-{target_session}/" in str(p)]
    if regex_filter and regex_filter != ".*":
        re_pat = re.compile(regex_filter)
        all_vols = [p for p in all_vols if re_pat.search(p.name)]
    return [str(p) for p in all_vols]

def create_chart_series_from_melodic_mix(mix_txt, TR, out_root, container, home_dir, binds):
    """
    Build CIFTI scalar-series for charts from MELODIC mix (timecourses),
    plus a second scalar-series for power spectra. Add IC names.
    Returns (timeseries_pscalar, powerspec_pscalar).
    """
    def _q(s): return "'" + s.replace("'", "'\\''") + "'"
        
    tc_full = np.loadtxt(mix_txt)
    if tc_full.ndim == 1:
        tc_full = tc_full[:, None]
    n_t_full, n_ic = tc_full.shape

    # --- Truncate time series to first 1000 time points (or less if shorter)
    n_keep = min(1000, n_t_full)
    tc_trunc = tc_full[:n_keep, :]

    # Write text for -cifti-create-scalar-series (rows=series)
    ts_txt = os.path.join(out_root, "melodic_IC_timeseries.txt")
    np.savetxt(ts_txt, tc_trunc.T, fmt="%.8f")

    # Names file: IC 1..N
    names_txt = os.path.join(out_root, "melodic_IC_names.txt")
    with open(names_txt, "w") as f:
        for k in range(1, n_ic+1):
            f.write(f"IC {k}\n")

    # Create scalar-series (unit SECOND, start=0, step=TR)
    ts_pseries = os.path.join(out_root, "melodic_IC_timeseries.sdseries.nii")
    cx(container, home_dir, binds,
       f"wb_command -cifti-create-scalar-series {_q(ts_txt)} {_q(ts_pseries)} "
       f"-name-file {_q(names_txt)} -series SECOND 0 {TR}")

    # Power spectra (magnitude^2 of FFT), freq grid
    freqs = np.fft.rfftfreq(n_t_full, d=TR)
    spec = (np.abs(np.fft.rfft(tc_full, axis=0))**2)  # shape (n_freq, n_ic)
    ps_txt = os.path.join(out_root, "melodic_IC_powerspectrum.txt")
    np.savetxt(ps_txt, spec.T, fmt="%.8f")

    # freq step & start
    if len(freqs) > 1:
        df = float(freqs[1] - freqs[0])
    else:
        df = 0.0
    ps_pseries = os.path.join(out_root, "melodic_IC_powerspectrum.sdseries.nii")
    cx(container, home_dir, binds,
       f"wb_command -cifti-create-scalar-series {_q(ps_txt)} {_q(ps_pseries)} "
       f"-name-file {_q(names_txt)} -series HERTZ {freqs[0]} {df}")

    return ts_pseries, ps_pseries

def create_chart_series_from_dualreg(dr_dir, n_runs, TR, out_root, container, home_dir, binds, prefix="dr"):
    """
    Concatenate dual_regression stage-1 time series across all runs, then:
      - Time-series sdseries: first 1000 timepoints of the concatenated series
      - Power spectrum sdseries: computed from the FULL concatenated series
    Returns (list_of_ts_sdseries, list_of_ps_sdseries), each length 1.
    """
    def _q(s): return "'" + s.replace("'", "'\\''") + "'"

    # ---- load & concatenate all runs along time ----
    ts_list = []
    N_ic_ref = None
    for r in range(n_runs):
        ts_path = os.path.join(dr_dir, f"dr_stage1_subject{r:05d}.txt")
        ts_run = np.loadtxt(ts_path)  # shape: T_run x N_ic
        ts_list.append(ts_run)

    # Concatenated: (T_total x N_ic)
    ts_all = np.concatenate(ts_list, axis=0)
    T_total, N_ic = ts_all.shape

    # ---- Time-series sdseries: first 1000 samples of concatenated ----
    T_keep = min(1000, T_total)
    ts_trunc = ts_all[:T_keep, :]  # (T_keep x N_ic)

    ts_out_txt = os.path.join(out_root, f"{prefix}_timeseries_concat.txt")
    np.savetxt(ts_out_txt, ts_trunc.T, fmt="%.8f")  # rows=series => N_ic x T_keep

    names_txt = os.path.join(out_root, f"{prefix}_names_concat.txt")
    with open(names_txt, "w") as f:
        for k in range(1, N_ic + 1):
            f.write(f"IC {k}\n")

    ts_pseries = os.path.join(out_root, f"{prefix}_timeseries.sdseries.nii")
    cx(container, home_dir, binds,
       f"wb_command -cifti-create-scalar-series {_q(ts_out_txt)} {_q(ts_pseries)} "
       f"-name-file {_q(names_txt)} -series SECOND 0 {TR}")

    # ---- Power spectrum: FULL concatenated series ----
    freqs = np.fft.rfftfreq(T_total, d=TR)
    spec = (np.abs(np.fft.rfft(ts_all, axis=0))**2)  # (n_freq x N_ic)

    ps_out_txt = os.path.join(out_root, f"{prefix}_powerspectrum_concat.txt")
    np.savetxt(ps_out_txt, spec.T, fmt="%.8f")
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0

    ps_pseries = os.path.join(out_root, f"{prefix}_powerspectrum.sdseries.nii")
    cx(container, home_dir, binds,
       f"wb_command -cifti-create-scalar-series {_q(ps_out_txt)} {_q(ps_pseries)} "
       f"-name-file {_q(names_txt)} -series HERTZ {freqs[0]} {df}")

    return ts_pseries, ps_pseries

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Run ICA (FSL MELODIC) for a participant using QuNex suite container; T1w volumes only')
    ap.add_argument('participant', help='BIDS participant ID (e.g., sub-UTS01 or UTS01)')
    ap.add_argument('-p', '--project', default='climblab', help='BIDS project name (default: climblab)')
    ap.add_argument('-s', '--sessions', default=None, help='Optional BIDS session ID to restrict; default: all')
    ap.add_argument('--preprocess-label', default=PREPROCESS_DEFAULT_KEY,
                    help='Name of fMRIPrep derivatives label (default: PREPROCESS_DEFAULT_KEY)')
    ap.add_argument('--container', default='/juice6/u/nlp/climblab/apptainer/images/qunex_suite-1.4.0.sif',
                    help='Path to QuNex suite container (.sif)')
    ap.add_argument('--bind', default='/juice6:/juice6',
                    help='Bind mounts for the container, comma/space-separated (default: /juice6:/juice6)')
    ap.add_argument('--highpass-sec', type=float, default=128.0,
                help='Temporal high-pass cutoff in seconds applied per run BEFORE concatenation (FSL -bptf). '
                     'Set to 0 to disable (default: 128.0).')
    ap.add_argument('--save-name', default='default')
    ap.add_argument('--migpN', type=float, default=-1,
            help='migpN parameter to melodic')
    ap.add_argument('--dummy-vols', type=float, default=0,
        help='Number of dummy volumes to exclude.')
    ap.add_argument('--TR', type=float, default=-1,
            help='Use only the specified TR')

    args = ap.parse_args()

    migpN = int(args.migpN)
    save_name = args.save_name
    dummy_vols = int(args.dummy_vols)
    participant = args.participant.replace('sub-', '')
    project = args.project
    target_TR = args.TR
    target_session = args.sessions
    node = 'session' if target_session else 'subject'
    project_path = os.path.join(BIDS_PATH, project)
    preprocess_label = args.preprocess_label
    regex_filter = '.*'

    # ----------------- locate fMRIPrep volumes in T1w space -----------------
    fmriprep_root = Path(project_path) / 'derivatives' / 'preprocess' / preprocess_label / f'sub-{participant}'
    assert fmriprep_root.exists(), f"fMRIPrep root not found: {fmriprep_root}"
    bolds = collect_t1w_bolds(fmriprep_root, regex_filter, target_session)
    assert bolds, (f'No *-T1w_desc-preproc_bold.nii.gz found for sub-{participant} '
                   f'(label="{preprocess_label}", regex="{regex_filter}", session="{target_session or "ALL"}").')

    ok_tr, TR, uniq_trs, trs = same_tr(bolds)
    if target_TR==-1:
        assert ok_tr, f"Mixed TRs across runs: {uniq_trs}. Split groups or homogenize."
    else:
        TR = target_TR
        bolds = [f for f, tr in zip(bolds, trs) if abs(tr-target_TR)<0.01]

    # FSL -bptf expects sigma in *volumes*. FEAT convention: sigma_vols = cutoff_sec / (2 * TR)
    hp_sigma_vols = None
    if args.highpass_sec and args.highpass_sec > 0:
        hp_sigma_vols = args.highpass_sec / (2.0 * TR)
        stderr(f"Per-run high-pass: cutoff={args.highpass_sec:.3f}s, TR={TR:.6f}s, sigma={hp_sigma_vols:.6f} vols\n")
    else:
        stderr("Per-run high-pass: DISABLED (--highpass-sec set to 0)\n")

    # ----------------- anatomy/surfaces (from same fMRIPrep label) -----------------
    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label=preprocess_label)
    if os.path.basename(os.path.dirname(anat_path)).startswith('ses-'):
        ses_str_anat = f'_ses-{os.path.basename(os.path.dirname(anat_path))[4:]}'
    else:
        ses_str_anat = ''

    # ----------------- outputs -----------------
    out_root = os.path.join(project_path, 'derivatives', 'ica', save_name, f'node-{node}', f'sub-{participant}')
    if target_session:
        out_root = os.path.join(out_root, f'ses-{target_session}')
    os.makedirs(out_root, exist_ok=True)

    #save a list of all bold files analyzed
    listfile = os.path.join(out_root, "inputs.txt")
    with open(listfile, "w") as f:
        for p in bolds:
            f.write(p + "\n")

    # container HOME (writable) to avoid AFS issues
    home_dir = os.path.join(out_root, "_qunex_home")
    os.makedirs(os.path.join(home_dir, ".cache"), exist_ok=True)
    os.makedirs(os.path.join(home_dir, ".local"), exist_ok=True)
    os.makedirs(os.path.join(home_dir, "qunex"), exist_ok=True)

    # ----------------- MELODIC inside container -----------------
    melodic_out = os.path.join(out_root, "melodic")
    os.makedirs(melodic_out, exist_ok=True)

    # --- NEW: per-run high-pass (if requested) ---
    tmpdir = os.path.join(out_root,"tmp")
    os.makedirs(tmpdir, exist_ok=True)

    to_concat = []
    if hp_sigma_vols is not None:
        for i, src in enumerate(bolds):
            dst = os.path.join(tmpdir, f"hp_run{i:03d}.nii.gz")
            
            # -bptf <hp_sigma_vols> -1 means: high-pass only (no low-pass)
            cx(args.container, home_dir, args.bind, f"fslmaths {_shlex(src)} -bptf {hp_sigma_vols:.6f} -1 {_shlex(dst)}")
            if dummy_vols > 0: 
                cx(args.container, home_dir, args.bind, f"fslroi {_shlex(dst)} {_shlex(dst)} {dummy_vols} -1")
            to_concat.append(dst)
    else:
        to_concat = bolds[:]  # no filtering

    #Build a list file for MELODIC (-i <listfile>)
    listfile = os.path.join(out_root, "inputs_bp.txt")
    with open(listfile, "w") as f:
        for p in to_concat:
            f.write(p + "\n")

    # unions mask (if present for all/most runs)
    masks = []
    for r in bolds:
        m = r.replace("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz")
        if os.path.exists(m):
            masks.append(m)
            
    mask_for_melodic = ""
    if masks:
        mask_for_melodic = os.path.join(out_root, "mask_intersect.nii.gz")
        cx(args.container, home_dir, args.bind,
           f"fslmaths {_shlex(masks[0])} -bin {_shlex(mask_for_melodic)}")
        for m in masks[1:]:
            cx(args.container, home_dir, args.bind,
               f"fslmaths {_shlex(mask_for_melodic)} -add {_shlex(m)} {_shlex(mask_for_melodic)}")

    if migpN>0:
        mel_cmd = f"melodic -i {_shlex(listfile)} -o {_shlex(melodic_out)} --tr={TR} --report -v --migpN={migpN}"
    else:
        mel_cmd = f"melodic -i {_shlex(listfile)} -o {_shlex(melodic_out)} --tr={TR} --report -v"
        
    if mask_for_melodic:
        mel_cmd += f" --mask={_shlex(mask_for_melodic)}"
        
    cx(args.container, home_dir, args.bind, mel_cmd)

    # ----------------- dual_regression (stage-1 time series per run) -----------------
    dr_out = os.path.join(out_root, "dual_regression")
    os.makedirs(dr_out, exist_ok=True)

    # dual_regression: provide IC maps and the original (HP-filtered or original) per-run vols
    ic_maps = os.path.join(melodic_out, "melodic_IC")
    assert os.path.exists(ic_maps + ".nii.gz") or os.path.exists(ic_maps + ".nii"), f"Missing {ic_maps}.*"

    # images to back-project = exactly what MELODIC saw (our to_concat list)
    # build arg string safely
    imgs_arg = " ".join(_shlex(p) for p in to_concat)
    dr_cmd = (
        f"dual_regression {_shlex(ic_maps)} 1 -1 0 "
        f"{_shlex(dr_out)} {imgs_arg}"
    )
    cx(args.container, home_dir, args.bind, dr_cmd)
    cx(args.container, home_dir, args.bind, f"find {_shlex(dr_out)} -maxdepth 1 -type f -name 'dr_stage2_*' -delete")
    shutil.rmtree(os.path.join(out_root, "dual_regression", "scripts+logs"))

    #--remove tmp filtered files--
    shutil.rmtree(tmpdir)

    # ----------------- Workbench packaging (.spec + volume & surface ICs) -----------------
    if participant[0:4] != 'sub-':
        climbprep.convert_inflated.main([project, 'sub-' + participant])
    else:
        climbprep.convert_inflated.main([project, participant])

    spec_path = os.path.join(out_root, f"sub-{participant}{ses_str_anat}_{save_name}.spec")
    with open(spec_path.replace('.spec', '.json'), 'w') as f:
        json.dump(dict(Description='Specification to load ICA outputs (volume & surface ICs) into wb_view.'), f, indent=2)

    # add surfaces
    add_surfaces_to_spec(spec_path, out_root, anat_path, participant, ses_str_anat,
                         args.container, home_dir, args.bind)

    # add 4D IC volume
    ic4d = os.path.join(melodic_out, "melodic_IC.nii.gz")
    assert os.path.exists(ic4d), f"Missing MELODIC IC volume: {ic4d}"
    cx(args.container, home_dir, args.bind,
       f"wb_command -add-to-spec-file {_shlex(spec_path)} INVALID {_shlex(ic4d)}")

    # surface ICs dscalar
    dscalar_all = map_ics_to_surface_and_merge_dscalar(ic4d, out_root, anat_path, participant, ses_str_anat,
                                                       args.container, home_dir, args.bind)
    cx(args.container, home_dir, args.bind,
       f"wb_command -add-to-spec-file {_shlex(spec_path)} CORTEX {_shlex(dscalar_all)}")

    # --- NEW: chartable IC series (lines) ---
    ts_series, ps_series = create_chart_series_from_dualreg(
        dr_dir=os.path.join(out_root, "dual_regression"),
        n_runs=len(bolds),
        TR=TR,
        out_root=out_root,
        container=args.container,
        home_dir=home_dir,
        binds=args.bind,
        prefix="melodic_IC"
    )
    
    #mix_txt = os.path.join(melodic_out, "melodic_mix")
    #assert os.path.exists(mix_txt), f"Missing MELODIC timecourses: {mix_txt}"
    #ts_pseries, ps_pseries = create_chart_series_from_melodic_mix(mix_txt, TR, out_root, args.container, home_dir, args.bind)

    # Add both scalar-series files to spec (Workbench shows them in Chart pane)
    cx(args.container, home_dir, args.bind,
       f"wb_command -spec-file-modify {_shlex(spec_path)} "
       f"-add INVALID {_shlex(ts_series)} "
       f"-add INVALID {_shlex(ps_series)}")

    #copy scene template for visualizing components
    with as_file(files("climbprep")/"resources"/"melodic_viewer.scene") as s: 
        shutil.copyfile(s, out_root + "/melodic_viewer.scene")

    stderr(f"\nDone.\nOutputs:\n"
           f"  MELODIC: {os.path.join(out_root, 'melodic')}\n"
           f"  Workbench spec: {spec_path}\n"
           f"  Surface ICs (merged): {dscalar_all}\n"
           f"  Chart series (timeseries): {ts_series}\n"
           f"  Chart series (power spectrum): {ps_series}\n")
