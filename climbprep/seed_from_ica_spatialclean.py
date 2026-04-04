import argparse
from climbprep.constants import *
from climbprep.util import *
import os
from pathlib import Path
import numpy as np
import json
import shutil
import scipy.stats
import nibabel as nib
import sys, re
import pandas as pd
import glob

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

def same_tr(files):
    trs = []
    for f in files:
        img = nib.load(f)
        z = img.header.get_zooms()
        if len(z) < 4:
            raise AssertionError(f"Not a 4D NIfTI: {f}")
        trs.append(float(z[3]))
    u = sorted(set(round(x, 6) for x in trs))
    return (len(u) == 1, float(u[0]) if u else None, u)
    
def cx(container: str, home_dir: str, binds: str, inner: str):
    """Run a bash snippet inside the QuNex container with clean env & writable HOME."""
    cmd = f"singularity exec --cleanenv -H {_shlex(home_dir)}{_bind_flags(binds)} {_shlex(container)} bash -lc {_shlex(inner)}"
    sh(cmd)
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser('Run ICA (FSL MELODIC) for a participant using QuNex suite container; T1w volumes only')
    ap.add_argument('participant', help='BIDS participant ID (e.g., sub-UTS01 or UTS01)')
    ap.add_argument('-p', '--project', default='climblab', help='BIDS project name (default: climblab)')
    ap.add_argument('-s', '--sessions', default=None, help='Optional BIDS session ID to restrict; default: all')
    ap.add_argument('--container', default='/juice6/u/nlp/climblab/apptainer/images/qunex_suite-1.4.0.sif',
                    help='Path to QuNex suite container (.sif)')
    ap.add_argument('--bind', default='/juice6:/juice6',
                    help='Bind mounts for the container, comma/space-separated (default: /juice6:/juice6)')
    ap.add_argument('--highpass-sec', type=float, default=128.0,
                help='Temporal high-pass cutoff in seconds applied per run BEFORE concatenation (FSL -bptf). '
                     'Set to 0 to disable (default: 128.0).')
    ap.add_argument('--dummy-vols', type=float, default=0,
            help='Number of dummy volumes to exclude.')
    ap.add_argument('--ica-label', default='fsnative',
                help='ica run label')
    ap.add_argument('--mode', default='regressors',
                help='keep_signal or remove_noise or regressors')
    
    args = ap.parse_args()
    participant = args.participant.replace('sub-', '')
    project = args.project
    target_session = args.sessions
    dummy_vols = int(args.dummy_vols)
    node = 'session' if target_session else 'subject'
    ica_label = args.ica_label
    clean_mode = args.mode

    #ica input dir
    ica_root = os.path.join(BIDS_PATH, project, 'derivatives', 'ica', ica_label, f'node-{node}', f'sub-{participant}')
    if target_session:
        ica_root = os.path.join(ica_root, f'ses-{target_session}')

    #seed output dir
    out_root = os.path.join(BIDS_PATH, project, 'derivatives', 'seed_from_ica_spatialclean', clean_mode, f'node-{node}', f'sub-{participant}')
    if target_session:
        out_root = os.path.join(out_root, f'ses-{target_session}')
    os.makedirs(out_root, exist_ok=True)

    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label='main')
    home_dir = os.path.join(out_root, "_qunex_home")
    bold_filenames = Path(os.path.join(ica_root, 'inputs.txt')).read_text(encoding="utf-8").splitlines()

    ok_tr, TR, uniq_trs = same_tr(bold_filenames)
    assert ok_tr, f"Mixed TRs across runs: {uniq_trs}. Split groups or homogenize."

    hp_sigma_vols = args.highpass_sec / (2.0 * TR)
    manual_signals = np.loadtxt(os.path.join(ica_root, 'signals.txt')).astype(np.int32)
    ic_path = os.path.join(ica_root, 'melodic', 'melodic_IC.nii.gz')
    ic_img = nib.load(str(ic_path))
    ics    = ic_img.get_fdata(dtype=np.float32)         # shape: (X,Y,Z,K)

    K = ics.shape[3]
    Amat = ics.reshape(-1, K).copy()        # (V, K)
    Amat = Amat[:,manual_signals-1]
    A_pinv = np.linalg.pinv(Amat, rcond=1e-6)

    A_all   = ics.reshape(-1, K).copy()                    # (V, K) all ICs
    Aall_pinv = np.linalg.pinv(A_all, rcond=1e-6)          # (K, V)
    noise_idx = np.setdiff1d(np.arange(0,K).astype(np.int32), manual_signals-1)
    
    #get noise components
    #NoiseMat = ics.reshape(-1, K).copy()        # (V, K)
    #noise_idx = np.setdiff1d(np.arange(0,K).astype(np.int32), manual_signals-1)
    #NoiseMat = NoiseMat[:,noise_idx]
    #NoiseMat_pinv = np.linalg.pinv(NoiseMat, rcond=1e-6)
    merge_command = "fslmerge -t " + _shlex(os.path.join(out_root, "concat_clean.nii.gz"))

    for file_idx in range(len(bold_filenames)):
        #highpass filter
        hp_file = os.path.join(out_root, 'hp_run_' + str(file_idx))
        cx(args.container, home_dir, args.bind, f"fslmaths {_shlex(bold_filenames[file_idx])} -bptf {hp_sigma_vols:.6f} -1 {_shlex(hp_file)}")

        #--spatial projection cleaning--
        bold_img = nib.load(str(hp_file) + '.nii.gz')
        bold     = bold_img.get_fdata(dtype=np.float32)     # shape: (X,Y,Z,T)
        aff      = bold_img.affine
        hdr      = bold_img.header.copy()
        X, Y, Z, T = bold.shape        
        K = ics.shape[3]

        #too short
        if T<50:
            continue
        
        Ymat = bold.reshape(-1, T)       # (V, T)
        V = Amat.shape[0]

        #normalize bold
        Ymat[np.isnan(Ymat)]=0
        Ymat = scipy.stats.zscore(Ymat, axis=1)
        Ymat[np.isnan(Ymat)]=0

        assert clean_mode=='regressorsonly' or clean_mode=='regressors' or clean_mode=='cleansignal', "invalid clean mode"

        confounds_path = bold_filenames[file_idx].replace("_space-T1w_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
        #confounds_regex = r'^(?:trans|rot)_[xyz](?:$|_(?:derivative1|power2|derivative1_power2)$)|global_signal(?:$|_derivative1|_power2|_derivative1_power2)$|a_comp_cor_.*|non_steady_state_outlier.*|motion_outlier.*|framewise_displacement$'
        confounds_regex = r'^(?:trans|rot)_[xyz](?:$|_(?:derivative1|power2)$)|global_signal(?:$|_derivative1|_power2)$|a_comp_cor_0[0-4]$|non_steady_state_outlier.*|motion_outlier.*|framewise_displacement$'
        
        confounds = pd.read_csv(confounds_path, sep='\t')
        confounds = confounds.filter(regex=confounds_regex)
        confounds = confounds.fillna(0)

        if clean_mode == 'regressorsonly':
            Y_residual = Ymat

            C = confounds.to_numpy(dtype=np.float64)          # (T, P)
            C = np.column_stack([np.ones((C.shape[0], 1)), C])  # add intercept -> (T, P+1)
            
            # Y_residual is (V, T); transpose to (T, V) to fit
            Yres_T = Y_residual.T.astype(np.float64)          # (T, V)
            
            beta, *_ = np.linalg.lstsq(C, Yres_T, rcond=None) # (P+1, V)
            Yres_clean_T = Yres_T - C @ beta                   # (T, V)
            Y_residual_cleaned = Yres_clean_T.T.astype(np.float32)  # (V, T)
            Yhat_m = Y_residual_cleaned

        elif clean_mode == 'regressors':
            C_all = Aall_pinv @ Ymat
            Y_signal = A_all[:, manual_signals-1] @ C_all[manual_signals-1, :]
            Y_residual = Ymat - A_all @ C_all

            C = confounds.to_numpy(dtype=np.float64)          # (T, P)
            C = np.column_stack([np.ones((C.shape[0], 1)), C])  # add intercept -> (T, P+1)
            
            # Y_residual is (V, T); transpose to (T, V) to fit
            Yres_T = Y_residual.T.astype(np.float64)          # (T, V)
            
            beta, *_ = np.linalg.lstsq(C, Yres_T, rcond=None) # (P+1, V)
            Yres_clean_T = Yres_T - C @ beta                   # (T, V)
            Y_residual_cleaned = Yres_clean_T.T.astype(np.float32)  # (V, T)
            Yhat_m = Y_signal + Y_residual_cleaned
            
        elif clean_mode == 'cleansignal':
            C_all = Aall_pinv @ Ymat
            Y_noise = A_all[:, noise_idx] @ C_all[noise_idx, :]
            Y_residual = Ymat - Y_noise

            C = confounds.to_numpy(dtype=np.float64)          # (T, P)
            C = np.column_stack([np.ones((C.shape[0], 1)), C])  # add intercept -> (T, P+1)
            
            # Y_residual is (V, T); transpose to (T, V) to fit
            Yres_T = Y_residual.T.astype(np.float64)          # (T, V)
            
            beta, *_ = np.linalg.lstsq(C, Yres_T, rcond=None) # (P+1, V)
            Yres_clean_T = Yres_T - C @ beta                   # (T, V)
            Y_residual_cleaned = Yres_clean_T.T.astype(np.float32)  # (V, T)
            Yhat_m = Y_residual_cleaned

        Yhat_4d = Yhat_m.reshape(X, Y, Z, T)

        #remove dummy volumes if specified
        if dummy_vols>0:
            Yhat_4d = Yhat_4d[:,:,:,dummy_vols:]
            
        clean_img = nib.Nifti1Image(Yhat_4d, affine=aff, header=hdr)
        denoised = str(os.path.join(out_root, f"clean_run_{file_idx:03d}.nii.gz"))
        clean_img.to_filename(denoised)

        #--project to surface--
        # Surfaces / ROIs from fMRIPrep anat
        if os.path.basename(os.path.dirname(anat_path)).startswith('ses-'):
            ses_str_anat = f'_ses-{os.path.basename(os.path.dirname(anat_path))[4:]}'
        else:
            ses_str_anat = ''
        
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
        LMET = os.path.join(out_root, "clean_run_" + str(file_idx) + ".L.func.gii")
        RMET = os.path.join(out_root, "clean_run_" + str(file_idx) + ".R.func.gii")
    
        # Map 4D volume -> multi-map metric
        cx(args.container, home_dir, args.bind,
           f"wb_command -volume-to-surface-mapping {_shlex(denoised)} {_shlex(LMID)} {_shlex(LMET)} "
           f"-ribbon-constrained {_shlex(LWHITE)} {_shlex(LPIAL)} -voxel-subdiv 7")
        cx(args.container, home_dir, args.bind,
           f"wb_command -volume-to-surface-mapping {_shlex(denoised)} {_shlex(RMID)} {_shlex(RMET)} "
           f"-ribbon-constrained {_shlex(RWHITE)} {_shlex(RPIAL)} -voxel-subdiv 7")
        
        # Create a single multi-map dscalar from the two multi-map metrics
        DS = os.path.join(out_root, "clean_run_" + str(file_idx) + ".dtseries.nii")
        cx(args.container, home_dir, args.bind,
           f"wb_command -cifti-create-dense-timeseries {_shlex(DS)} -left-metric {_shlex(LMET)} -right-metric {_shlex(RMET)} -timestep {TR} -timestart 0")

        #for later volume examination
        merge_command = merge_command + " " + _shlex(denoised)

    #merge clean volumes for later analysis
    cx(args.container, home_dir, args.bind, merge_command)
    
    #merge all dtseries
    merged_DS = os.path.join(out_root, "clean_merge_all.dtseries.nii")
    cmd = f'wb_command -cifti-merge {merged_DS}'
    for file_idx in range(len(bold_filenames)):
        cleaned_run = os.path.join(out_root, "clean_run_" + str(file_idx) + ".dtseries.nii")
        if os.path.exists(cleaned_run):
            cmd += f' -cifti {cleaned_run}'

    cx(args.container, home_dir, args.bind, cmd)

    #smooth 2mm
    smoothed_DS = os.path.join(out_root, "clean_merge_all_2mm.dtseries.nii")

    cx(args.container, home_dir, args.bind,
       f"wb_command -cifti-smoothing {_shlex(merged_DS)} 2 0 COLUMN {_shlex(smoothed_DS)} "
       f"-left-surface {_shlex(LMID)} -right-surface {_shlex(RMID)} -fwhm")
    
    #make spec file
    spec_path = os.path.join(out_root, "merged_all.spec")
    with open(spec_path.replace('.spec', '.json'), 'w') as f:
        json.dump(dict(Description='Specification to load merged dtseries for seed point analysis, cleaned with ICA.'), f, indent=2)
        
    for surf in ('pial', 'white', 'midthickness', 'inflated'):
        for hemi in ('L', 'R'):
            suffix = '.shape.gii' if surf == 'sulc' else '.surf.gii'
            file = list(Path(anat_path).glob(f'sub-{participant}*_hemi-{hemi}_{surf}{suffix}'))[0]
            src = str(file)
            print(src)
            
            dst = os.path.join(out_root, os.path.basename(src))
            shutil.copy(src, dst)
            
            which_hemi = 'LEFT' if hemi == 'L' else 'RIGHT'
            cx(args.container, home_dir, args.bind,
               f"wb_command -add-to-spec-file {_shlex(spec_path)} CORTEX_{which_hemi} {_shlex(dst)}")

    #add merged dtseries
    cx(args.container, home_dir, args.bind,
       f"wb_command -add-to-spec-file {_shlex(spec_path)} CORTEX {_shlex(merged_DS)}")

    #remove individual run files
    import subprocess
    subprocess.run('rm ' + out_root + '/*run*', shell=True, check=True)
    