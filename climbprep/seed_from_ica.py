import argparse
from climbprep.constants import *
from climbprep.util import *
import os
from pathlib import Path
import numpy as np
import json
import shutil
import nibabel as nib
import scipy.stats

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

    args = ap.parse_args()
    participant = args.participant.replace('sub-', '')
    project = args.project
    target_session = args.sessions
    node = 'session' if target_session else 'subject'
    ica_label = 'fsnative'

    #ica input dir
    ica_root = os.path.join(BIDS_PATH, project, 'derivatives', 'ica', ica_label, f'node-{node}', f'sub-{participant}')
    if target_session:
        ica_root = os.path.join(ica_root, f'ses-{target_session}')

    #seed output dir
    out_root = os.path.join(BIDS_PATH, project, 'derivatives', 'seed_from_ica', ica_label, f'node-{node}', f'sub-{participant}')
    if target_session:
        out_root = os.path.join(out_root, f'ses-{target_session}')

    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label='main')
    home_dir = os.path.join(out_root, "_qunex_home")
    TR = 2.0
    hp_sigma_vols = args.highpass_sec / (2.0 * TR)
            
    bold_filenames = Path(os.path.join(ica_root, 'inputs.txt')).read_text(encoding="utf-8").splitlines()
    manual_signals = np.loadtxt(os.path.join(ica_root, 'signals.txt')).astype(np.int32)
    all_bold = []
    all_dr = []
    
    for file_idx in range(len(bold_filenames)):
        #highpass filter
        hp_file = os.path.join(out_root, 'hp_run_' + str(file_idx))
        cx(args.container, home_dir, args.bind, f"fslmaths {_shlex(bold_filenames[file_idx])} -bptf {hp_sigma_vols:.6f} -1 {_shlex(hp_file)}")

        #load and accumulate for the great regression
        bold_img = nib.load(str(hp_file) + '.nii.gz')
        bold = bold_img.get_fdata(dtype=np.float32)     # shape: (X,Y,Z,T)
        bold = bold.copy()
        
        X = bold.shape[0]
        Y = bold.shape[1]
        Z = bold.shape[2]
        
        aff = bold_img.affine
        hdr = bold_img.header.copy()
        bold = bold.reshape(-1, bold.shape[3])

        #z-score witihn each run
        bold[np.isnan(bold)]=0
        bold = scipy.stats.zscore(bold, axis=1)
        bold[np.isnan(bold)]=0
        
        all_bold.append(bold.T)

        #get dr time series
        dr_txt = os.path.join(ica_root, "dual_regression", f"dr_stage1_subject{file_idx:05d}.txt")
        all_dr.append(np.loadtxt(dr_txt).astype(np.float32))

    #mega-concat
    all_bold = np.concatenate(all_bold, axis=0)
    all_dr = np.concatenate(all_dr, axis=0)

    #regression with signal + noise components
    #[T, S] @ B = [T, V] 
    noise_idx = np.setdiff1d(np.arange(0, all_dr.shape[1]).astype(np.int32), manual_signals-1)
    
    #coef = np.linalg.pinv(all_dr) @ all_bold
    #noise_coef = coef[noise_idx,:]
    #noise_recon = all_dr[:,noise_idx] @ noise_coef    
    #all_bold = all_bold - noise_recon

    X_mat = np.hstack([np.ones((all_dr.shape[0], 1), dtype=np.float32), all_dr])
    coef, *_ = np.linalg.lstsq(X_mat, all_bold, rcond=None)   # coef shape: (1+Ncomp, V)
    noise_coef = coef[1 + noise_idx, :]
    
    noise_recon = all_dr[:,noise_idx] @ noise_coef    
    all_bold = all_bold - noise_recon

    #save off pieces
    all_bold = (all_bold.T).reshape(X, Y, Z, all_bold.shape[0])
    c_idx = 0
    n_pieces = 0
    
    done = False
    while not done:
        tmp = all_bold[:,:,:,c_idx:(c_idx+5000)]

        print(tmp.shape)
        clean_img = nib.Nifti1Image(tmp, affine=aff, header=hdr)
        denoised = str(os.path.join(out_root, f"clean_piece_{n_pieces:03d}.nii.gz"))
        clean_img.to_filename(denoised)

        c_idx += 5000
        n_pieces += 1
        if c_idx>=all_bold.shape[3]-1:
            done = True
    
    #surface-project each piece
    for piece_idx in range(n_pieces):
        #--project to surface--
        # Surfaces / ROIs from fMRIPrep anat
        if os.path.basename(os.path.dirname(anat_path)).startswith('ses-'):
            ses_str_anat = f'_ses-{os.path.basename(os.path.dirname(anat_path))[4:]}'
        else:
            ses_str_anat = ''
        
        LWHITE = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-L_white.surf.gii")
        LPIAL  = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-L_pial.surf.gii")
        LMID   = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-L_midthickness.surf.gii")
        RWHITE = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-R_white.surf.gii")
        RPIAL  = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-R_pial.surf.gii")
        RMID   = os.path.join(anat_path, f"sub-{participant}{ses_str_anat}_hemi-R_midthickness.surf.gii")
    
        # One mapping per hemisphere -> multi-map metrics (one map per IC)
        LMET = os.path.join(out_root, "clean_piece_" + str(piece_idx) + ".L.func.gii")
        RMET = os.path.join(out_root, "clean_piece_" + str(piece_idx) + ".R.func.gii")
    
        # Map 4D volume -> multi-map metric
        denoised = str(os.path.join(out_root, f"clean_piece_{piece_idx:03d}.nii.gz"))
        
        cx(args.container, home_dir, args.bind,
           f"wb_command -volume-to-surface-mapping {_shlex(denoised)} {_shlex(LMID)} {_shlex(LMET)} "
           f"-ribbon-constrained {_shlex(LWHITE)} {_shlex(LPIAL)}")
        cx(args.container, home_dir, args.bind,
           f"wb_command -volume-to-surface-mapping {_shlex(denoised)} {_shlex(RMID)} {_shlex(RMET)} "
           f"-ribbon-constrained {_shlex(RWHITE)} {_shlex(RPIAL)}")
        
        # Create a single multi-map dscalar from the two multi-map metrics
        DS = os.path.join(out_root, "clean_piece_" + str(piece_idx) + ".dtseries.nii")
        cx(args.container, home_dir, args.bind,
           f"wb_command -cifti-create-dense-timeseries {_shlex(DS)} -left-metric {_shlex(LMET)} -right-metric {_shlex(RMET)} -timestep 2.0 -timestart 0")
        
    #merge all surface time series
    merged_DS = os.path.join(out_root, "clean_merge_all.dtseries.nii")
    if n_pieces>1:
        cmd = f'wb_command -cifti-merge {merged_DS}'
        for piece_idx in range(n_pieces):
            cleaned_run = os.path.join(out_root, "clean_piece_" + str(piece_idx) + ".dtseries.nii")
            cmd += f' -cifti {cleaned_run}'
        cx(args.container, home_dir, args.bind, cmd)
    else:
        shutil.copy2(os.path.join(out_root, "clean_piece_0.dtseries.nii"), merged_DS)

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
    subprocess.run('rm ' + out_root + '/*piece*', shell=True, check=True)
    subprocess.run('rm ' + out_root + '/*run*', shell=True, check=True)
