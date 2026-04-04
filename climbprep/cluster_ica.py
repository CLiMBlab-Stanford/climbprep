import argparse
from climbprep.constants import *
from climbprep.util import *
import os
from pathlib import Path
import numpy as np
import json
import shutil
import nibabel as nib

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
    out_root = os.path.join(BIDS_PATH, project, 'derivatives', 'cluster_ica', ica_label, f'node-{node}', f'sub-{participant}')
    if target_session:
        out_root = os.path.join(out_root, f'ses-{target_session}')
    os.makedirs(out_root, exist_ok=True)

    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label='main')
    home_dir = os.path.join(out_root, "_qunex_home")
    TR = 2.0
    hp_sigma_vols = args.highpass_sec / (2.0 * TR)
            
    bold_filenames = Path(os.path.join(ica_root, 'inputs.txt')).read_text(encoding="utf-8").splitlines()
    manual_signals = np.loadtxt(os.path.join(ica_root, 'signals.txt')).astype(np.int32)

    ic_path = os.path.join(ica_root, 'melodic', 'melodic_IC.nii.gz')
    ic_img = nib.load(str(ic_path))
    ics    = ic_img.get_fdata(dtype=np.float32)         # shape: (X,Y,Z,K)
    aff      = ic_img.affine
    hdr      = ic_img.header.copy()

    Amat = ics.reshape(-1, ics.shape[3])        # (V, K)
    Amat = Amat[:, manual_signals-1]

    import numpy as np
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    
    # X: shape (n_samples, n_features). We cluster the columns (features).
    D = pdist(Amat.T, metric='correlation')          # 1 - Pearson r between columns
    Z = linkage(D, method='average')              # or 'ward','complete','single', etc. (not with 'correlation' for ward)
    Z = optimal_leaf_ordering(Z, D)               # nicer, distance-aware ordering
    order = leaves_list(Z)                        # column permutation
    Amat_reordered = Amat[:, order]                     # columns reordered by hierarchical clustering

    #reshape and save again as a volume
    Amat_reordered_4d = Amat_reordered.reshape(ics.shape[0], ics.shape[1], ics.shape[2], Amat_reordered.shape[1])

    img_file = nib.Nifti1Image(Amat_reordered_4d, affine=aff, header=hdr)
    sig_comp = str(os.path.join(out_root, f"reordered_signal_components.nii.gz"))
    img_file.to_filename(sig_comp)

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
    LMET = os.path.join(out_root, "sig_clustered.L.func.gii")
    RMET = os.path.join(out_root, "sig_clustered.R.func.gii")

    # Map 4D volume -> multi-map metric
    cx(args.container, home_dir, args.bind,
       f"wb_command -volume-to-surface-mapping {_shlex(sig_comp)} {_shlex(LMID)} {_shlex(LMET)} "
       f"-ribbon-constrained {_shlex(LWHITE)} {_shlex(LPIAL)}")
    cx(args.container, home_dir, args.bind,
       f"wb_command -volume-to-surface-mapping {_shlex(sig_comp)} {_shlex(RMID)} {_shlex(RMET)} "
       f"-ribbon-constrained {_shlex(RWHITE)} {_shlex(RPIAL)}")
    
    # Create a single multi-map dscalar from the two multi-map metrics
    DS = os.path.join(out_root, "sig_clustered.dscalar.nii")
    cx(args.container, home_dir, args.bind,
       f"wb_command -cifti-create-dense-scalar {_shlex(DS)} -left-metric {_shlex(LMET)} -right-metric {_shlex(RMET)}")

    #make spec file
    spec_path = os.path.join(out_root, "sig_clustered.spec")
    with open(spec_path.replace('.spec', '.json'), 'w') as f:
        json.dump(dict(Description='Specification to load clustered ICA signal components.'), f, indent=2)
        
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

    #add components
    cx(args.container, home_dir, args.bind,
       f"wb_command -add-to-spec-file {_shlex(spec_path)} CORTEX {_shlex(DS)}")

    #add volume components
    cx(args.container, home_dir, args.bind,
       f"wb_command -add-to-spec-file {_shlex(spec_path)} INVALID {_shlex(sig_comp)}")

    