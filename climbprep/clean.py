import json
import yaml
import pandas as pd
from nilearn import image, surface, maskers, signal, interfaces
import argparse

from climbprep.constants import *

IMG_CACHE = {}

def load_img(path, cache=IMG_CACHE):
    if path in cache:
        return cache[path]
    img = image.load_img(path)
    cache[path] = img
    return img


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Clean (denoise) a participant's functional data")
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default=CLEAN_DEFAULT_KEY, help=('Keyword (currently `firstlevels` or '
        '`fc`) or YAML config file to used to parameterize cleaning. If a keyword is provided, will use the default '
        'settings for that type of downstream analysis. If not provided, will use default settings. '
        'To view the available config options, see `DEFAULTS["clean"]["firstlevels"]` in `climbprep/constants.py`.'))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    config = args.config
    if config in DEFAULTS['clean']:
        config_default = DEFAULTS['clean'][config]
        config = {}
    else:
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        config_default = DEFAULTS['clean'][CLEAN_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'cleaning_label' in config, 'Required field `cleaning_label` not found in config. ' \
                                       'Please provide a valid config file or keyword.'
    cleaning_label = config['cleaning_label']
    assert 'preprocessing_label' in config, 'Required field `preprocessing_label` not found in config. ' \
                                            'Please provide a valid config file or keyword.'
    preprocessing_label = config['preprocessing_label']

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, args.project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path

    sessions = set()
    for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}
    for session in sessions:
        # Set session-dependent paths
        subdir = 'sub-%s' % participant
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        fmriprep_path = os.path.join(derivatives_path, 'fmriprep', preprocessing_label, subdir)
        assert os.path.exists(fmriprep_path), 'Path not found: %s' % fmriprep_path
        raw_path = os.path.join(project_path, subdir)
        assert os.path.exists(raw_path), 'Path not found: %s' % raw_path
        func_path = os.path.join(fmriprep_path, 'func')
        assert os.path.exists(func_path), 'Path not found: %s' % func_path
        anat_path = os.path.join(fmriprep_path, 'anat')
        assert os.path.exists(anat_path), 'Path not found: %s' % anat_path

        if session:
            ses_str = f'_ses-{session}'
        else:
            ses_str = ''

        datasets = {}  # Structure: space > run > filetype (func, mask, confounds, tr) > value
        type_by_space = {}
        for img_path in os.listdir(func_path):
            if img_path.endswith('desc-preproc_bold.nii.gz'):
                space = SPACE_RE.match(img_path)
                run = RUN_RE.match(img_path)
                task = TASK_RE.match(img_path)
                if space and run and task:
                    space = space.group(1)
                    type_by_space[space] = 'vol'
                    run = run.group(1)
                    task = task.group(1)
                    if space == 'T1w':
                        mask = f'sub-{participant}{ses_str}_label-GM_probseg.nii.gz'
                    else:
                        mask = f'sub-{participant}{ses_str}_space-{space}_label-GM_probseg.nii.gz'
                    mask = os.path.join(anat_path, mask)
                    # confounds = f'sub-{participant}{ses_str}_task-{task}_run-{run}_desc-confounds_timeseries.tsv'
                    confounds = os.path.join(func_path, img_path)
                    assert os.path.exists(confounds), 'Confounds file not found: %s' % confounds
                    func = os.path.join(func_path, img_path)
                    raw_sidecar_path = os.path.join(
                        raw_path, 'func', img_path.split('_run-')[0] + '_run-' + run + '_bold.json'
                    )
                    assert os.path.exists(raw_sidecar_path), 'Path not found: %s' % raw_sidecar_path
                    with open(raw_sidecar_path, 'r') as f:
                        raw_sidecar = json.load(f)
                    TR = raw_sidecar.get('RepetitionTime', None)
                    assert TR, 'RepetitionTime information not found in raw sidecar: %s' % raw_sidecar_path
                    if space not in datasets:
                        datasets[space] = {}
                    if run not in datasets[space]:
                        datasets[space][run] = {}
                    datasets[space][run]['func'] = func
                    datasets[space][run]['mask'] = mask
                    datasets[space][run]['confounds'] = confounds
                    datasets[space][run]['TR'] = TR
            elif img_path.endswith('_bold.func.gii') and '_hemi-L_' in img_path:
                space = SPACE_RE.match(img_path)
                run = RUN_RE.match(img_path)
                task = TASK_RE.match(img_path)
                if space and run and task:
                    space = space.group(1)
                    type_by_space[space] = 'surf'
                    run = run.group(1)
                    task = task.group(1)
                    # confounds = f'sub-{participant}{ses_str}_task-{task}_run-{run}_desc-confounds_timeseries.tsv'
                    confounds = os.path.join(func_path, img_path)
                    assert os.path.exists(confounds), 'Confounds file not found: %s' % confounds
                    func = os.path.join(func_path, img_path)
                    raw_sidecar_path = os.path.join(
                        raw_path, 'func', img_path.split('_run-')[0] + '_run-' + run + '_bold.json'
                    )
                    assert os.path.exists(raw_sidecar_path), 'Path not found: %s' % raw_sidecar_path
                    with open(raw_sidecar_path, 'r') as f:
                        raw_sidecar = json.load(f)
                    TR = raw_sidecar.get('RepetitionTime', None)
                    assert TR, 'RepetitionTime information not found in raw sidecar: %s' % raw_sidecar_path
                    if space not in datasets:
                        datasets[space] = {}
                    if run not in datasets[space]:
                        datasets[space][run] = {}
                    datasets[space][run]['func'] = func
                    datasets[space][run]['mask'] = None
                    datasets[space][run]['confounds'] = confounds
                    datasets[space][run]['TR'] = TR


        out_dir = os.path.join(derivatives_path, 'cleaned', cleaning_label, subdir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save the configuration used for cleaning
        config_path = os.path.join(out_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Volumetric data
        for space in datasets:
            for run in datasets[space]:
                confounds = datasets[space][run]['confounds']
                mask = datasets[space][run]['mask']
                func_path = datasets[space][run]['func']
                func_file = os.path.basename(func_path)
                TR = datasets[space][run]['TR']

                confounds, sample_mask = interfaces.fmriprep.load_confounds(
                    confounds,
                    strategy=config['strategy'],
                    std_dvars_threshold=config['std_dvars_threshold'],
                    fd_threshold=config['fd_threshold']
                )
                if sample_mask is None:
                    sample_mask = []

                _sample_mask = pd.DataFrame(dict(sample_mask=list(sample_mask)))
                sample_mask_path = os.path.join(
                    out_dir, func_file.replace('_bold.nii.gz', '_samplemask.tsv')
                )
                _sample_mask.to_csv(sample_mask_path, sep='\t', index=False)

                if type_by_space[space] == 'vol':  # Volumetric data
                    mask_nii = load_img(mask)
                    func = load_img(func_path)
                    mask_nii = image.math_img(
                        'img > 0.5',
                        img=image.resample_to_img(
                            mask_nii, func, interpolation='nearest', force_resample=True, copy_header=True
                        )
                    )

                    masker = maskers.NiftiMasker(
                        mask_img=mask_nii,
                        standardize=config['standardize'],
                        detrend=config['detrend'],
                        t_r=TR,
                        low_pass=config['smoothing_fwhm'],
                        high_pass=config['high_pass']
                    )
                    kwargs = dict(confounds=confounds)
                    if config['scrub']:
                        kwargs['sample_mask'] = sample_mask[i]
                        desc = 'desc-cleanscrubbed'
                    else:
                        desc = 'desc-clean'
                    run = masker.fit_transform(func_path, **kwargs)
                    run = masker.inverse_transform(run)
                    run_path = os.path.join(
                        out_dir, func_file.replace('desc-preproc', desc)
                    )
                    run.to_filename(run_path)
                elif type_by_space[space] == 'vol':  # Surface data
                    mask_nii = None
                    if space == 'fsnative':
                        space_str = ''
                    else:
                        space_str = '_space-%s' % space
                    surf_L_path = os.path.join(
                        fmriprep_path, 'anat', f'sub-{participant}{ses_str}{space_str}_hemi-L_pial.surf.gii'
                    )
                    surf_R_path = surf_L_path.replace('_hemi-L_', '_hemi-R_')

                    masker = maskers.SurfaceMasker(
                        mask_img=None,
                        standardize=config['standardize'],
                        detrend=config['detrend'],
                        t_r=TR,
                        low_pass=config['smoothing_fwhm'],
                        high_pass=config['high_pass']
                    )
                    kwargs = dict(confounds=confounds)
                    if config['scrub']:
                        kwargs['sample_mask'] = sample_mask[i]
                        desc = 'desc-cleanscrubbed'
                    else:
                        desc = 'desc-clean'
                    mesh = surface.PolyMesh(left=surf_L_path, right=surf_R_path)
                    data = surface.PolyData(left=func_path, right=func_path.replace('_hemi-L_', '_hemi-R_'))
                    img = surface.SurfaceImage(mesh, data)

                    run = masker.fit_transform(img, **kwargs)
                    run = masker.inverse_transform(run)
                    for hemi in ('left', 'right'):
                        if hemi == 'left':
                            desc_hemi = '_hemi-L_'
                        else:
                            desc_hemi = '_hemi-R_'
                        img_path_hemi = func_file.replace('_hemi-L_', desc_hemi)
                        run_path = os.path.join(
                            out_dir, img_path_hemi.replace('_bold.func.gii', '_%s_bold.func.gii' % desc)
                        )
                        run.data.to_filename(run_path)
                else:
                    raise ValueError('Unknown space: %s' % space)
