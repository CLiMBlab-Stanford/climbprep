import gc
import json
import yaml
import numpy as np
import pandas as pd
from nilearn import image, surface, maskers, signal, interfaces, glm
from nilearn import datasets as nidatasets
import argparse

from climbprep.constants import *
from climbprep.util import *
from climbprep.core import get_geodesic_smoothing_weights, apply_geodesic_smoothing_weights

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
    argparser.add_argument('-s', '--sessions', nargs='+', help="BIDS session ID(s).")
    argparser.add_argument('-c', '--config', default=CLEAN_DEFAULT_KEY, help=('Config name (default `fc`) '
        'or YAML config file to used to parameterize cleaning. '
        'See `climbprep.constants.CONFIG["clean"]` for available config names and their settings. '))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    target_sessions = args.sessions
    if target_sessions:
        target_sessions = set(target_sessions)
    else:
        target_sessions = set()
    config = args.config
    if config in CONFIG['clean']:
        cleaning_label = config
        config_default = CONFIG['clean'][config]
        config = {}
    elif SMOOTHING_RE.match(config):
        config, fwhm = SMOOTHING_RE.match(config).groups()
        cleaning_label = f'{config}{fwhm}mm'
        assert config in CONFIG['clean'], 'Provided config (%s) does not match any known keyword.' % config
        config_default = CONFIG['clean'][config]
        config_default['volume_fwhm'] = float(fwhm)
        config_default['surface_fwhm'] = float(fwhm)
        config = {}
    else:
        assert config.endswith('_clean.yml'), 'config must either be a known keyword or a file ending in ' \
                '_clean.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        cleaning_label = os.path.basename(config)[:-10]
        config_default = CONFIG['clean'][CLEAN_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'preprocessing_label' in config, 'Required field `preprocessing_label` not found in config. ' \
                                            'Please provide a valid config file or keyword.'
    preprocessing_label = config['preprocessing_label']
    clean_surf = config['clean_surf']
    min_T = config['min_T']
    mask_suffix = config.get('mask_suffix', DEFAULT_MASK_SUFFIX)
    mask_fwhm = config.get('mask_fwhm', DEFAULT_MASK_FWHM)
    target_affine = config.get('target_affine', DEFAULT_TARGET_AFFINE)

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path

    stderr(f'Cleaning outputs will be written to {os.path.join(derivatives_path, "clean", cleaning_label)}\n')

    sessions = set()
    for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}
    for session in sorted(list(sessions)):
        if target_sessions and not session in target_sessions:
            continue
        # Set session-dependent paths
        subdir = 'sub-%s' % participant
        participant_dir = subdir
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        preprocess_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label, subdir)
        assert os.path.exists(preprocess_path), 'Path not found: %s' % preprocess_path
        bids_path = os.path.join(project_path, subdir)
        assert os.path.exists(bids_path), 'Path not found: %s' % bids_path
        func_path = os.path.join(preprocess_path, 'func')
        assert os.path.exists(func_path), 'Path not found: %s' % func_path
        anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label=preprocessing_label)
        assert os.path.exists(anat_path), 'Path not found: %s' % anat_path

        if session:
            ses_str = f'_ses-{session}'
        else:
            ses_str = ''
        if os.path.basename(os.path.dirname(anat_path)).startswith('ses-'):
            ses_str_anat = f'_ses-{os.path.basename(os.path.dirname(anat_path))[4:]}'
        else:
            ses_str_anat = ''

        datasets = {}  # Structure: space > task > run > filetype (func, mask, confounds, tr) > value
        type_by_space = {}
        for img_path in sorted(list(os.listdir(func_path))):
            if img_path.endswith('desc-preproc_bold.nii.gz'):
                space = SPACE_RE.match(img_path)
                run = RUN_RE.match(img_path)
                task = TASK_RE.match(img_path)
                if space and task:
                    space = space.group(1)
                    type_by_space[space] = 'vol'
                    if run:
                        run = run.group(1)
                        run_str = '_run-%s' % run
                    else:
                        run = None
                        run_str = ''
                    task = task.group(1)
                    if space == 'T1w':
                        mask = f'sub-{participant}{ses_str_anat}{config.get("mask_suffix", DEFAULT_MASK_SUFFIX)}'
                    else:
                        mask = f'sub-{participant}{ses_str_anat}_space-{space}' \
                               f'{config.get("mask_suffix", DEFAULT_MASK_SUFFIX)}'
                    mask = os.path.join(anat_path, mask)
                    if not os.path.exists(mask):
                        if space == 'T1w':
                            mask = f'sub-{participant}{ses_str_anat}_label-GM_probseg.nii.gz'
                        else:
                            mask = f'sub-{participant}{ses_str_anat}_space-{space}_label-GM_probseg.nii.gz'
                        mask = os.path.join(anat_path, mask)
                    assert os.path.exists(mask), 'Mask file not found: %s' % mask
                    confounds = os.path.join(
                        func_path,
                        f'sub-{participant}{ses_str}_task-{task}{run_str}_desc-confounds_timeseries.tsv'
                    )
                    assert os.path.exists(confounds), 'Confounds file not found: %s' % confounds
                    func = os.path.join(func_path, img_path)
                    sidecar_path = func.replace('.nii.gz', '.json')
                    assert os.path.exists(sidecar_path), 'Path not found: %s' % sidecar_path
                    with open(sidecar_path, 'r') as f:
                        sidecar = json.load(f)
                    eventfile_path = os.path.join(
                        bids_path, 'func', img_path.split('_task-')[0] + '_task-' + task + run_str + '_events.tsv'
                    )
                    TR = sidecar.get('RepetitionTime', None)
                    StartTime = sidecar.get('StartTime', None)
                    assert TR, 'RepetitionTime information not found in sidecar: %s' % sidecar_path
                    assert StartTime, 'StartTime information not found in sidecar: %s' % sidecar_path
                    if space not in datasets:
                        datasets[space] = {}
                    if task not in datasets[space]:
                        datasets[space][task] = {}
                    if run not in datasets[space][task]:
                        datasets[space][task][run] = {}
                    datasets[space][task][run]['func'] = func
                    datasets[space][task][run]['mask'] = mask
                    datasets[space][task][run]['confounds'] = confounds
                    datasets[space][task][run]['eventfile_path'] = eventfile_path
                    datasets[space][task][run]['TR'] = TR
                    datasets[space][task][run]['StartTime'] = StartTime
            elif clean_surf and img_path.endswith('_bold.func.gii') and '_hemi-L_' in img_path:
                space = SPACE_RE.match(img_path)
                run = RUN_RE.match(img_path)
                task = TASK_RE.match(img_path)
                if space and task:
                    space = space.group(1)
                    type_by_space[space] = 'surf'
                    if run:
                        run = run.group(1)
                        run_str = '_run-%s' % run
                    else:
                        run = None
                        run_str = ''
                    task = task.group(1)
                    confounds = os.path.join(
                        func_path,
                        f'sub-{participant}{ses_str}_task-{task}{run_str}_desc-confounds_timeseries.tsv'
                    )
                    assert os.path.exists(confounds), 'Confounds file not found: %s' % confounds
                    confounds_sidecar = os.path.join(
                        func_path,
                        f'sub-{participant}{ses_str}_task-{task}{run_str}_desc-confounds_timeseries.json'
                    )
                    assert os.path.exists(confounds), 'Confounds sidecar file not found: %s' % confounds_sidecar
                    with open(confounds_sidecar, 'r') as f:
                        confounds_sidecar = json.load(f)
                    func = os.path.join(func_path, img_path)
                    sidecar_path = func.replace('.func.gii', '.json')
                    assert os.path.exists(sidecar_path), 'Path not found: %s' % sidecar_path
                    with open(sidecar_path, 'r') as f:
                        sidecar = json.load(f)
                    sidecar['CleanParameters'] = config
                    eventfile_path = os.path.join(
                        bids_path, 'func', img_path.split('_task-')[0] + '_task-' + task + run_str + '_events.tsv'
                    )
                    TR = sidecar.get('RepetitionTime', None)
                    StartTime = sidecar.get('StartTime', None)
                    assert TR, 'RepetitionTime information not found in sidecar: %s' % sidecar_path
                    assert StartTime, 'StartTime information not found in sidecar: %s' % sidecar_path
                    if space not in datasets:
                        datasets[space] = {}
                    if task not in datasets[space]:
                        datasets[space][task] = {}
                    if run not in datasets[space][task]:
                        datasets[space][task][run] = {}
                    datasets[space][task][run]['func'] = func
                    datasets[space][task][run]['mask'] = None
                    datasets[space][task][run]['confounds'] = confounds
                    datasets[space][task][run]['confounds_sidecar'] = confounds_sidecar.copy()
                    datasets[space][task][run]['eventfile_path'] = eventfile_path
                    datasets[space][task][run]['TR'] = TR
                    datasets[space][task][run]['StartTime'] = StartTime
                    datasets[space][task][run]['sidecar'] = sidecar

        out_dir = os.path.join(derivatives_path, 'clean', cleaning_label, subdir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for space in datasets:
            geodesic_smoothing_weights = None
            for task in datasets[space]:
                for run in datasets[space][task]:
                    confounds = datasets[space][task][run]['confounds']
                    confounds_sidecar = datasets[space][task][run]['confounds_sidecar']
                    eventfile_path = datasets[space][task][run]['eventfile_path']
                    mask = datasets[space][task][run]['mask']
                    func_path = datasets[space][task][run]['func']
                    func_file = os.path.basename(func_path)
                    TR = datasets[space][task][run]['TR']
                    StartTime = datasets[space][task][run]['StartTime']
                    sidecar = datasets[space][task][run]['sidecar']

                    stderr(f'Cleaning {func_file}\n')

                    confounds = pd.read_csv(confounds, sep='\t')
                    confounds = confounds.filter(
                        regex=r'{}'.format(config['confounds_regex'])
                    )
                    convolved = []
                    if config['regress_out_task'] and os.path.exists(eventfile_path):
                        events = pd.read_csv(eventfile_path, sep='\t')
                        if 'trial_type' not in events:
                            events['trial_type'] = 'dummy'
                        events = events[['trial_type', 'onset', 'duration']]
                        dummies = pd.get_dummies(events.trial_type, prefix='trial_type', prefix_sep='.')
                        events = pd.concat([dummies, events[['onset', 'duration']]], axis=1)
                        cols = [x for x in events.columns if x.startswith('trial_type')]
                        frame_times = np.arange(len(confounds)) * TR + StartTime
                        for col in cols:
                            convolved_, convolved_names = glm.first_level.compute_regressor(
                                (events['onset'], events['duration'], events[col]),
                                'spm',
                                frame_times
                            )
                            convolved_ = pd.DataFrame(convolved_, columns=[col])
                            convolved.append(convolved_)
                            confounds_sidecar[col] = dict(
                                Description='Task regressor for events of type %s' % col
                            )
                    for col in list(confounds_sidecar.keys()):
                        if col not in confounds.columns:
                            confounds_sidecar.pop(col)
                    confounds = pd.concat(convolved + [confounds], axis=1)

                    if type_by_space[space] == 'vol':  # Volumetric data
                        mask_nii = load_img(mask)
                        mask_nii = image.new_img_like(mask_nii, image.get_data(mask_nii).astype(np.float32))
                        if mask_fwhm:
                            mask_nii = image.smooth_img(mask_nii, fwhm=mask_fwhm)
                        if target_affine is not None:
                            mask_nii = image.resample_img(
                                mask_nii, target_affine=target_affine, interpolation='linear'
                            )
                        mask_nii = image.math_img('x > 0.', x=mask_nii)
                        mask_nii = image.crop_img(mask_nii)

                        func = load_img(func_path)
                        if not len(func.shape) > 3 or func.shape[3] < min_T:
                            continue
                        func = image.resample_to_img(func, mask_nii)

                        masker = maskers.NiftiMasker(
                            mask_img=None,
                            mask_strategy='background',
                            standardize=config['standardize'],
                            detrend=config['detrend'],
                            t_r=TR,
                            smoothing_fwhm=config['volume_fwhm'],
                            low_pass=config['low_pass'],
                            high_pass=config['high_pass']
                        )
                        kwargs = dict(confounds=confounds.fillna(0))
                        desc = 'desc-clean'
                        run = masker.fit_transform(func_path, **kwargs)
                        run = masker.inverse_transform(run)
                        run_path = os.path.join(
                            out_dir, func_file.replace('desc-preproc', desc)
                        )
                        run.to_filename(run_path)
                        confounds_outpath = os.path.join(
                            out_dir, func_file.replace('_hemi-L', '').replace('_space-%s' % space, '').
                                replace('desc-preproc_bold.nii.gz', 'desc-confounds_timeseries.tsv')
                        )
                        sidecar_outpath = os.path.join(
                            out_dir, run_path.replace('_bold.nii.gz', '_bold.json')
                        )
                        with open(sidecar_outpath, 'w') as f:
                            json.dump(sidecar, f, indent=2)
                    elif clean_surf and type_by_space[space] == 'surf':  # Surface data
                        mask_nii = None
                        if space == 'fsnative':
                            space_str = ''
                        else:
                            space_str = '_space-%s' % space
                        surf_L_path = os.path.join(
                            anat_path, f'sub-{participant}{ses_str_anat}{space_str}_hemi-L_pial.surf.gii'
                        )
                        surf_R_path = surf_L_path.replace('_hemi-L_', '_hemi-R_')

                        masker = maskers.SurfaceMasker(
                            mask_img=None,
                            standardize=config['standardize'],
                            detrend=config['detrend'],
                            t_r=TR,
                            low_pass=config['low_pass'],
                            high_pass=config['high_pass']
                        )
                        kwargs = dict(confounds=confounds.fillna(0))
                        desc = 'desc-clean'
                        if space == 'fsnative':
                            mesh = surface.PolyMesh(left=surf_L_path, right=surf_R_path)
                        elif space == 'fsaverage':
                            fsaverage = nidatasets.fetch_surf_fsaverage(mesh='fsaverage')
                            mesh = surface.PolyMesh(
                                left=fsaverage['pial_left'],
                                right=fsaverage['pial_right']
                            )
                        else:
                            raise ValueError('Unknown space: %s' % space)
                        if config['surface_fwhm'] and geodesic_smoothing_weights is None:
                            geodesic_smoothing_weights = {}
                            for hemi in ('left', 'right'):
                                geodesic_smoothing_weights[hemi] = get_geodesic_smoothing_weights(
                                    mesh.parts[hemi].faces, mesh.parts[hemi].coordinates, fwhm=config['surface_fwhm']
                                )
                        data = surface.PolyData(left=func_path, right=func_path.replace('_hemi-L_', '_hemi-R_'))
                        if not len(data.shape) > 1 or data.shape[1] < min_T:
                            continue
                        img = surface.SurfaceImage(mesh, data)

                        run = masker.fit_transform(img, **kwargs)
                        run = masker.inverse_transform(run)
                        for hemi in ('left', 'right'):
                            if geodesic_smoothing_weights is not None:
                                run.data.parts[hemi] = apply_geodesic_smoothing_weights(
                                    run.data.parts[hemi], geodesic_smoothing_weights[hemi]
                                )
                            if hemi == 'left':
                                desc_hemi = '_hemi-L_'
                            else:
                                desc_hemi = '_hemi-R_'
                            img_path_hemi = func_file.replace('_hemi-L_', desc_hemi)
                            run_path = os.path.join(
                                out_dir, img_path_hemi.replace('_bold.func.gii', '_%s_bold.func.gii' % desc)
                            )
                            run.data.to_filename(run_path)
                            sidecar_outpath = os.path.join(
                                out_dir, run_path.replace('_bold.func.gii', '_bold.json')
                            )
                            with open(sidecar_outpath, 'w') as f:
                                json.dump(sidecar, f, indent=2)
                        confounds_outpath = os.path.join(
                            out_dir, func_file.replace('_hemi-L', '').replace('_space-%s' % space, '').
                                replace('_bold.func.gii', '_desc-confounds_timeseries.tsv')
                        )
                    else:
                        raise ValueError('Unknown space: %s' % space)

                    confounds.to_csv(confounds_outpath, sep='\t', index=False)
                    confounds_sidecar_outpath = confounds_outpath.replace('.tsv', '.json')
                    with open(confounds_sidecar_outpath, 'w') as f:
                        json.dump(confounds_sidecar, f, indent=2)

                    gc.collect()
