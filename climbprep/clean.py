import os
import re
import json
import yaml
import pandas as pd
from nilearn import image, surface, maskers, signal, interfaces
import argparse

SPACE = re.compile('.+_space-([a-zA-Z0-9]+)_')
RUN = re.compile('.+_run-([0-9]+)_')

from climbprep.constants import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run fMRIprep on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default='firstlevels', help=('Keyword (currently `firstlevels` or '
        '`fc`) or YAML config file to used to parameterize cleaning. If a keyword is provided, will use the default '
        'settings for that type of downstream analysis. If not provided, will use default settings. '
        'To see the list of possible values and their defaults, run this script without using this '
        'argument and look at the `config.yml` file that is automatically generated in the output '
        'directory (the `derivatives/cleaning` directory of the relevant BIDS project).'))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    config = args.config
    if config == 'firstlevels':
        config_default = DEFAULTS['clean']['firstlevels']
        config = {}
    elif config == 'fc':
        config_default = DEFAULTS['clean']['fc']
        config = {}
    else:
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        config_default = DEFAULTS['clean']['firstlevels']
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

        volumes_by_space = {}
        masks_by_space = {}
        surfaces_by_space = {}
        TRs_by_file = {}
        confounds_by_run = {}
        runs = set()
        for img_path in os.listdir(func_path):
            if img_path.endswith('desc-preproc_bold.nii.gz'):
                space = SPACE.match(img_path)
                run = RUN.match(img_path)
                if space and run:
                    space = space.group(1)
                    run = run.group(1)
                    runs.add(run)
                    if space not in volumes_by_space:
                        volumes_by_space[space] = []
                        if space == 'T1w':
                            mask_path = f'sub-{participant}{ses_str}_label-GM_probseg.nii.gz'
                        else:
                            mask_path = f'sub-{participant}{ses_str}_space-{space}_label-GM_probseg.nii.gz'
                        masks_by_space[space] = image.load_img(os.path.join(anat_path, mask_path))
                    if space == 'T1w':
                        confounds_by_run[run] = os.path.join(func_path, img_path)
                    volumes_by_space[space].append(img_path)
                    raw_sidecar_path = os.path.join(
                        raw_path, 'func', img_path.split('_run-')[0] + '_run-' + run + '_bold.json'
                    )
                    assert os.path.exists(raw_sidecar_path), 'Path not found: %s' % raw_sidecar_path
                    with open(raw_sidecar_path, 'r') as f:
                        raw_sidecar = json.load(f)
                    TR = raw_sidecar.get('RepetitionTime', None)
                    assert TR, 'RepetitionTime information not found in raw sidecar: %s' % raw_sidecar_path
                    TRs_by_file[img_path] = TR
            elif img_path.endswith('_bold.func.gii') and '_hemi-L_' in img_path:
                space = SPACE.match(img_path)
                run = RUN.match(img_path)
                run = run.group(1)
                runs.add(run)
                if space and run:
                    space = space.group(1)
                    if space not in surfaces_by_space:
                        surfaces_by_space[space] = []
                    surfaces_by_space[space].append(img_path)
                    raw_sidecar_path = os.path.join(
                        raw_path, 'func', img_path.split('_run-')[0] + '_run-' + run + '_bold.json'
                    )
                    assert os.path.exists(raw_sidecar_path), 'Path not found: %s' % raw_sidecar_path
                    with open(raw_sidecar_path, 'r') as f:
                        raw_sidecar = json.load(f)
                    TR = raw_sidecar.get('RepetitionTime', None)
                    assert TR, 'RepetitionTime information not found in raw sidecar: %s' % raw_sidecar_path
                    TRs_by_file[img_path] = TR

        out_dir = os.path.join(derivatives_path, 'cleaned', cleaning_label, subdir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save the configuration used for cleaning
        config_path = os.path.join(out_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Volumetric data
        for space in volumes_by_space:
            img_paths = volumes_by_space[space]
            img_fullpaths = [os.path.join(func_path, x) for x in img_paths]
            confounds_imgs = [confounds_by_run[RUN.match(x).group(1)] for x in img_paths]
            confounds, sample_mask = interfaces.fmriprep.load_confounds(
                confounds_imgs,
                strategy=config['strategy'],
                std_dvars_threshold=config['std_dvars_threshold'],
                fd_threshold=config['fd_threshold']
            )

            for img_path, _sample_mask in zip(img_paths, sample_mask):
                _sample_mask = pd.DataFrame(dict(sample_mask=list(_sample_mask)))
                sample_mask_path = os.path.join(
                    out_dir, img_path.replace('_bold.nii.gz', '_samplemask.tsv')
                )
                _sample_mask.to_csv(sample_mask_path, sep='\t', index=False)

            mask_nii = image.load_img(masks_by_space[space])
            nii_ref = image.load_img(img_fullpaths[0])
            mask_nii = image.math_img(
                'img > 0.5',
                img=image.resample_to_img(
                    mask_nii, nii_ref, interpolation='nearest', force_resample=True, copy_header=True
                )
            )
            for i, (img_path, img_fullpath) in enumerate(zip(img_paths, img_fullpaths)):
                masker = maskers.NiftiMasker(
                    mask_img=mask_nii,
                    standardize=config['standardize'],
                    detrend=config['detrend'],
                    t_r=TRs_by_file[img_path],
                    low_pass=config['smoothing_fwhm'],
                    high_pass=config['high_pass']
                )
                kwargs = dict(confounds=confounds)
                if config['scrub']:
                    kwargs['sample_mask'] = sample_mask[i]
                    desc = 'desc-cleanscrubbed'
                else:
                    desc = 'desc-clean'
                run = masker.fit_transform(img_fullpath, **kwargs)
                run = masker.inverse_transform(run)
                run_path = os.path.join(
                   out_dir, img_path.replace('desc-preproc', desc)
                )
                run.to_filename(run_path)

        # Surface data
        for space in surfaces_by_space:
            if space == 'fsnative':
                space_str = ''
            else:
                space_str = '_space-%s' % space
            surf_L_path = os.path.join(
                fmriprep_path, 'anat', f'sub-{participant}{ses_str}{space_str}_hemi-L_pial.surf.gii'
            )
            surf_R_path = surf_L_path.replace('_hemi-L_', '_hemi-R_')

            img_paths = surfaces_by_space[space]
            img_fullpaths = [os.path.join(func_path, x) for x in img_paths]
            confounds_imgs = [confounds_by_run[RUN.match(x).group(1)] for x in img_paths]
            confounds, sample_mask = interfaces.fmriprep.load_confounds(
                confounds_imgs,
                strategy=config['strategy'],
                std_dvars_threshold=config['std_dvars_threshold'],
                fd_threshold=config['fd_threshold']
            )

            for img_path, _sample_mask in zip(img_paths, sample_mask):
                _sample_mask = pd.DataFrame(dict(sample_mask=list(_sample_mask)))
                sample_mask_path = os.path.join(
                    out_dir, img_path.replace('_bold.func.gii', '_samplemask.tsv')
                )
                _sample_mask.to_csv(sample_mask_path, sep='\t', index=False)
                _sample_mask.to_csv(sample_mask_path.replace('_hemi-L_', '_hemi-R_'), sep='\t', index=False)

            for i, (img_path, img_fullpath) in enumerate(zip(img_paths, img_fullpaths)):
                masker = maskers.SurfaceMasker(
                    mask_img=None,
                    standardize=config['standardize'],
                    detrend=config['detrend'],
                    t_r=TRs_by_file[img_path],
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
                data = surface.PolyData(left=img_fullpath, right=img_fullpath.replace('_hemi-L_', '_hemi-R_'))
                img = surface.SurfaceImage(mesh, data)

                run = masker.fit_transform(img, **kwargs)
                run = masker.inverse_transform(run)
                for hemi in ('left', 'right'):
                    if hemi == 'left':
                        desc_hemi = '_hemi-L_'
                    else:
                        desc_hemi = '_hemi-R_'
                    img_path_hemi = img_path.replace('_hemi-L_', desc_hemi)
                    run_path = os.path.join(
                        out_dir, img_path_hemi.replace('_bold.func.gii', '_%s_bold.func.gii' % desc)
                    )
                    run.data.to_filename(run_path)