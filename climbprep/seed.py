import os
import shutil
import yaml
import argparse
from tempfile import TemporaryDirectory
from nilearn import surface

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run an interactive seed analysis on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default='fsnative', help=('Config name (default: `fsnative`) '
        'or YAML config file to used to parameterize seed analysis. '
        'See `climbprep.constants.CONFIG["seed"]` for available config names and their settings.'))
    argparser.add_argument('-i', '--interactive', action='store_true', help=("Launch wb_view interactively. Otherwise "
                                                                             "just zip all files into an archive "
                                                                             "for download, including a *.spec file "
                                                                             "for easy loading into wb_view."))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)
    interactive = args.interactive

    config = args.config
    if config in CONFIG['seed']:
        seed_label = config
        config_default = CONFIG['seed'][config]
        config = {}
    elif SMOOTHING_RE.match(config):
        config, fwhm = SMOOTHING_RE.match(config).groups()
        seed_label = f'{config}{fwhm}mm'
        assert config in CONFIG['seed'], 'Provided config (%s) does not match any known keyword.' % config
        config_default = CONFIG['seed'][config]
        if not config_default['cleaning_label'].endswith(f'{fwhm}mm'):
            config_default['cleaning_label'] = f'{config_default["cleaning_label"]}{fwhm}mm'
        config = {}
    else:
        assert config.endswith('_seed.yml') , \
                'config must either by a known keyword or a file ending in `_seed.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        seed_label = os.path.basename(config)[:-9]
        config_default = CONFIG['seed'][PREPROCESS_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    for key in config_default:
        if key not in config:
            config[key] = config_default[key]
    cleaning_label = config['cleaning_label']
    space = config['space']
    match = re.compile(config['regex_filter'])

    timeseries_path = os.path.join(project_path, 'derivatives', 'clean', cleaning_label, f'sub-{participant}')
    session_paths = [x for x in os.listdir(timeseries_path) if x.startswith('ses-')]
    functionals = set()
    if session_paths:
        for session_path_ in session_paths:
            session_path = os.path.join(timeseries_path, session_path_)
            for x in os.listdir(session_path):
                if x.endswith('_desc-clean_bold.func.gii') and match.match(x) and \
                        (HEMI_RE.match(x) and HEMI_RE.match(x).group(1) == 'L') and \
                        (SPACE_RE.match(x) and SPACE_RE.match(x).group(1) == space):
                    functionals.add(os.path.join(timeseries_path, session_path, x))
    else:
        for x in os.listdir(timeseries_path):
            if x.endswith('_desc-clean_bold.func.gii') and match.match(x) and \
                    (HEMI_RE.match(x) and HEMI_RE.match(x).group(1) == 'L') and \
                    (SPACE_RE.match(x) and SPACE_RE.match(x).group(1) == space):
                functionals.add(os.path.join(timeseries_path, x))
    assert len(functionals), 'No functional files found for participant %s with config %s. ' \
                             'Common causes include a misspecified regex_filter (%s) or space (%s) ' \
                             'in the config, or trying to run `seed` before `clean`. Make sure that ' \
                             '`clean` has run to completion, double-check your regex_filter, and ensure ' \
                             'that you have requested a surface (not volumetric) space (seed analysis of' \
                             'volumetric data is not currently supported).' % (
                                    participant, seed_label, config['regex_filter'], space
                             )
    with open(list(functionals)[0].replace('.func.gii', '.json'), 'r') as f:
        preprocessing_label = yaml.safe_load(f)['CleanParameters']['preprocessing_label']
    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label=preprocessing_label)
    if os.path.basename(os.path.dirname(anat_path)).startswith('ses-'):
        ses_str_anat = f'_ses-{os.path.basename(os.path.dirname(anat_path))[4:]}'
    else:
        ses_str_anat = ''
    if space == 'fsnative':
        space_str = ''
    else:
        space_str = f'_space-{space}'

    with TemporaryDirectory() as tmp_dir:
        spec_path = os.path.join(tmp_dir, 'wb.spec')
        for surf in ('pial', 'white', 'midthickness', 'inflated', 'sulc'):
            for hemi in ('LEFT', 'RIGHT'):
                if surf == 'inflated':
                    surf_path = os.path.join(
                        BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, 'sourcedata',
                        'freesurfer', f'sub-{participant}', 'surf', f'{hemi[0].lower()}h.inflated'
                    )
                    mesh = surface.PolyMesh(**{hemi.lower(): surf_path})
                    surf_path = os.path.join(tmp_dir, f'inflated_hemi-{hemi[0]}.surf.gii')
                    mesh.to_filename(surf_path)
                else:
                    if surf == 'sulc':
                        suffix = '.shape.gii'
                    else:
                        suffix = '.surf.gii'
                    surf_path_ = os.path.join(anat_path, f'sub-{participant}{ses_str_anat}_hemi-{hemi[0]}_{surf}{suffix}')
                    surf_path = os.path.join(tmp_dir, os.path.basename(surf_path_))
                    shutil.copy2(surf_path_, surf_path)
                cmd = f'wb_command -add-to-spec-file {spec_path} CORTEX_{hemi} {surf_path}'
                stderr(cmd + '\n\n')
                status = os.system(cmd)
                assert not status, 'Adding surf to spec file failed with exit status %s' % status

        with TemporaryDirectory() as tmp_dir_:
            dtseries_path = os.path.join(tmp_dir, 'merged.dtseries.nii')
            cmd = f'wb_command -cifti-merge {dtseries_path}'
            for i, functional in enumerate(sorted(list(functionals))):
                out_path = os.path.basename(functional).replace('_hemi-L', '').replace('.func.gii', '.dtseries.nii')
                out_path = os.path.join(tmp_dir_, out_path)
                left_path = functional
                sidecar_path = functional.replace('_bold.func.gii', '_bold.json')
                assert os.path.exists(sidecar_path), f'Sidecar file {sidecar_path} not found'
                with open(sidecar_path, 'r') as f:
                    sidecar = yaml.safe_load(f)
                assert 'RepetitionTime' in sidecar, f'RepetitionTime not found in {sidecar_path}'
                TR = sidecar['RepetitionTime']
                assert 'StartTime' in sidecar, f'StartTime not found in {sidecar_path}'
                StartTime = sidecar['StartTime']
                right_path = functional.replace('_hemi-L', '_hemi-R')
                cmd_ = f'wb_command -cifti-create-dense-timeseries {out_path} ' \
                                  f'-left-metric {left_path} -right-metric {right_path} ' \
                                  f'-timestep {TR} -timestart {StartTime}'
                stderr(cmd_ + '\n\n')
                status = os.system(cmd_)
                assert not status, f'Creating CIFTI {out_path} failed with exit status {status}'
                cmd += f' -cifti {out_path}'
            stderr(cmd + '\n\n')
            status = os.system(cmd)
            assert not status, 'Merging CIFTIs failed with exit status %s' % status

        cmd = f'wb_command -add-to-spec-file {spec_path} CORTEX {dtseries_path}'
        stderr(cmd + '\n\n')
        status = os.system(cmd)
        assert not status, 'Adding dtseries to spec file failed with exit status %s' % status

        if interactive:
            cmd = f'wb_view -no-splash -spec-load-all {spec_path}'
            stderr(cmd + '\n\n')
            status = os.system(cmd)
        else:
            out_dir = os.path.join(BIDS_PATH, project, 'derivatives', 'seed', seed_label, f'sub-{participant}')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, f'sub-{participant}_seed.zip')
            cmd = f'zip -FSrj {out_path} {tmp_dir}'
            stderr(cmd + '\n\n')
            status = os.system(cmd)



