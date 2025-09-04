import os
import yaml
import argparse
from tempfile import TemporaryDirectory

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run an interactive seed analysis on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default='main', help=('Config name (default: `fsnative`) '
        'or YAML config file to used to parameterize seed analysis. '
        'See `climbprep.constants.CONFIG["seed"]` for available config names and their settings.'))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)

    config = args.config
    if config in CONFIG['seed']:
        seed_label = config
        config_default = CONFIG['seed'][config]
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
    surface = config['surface']

    timeseries_path = os.path.join(project_path, 'derivatives', 'clean', cleaning_label, participant)
    session_paths = [x for x in os.listdir(timeseries_path) if x.startswith('ses-')]
    functionals = set()
    if session_paths:
        cleaning_config_path = os.path.join(timeseries_path, session_paths[0], 'config.yml')
        for session_path_ in session_paths:
            session_path = os.path.join(timeseries_path, session_path_)
            for x in os.listdir(session_path):
                if x.endswith('_desc-clean_bold.dtseries.nii') and match.match(x) and \
                        ((space == 'fsnative' and not SPACE_RE.match(x)) or
                            (space != 'fsnative' and SPACE_RE.match(x) and SPACE_RE.match(x).group(1) == space)):
                    functionals.add(os.path.join(timeseries_path, session_path, x))
    else:
        cleaning_config_path = os.path.join(timeseries_path, 'config.yml')
        for x in os.listdir(timeseries_path):
            if x.endswith('_desc-clean_bold.dtseries.nii') and match.match(x) and \
                    ((space == 'fsnative' and not SPACE_RE.match(x)) or
                        (space != 'fsnative' and SPACE_RE.match(x) and SPACE_RE.match(x).group(1) == space)):
                functionals.add(os.path.join(timeseries_path, x))
    with open(cleaning_config_path, 'r') as f:
        preprocessing_label = yaml.safe_load(f)['preprocessing_label']
    anat_path = get_preprocessed_anat_dir(project, participant, preprocessing_label=preprocessing_label)
    if space == 'fsnative':
        space_str = ''
    else:
        space_str = f'_space-{space}'

    with TemporaryDirectory() as tmp_dir:
        spec_path = os.path.join(tmp_dir, 'wb.spec')
        for hemi in ('LEFT', 'RIGHT'):
            if surface == 'inflated':
                surf_path = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, 'sourcedata',
                    'freesurfer', f'sub-{participant}', 'surf', f'{hemi[0].lower()}h.inflated'
                )
            else:
                surf_path = os.path.join(anat_path, f'sub-{participant}_hemi-{hemi[0]}_{surface}.surf.gii')
            cmd = f'wb_command -add-to-spec-file {spec_path} CORTEX_{hemi} {surf_path}'
            cmd = f'singularity exec {os.path.join(APPTAINER_PATH, "images", WB_IMG)} bash -c "{cmd}"'
            stderr(cmd + '\n\n')
            status = os.system(cmd)
            assert not status, 'Adding surface to spec file failed with exit status %s' % status

        dtseries_path = os.path.join(tmp_dir, 'merged.dtseries.nii')
        cmd = f'wb_command -cifti-merge {dtseries_path}'
        for functional in sorted(list(functionals)):
            cmd += f' -cifti {functional}'
        cmd = f'singularity exec {os.path.join(APPTAINER_PATH, "images", FMRIPREP_IMG)} bash -c "{cmd}"'
        stderr(cmd + '\n\n')
        status = os.system(cmd)
        assert not status, 'Merging CIFTIs failed with exit status %s' % status

        cmd = f'wb_command -add-to-spec-file {spec_path} CORTEX {dtseries_path}'
        cmd = f'singularity exec {os.path.join(APPTAINER_PATH, "images", WB_IMG)} bash -c "{cmd}"'
        stderr(cmd + '\n\n')
        status = os.system(cmd)
        assert not status, 'Adding dtseries to spec file failed with exit status %s' % status

        cmd = f'wb_view -no-splash -spec-load-all {spec_path}'
        cmd = f'singularity exec --nv {os.path.join(APPTAINER_PATH, "images", WB_IMG)} bash -c "{cmd}"'
        stderr(cmd + '\n\n')
        status = os.system(cmd)



