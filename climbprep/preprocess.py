import os
import yaml
import argparse

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run fMRIprep on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default='main', help=('Config name (default: `main`) '
        'or YAML config file to used to parameterize preprocessing. '
        'See `climbprep.constants.CONFIG["preprocess"]` for available config names and their settings. '
        'The possible config fields and values are just the `fmriprep` command-line arguments and their possible'
        'values. For details, see the `fmriprep` documentation.'))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)

    config = args.config
    if config in CONFIG['preprocess']:
        preprocessing_label = config
        config_default = CONFIG['preprocess'][config]
        config = {}
    else:
        assert config.endswith('_preprocess.yml') , \
                'config must either by a known keyword or a file ending in `_preprocess.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        preprocessing_label = config[:-15]
        config_default = CONFIG['preprocess'][PREPROCESS_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    for key in config_default:
        if key not in config:
            config[key] = config_default[key]

    out_path = os.path.join(project_path, 'derivatives', 'preprocess', preprocessing_label)
    work_path = os.path.join(WORK_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}')

    stderr(f'Preprocessing outputs will be written to {out_path}\n')
    stderr(f'Working directory will be {work_path}\n')

    kwarg_strings = []
    for key in config:
        key_str = '--%s' % key.replace('_', '-')
        val =  config[key]
        if key == 'submm_recon':
            if not val:
                kwarg_strings.append('--no-submm-recon')
        else:
            if isinstance(val, list) or isinstance(val, tuple):
                val = ' '.join(val)
            kwarg_strings.append(f'{key_str} {val}')

    args = [
        project_path,
        out_path,
        'participant',
        f'--work-dir {work_path}',
        f'--participant-label {participant}'
    ] + kwarg_strings
    cmd = f'fmriprep {" ".join(args)}'
    cmd = f'singularity exec {FMRIPREP_IMG} bash -c "{cmd}"'

    stderr(cmd + '\n')
    os.system(cmd)
