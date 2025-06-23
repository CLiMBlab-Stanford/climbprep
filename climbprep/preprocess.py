import os
import yaml
import argparse

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run fMRIprep on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", "evlab", '
                                                                        'etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default='main', help=('Keyword (currently `main`) '
        'or YAML config file to used to parameterize preprocessing. If a keyword is provided, will '
        'the default settings for associated with that keyword. '
        'The possible config fields and values are just the `fmriprep` command-line arguments and their possible'
        'values. For details, see the `fmriprep` documentation.'))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)

    config = args.config
    if config in DEFAULTS['preprocess']:
        config_default = DEFAULTS['preprocess'][config]
        config = {}
    else:
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        config_default = DEFAULTS['preprocess'][PREPROCESS_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    for key in config_default:
        if key not in config:
            config[key] = config_default[key]
    assert 'preprocessing_label' in config, 'Required field `preprocessing_label` not found in config. ' \
                                            'Please provide a valid config file or keyword.'
    preprocessing_label = config.pop('preprocessing_label')

    out_path = os.path.join(project_path, 'derivatives', 'fmriprep', preprocessing_label)
    work_path = os.path.join(WORK_PATH, project, participant)

    kwarg_strings = []
    for key in config:
        key_str = '--%s' % key.replace('_', '-')
        val =  config[key]
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

    print(cmd)
    os.system(cmd)
