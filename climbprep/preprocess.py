import os
import yaml
import argparse
import pwd

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run fMRIprep on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab',
                           help='Name of BIDS project. Default: "climblab"')
    argparser.add_argument('-c', '--config', default='main',
                           help=('Config name (default: `main`) or YAML file ending in '
                                 '`_preprocess.yml` containing fMRIPrep CLI args.'))
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
        assert config.endswith('_preprocess.yml'), \
            'config must be a known keyword or a file ending in `_preprocess.yml`'
        assert os.path.exists(config), \
            f'Provided config ({config}) is neither a known keyword nor an existing file.'
        preprocessing_label = os.path.basename(config)[:-15]
        config_default = CONFIG['preprocess'][PREPROCESS_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    # fill unspecified keys from defaults
    for key in config_default:
        if key not in config:
            config[key] = config_default[key]

    out_path = os.path.join(project_path, 'derivatives', 'preprocess', preprocessing_label)
    work_path = os.path.join(WORK_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}')

    stderr(f'Preprocessing outputs will be written to {out_path}\n')
    stderr(f'Working directory will be {work_path}\n')

    # Node/V8 heap for in-container bids-validator (MB). Default 24 GiB.
    heap_mb = str(config.pop('node_max_old_space_mb', 24576))

    # Build fMRIPrep CLI args
    kwarg_strings = []
    for key in list(config.keys()):
        key_str = '--%s' % key.replace('_', '-')
        val = config[key]
        if key == 'submm_recon':
            if not val:
                kwarg_strings.append('--no-submm-recon')
        elif key == 'use_syn_sdc':
            if val:
                kwarg_strings.append('--use-syn-sdc')
        elif key == 'skip_bids_validation':
            if val:
                kwarg_strings.append('--skip-bids-validation')
        else:
            if isinstance(val, (list, tuple)):
                val = ' '.join(val)
            kwarg_strings.append(f'{key_str} {val}')

    args_cli = [
        project_path,
        out_path,
        'participant',
        f'--work-dir {work_path}',
        f'--participant-label {participant}',
    ] + kwarg_strings

    # Container paths and binds (no SBATCH env required)
    img = os.path.join(APPTAINER_PATH, 'images', FMRIPREP_IMG)
    tf_host = os.path.join(LAB_PATH, 'templateflow')
    user = os.environ.get('USER') or pwd.getpwuid(os.getuid()).pw_name
    tf_cont = os.environ.get('TEMPLATEFLOW_HOME') or os.path.join('/sailhome', user, '.cache', 'templateflow')

    binds = [
        f'{LAB_PATH}:{LAB_PATH}',          # lab root
        f'{tf_host}:{tf_cont}',            # TemplateFlow cache path expected by older pickles
    ]
    bind_str = ','.join(binds)

    inner = f'umask 002; fmriprep {" ".join(args_cli)}'

    # Optional sanity print of effective V8 heap (MB) inside the container
    precheck = (
        'singularity exec --cleanenv --no-home '
        f'--bind {bind_str} '
        f'--env TEMPLATEFLOW_HOME="{tf_cont}" '
        f'--env NODE_OPTIONS="--max-old-space-size={heap_mb}" '
        f'{img} node -p "v8.getHeapStatistics().heap_size_limit/1024/1024"'
    )
    stderr(precheck + '\n')
    os.system(precheck)

    # Run fMRIPrep with clean env, no HOME, explicit binds, TF env, and raised Node heap
    cmd = (
        'singularity exec --cleanenv --no-home '
        f'--bind {bind_str} '
        f'--env TEMPLATEFLOW_HOME="{tf_cont}" '
        f'--env NODE_OPTIONS="--max-old-space-size={heap_mb}" '
        f'{img} /bin/bash --noprofile --norc -lc \'{inner}\''
    )

    stderr(cmd + '\n')
    os.system(cmd)
