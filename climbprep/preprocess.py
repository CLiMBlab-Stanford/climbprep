import os
import yaml
import argparse
import pwd

from climbprep.constants import *
from climbprep.util import *

def pick_heap_mb(cfg):
    v = cfg.pop('node_max_old_space_mb', None)
    if v not in (None, ''):
        return str(int(v))
    for k in ('CLIMB_NODE_OLD_SPACE_MB', 'NODE_MAX_OLD_SPACE_MB'):
        ev = os.environ.get(k)
        if ev:
            return str(int(ev))
    mem_mb = None
    if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
        mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
    elif os.environ.get('SLURM_MEM_PER_CPU', '').isdigit():
        mpc = int(os.environ['SLURM_MEM_PER_CPU'])
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_CPUS_ON_NODE', '1')))
        mem_mb = mpc * cpus
    if mem_mb:
        return str(min(int(mem_mb * 0.75), 131072))  # cap 128 GiB
    return '65536'  # fallback 64 GiB

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Run fMRIprep on a participant')
    ap.add_argument('participant', help='BIDS participant ID')
    ap.add_argument('-p', '--project', default='climblab', help='Name of BIDS project. Default: "climblab"')
    ap.add_argument('-c', '--config', default='main',
                    help='Config name (default: `main`) or YAML file ending in `_preprocess.yml`.')
    args = ap.parse_args()

    participant = args.participant.replace('sub-', '')
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)

    config = args.config
    if config in CONFIG['preprocess']:
        preprocessing_label = config
        config_default = CONFIG['preprocess'][config]
        config = {}
    else:
        assert config.endswith('_preprocess.yml'), 'config must end with `_preprocess.yml`'
        assert os.path.exists(config), f'Provided config ({config}) does not exist.'
        preprocessing_label = os.path.basename(config)[:-15]
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

    # Heap for Node (bids-validator)
    heap_mb = pick_heap_mb(config)
    stderr(f'Node/V8 heap target (MB): {heap_mb}\n')

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

    # Container paths and binds
    img = os.path.join(APPTAINER_PATH, 'images', FMRIPREP_IMG)
    tf_host = os.path.join(LAB_PATH, 'templateflow')
    user = os.environ.get('USER') or pwd.getpwuid(os.getuid()).pw_name
    # Force historical in-container TF path
    tf_cont = os.path.join('/sailhome', user, '.cache', 'templateflow')

    binds = [
        f'{LAB_PATH}:{LAB_PATH}',   # lab root
        f'{tf_host}:{tf_cont}',     # map lab TF → historical path inside container
    ]
    bind_str = ','.join(binds)

    inner = f'umask 002; fmriprep {" ".join(args_cli)}'

    # Precheck: ensure TF path exists and show effective V8 heap
    precheck = (
        'singularity exec --cleanenv --no-home '
        f'--bind {bind_str} '
        f'--env TEMPLATEFLOW_HOME="{tf_cont}" '
        f'--env NODE_OPTIONS="--max-old-space-size={heap_mb}" '
        f'--env MPLCONFIGDIR="/tmp/mpl-{user}" '
        f'{img} bash -lc \'mkdir -p "{tf_cont}"; node -p "v8.getHeapStatistics().heap_size_limit/1024/1024"\''
    )
    stderr(precheck + '\n')
    os.system(precheck)

    # Run fMRIPrep
    cmd = (
        'singularity exec --cleanenv --no-home '
        f'--bind {bind_str} '
        f'--env TEMPLATEFLOW_HOME="{tf_cont}" '
        f'--env NODE_OPTIONS="--max-old-space-size={heap_mb}" '
        f'--env MPLCONFIGDIR="/tmp/mpl-{user}" '
        f'{img} /bin/bash --noprofile --norc -lc \'{inner}\''
    )
    stderr(cmd + '\n')
    os.system(cmd)
