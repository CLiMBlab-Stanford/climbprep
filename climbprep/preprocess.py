import os
import sys
import yaml
import pwd
import argparse

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
        return str(min(int(mem_mb * 0.75), 131072))  # cap at 128 GiB
    return '65536'  # default 64 GiB

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Run fMRIPrep on a participant')
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

    # fill unspecified keys from defaults
    for key in config_default:
        if key not in config:
            config[key] = config_default[key]

    out_path = os.path.join(project_path, 'derivatives', 'preprocess', preprocessing_label)
    work_path = os.path.join(WORK_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}')

    stderr(f'Preprocessing outputs will be written to {out_path}\n')
    stderr(f'Working directory will be {work_path}\n')

    # Heap for host-side Node (bids-validator)
    heap_mb = pick_heap_mb(config)
    stderr(f'Node/V8 heap target for host validation (MB): {heap_mb}\n')

    # 1) Host-side BIDS validation (mirrors bidsify behavior)
    host_val_cmd = f'UMASK=002 NODE_OPTIONS="--max-old-space-size={heap_mb}" bids-validator "{project_path}"'
    # Use umask via shell so outputs are group-writable
    host_val_cmd = f'umask 002; NODE_OPTIONS="--max-old-space-size={heap_mb}" bids-validator "{project_path}"'
    stderr(host_val_cmd + '\n')
    status = os.system(host_val_cmd)
    if status != 0:
        stderr('Host BIDS validation failed. Aborting before fMRIPrep.\n')
        sys.exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)
    stderr('Host BIDS validation passed.\n')

    # 2) Build fMRIPrep CLI args, always skip in-container validation
    kwarg_strings = []
    for key in list(config.keys()):
        if key == 'submm_recon':
            if not config[key]:
                kwarg_strings.append('--no-submm-recon')
        elif key == 'use_syn_sdc':
            if config[key]:
                kwarg_strings.append('--use-syn-sdc')
        elif key == 'skip_bids_validation':
            # Ignore any user setting; we will force skipping below.
            continue
        else:
            key_str = '--%s' % key.replace('_', '-')
            val = config[key]
            if isinstance(val, (list, tuple)):
                val = ' '.join(val)
            kwarg_strings.append(f'{key_str} {val}')

    # Force skip inside container to avoid fixed Node heap there
    kwarg_strings.append('--skip-bids-validation')

    args_cli = [
        project_path,
        out_path,
        'participant',
        f'--work-dir {work_path}',
        f'--participant-label {participant}',
    ] + kwarg_strings

    # 3) Container paths and binds
    img = os.path.join(APPTAINER_PATH, 'images', FMRIPREP_IMG)
    tf_host = os.path.join(LAB_PATH, 'templateflow')
    user = os.environ.get('USER') or pwd.getpwuid(os.getuid()).pw_name
    tf_cont = os.path.join('/sailhome', user, '.cache', 'templateflow')  # historical in-container path

    binds = [
        f'{LAB_PATH}:{LAB_PATH}',       # lab root
        f'{tf_host}:{tf_cont}',         # map lab TF → historical path inside container
    ]
    bind_str = ','.join(binds)

    inner = f'umask 002; fmriprep {" ".join(args_cli)}'

    # 4) Run fMRIPrep with a clean container env, no in-container validation
    cmd = (
        'singularity exec --cleanenv --no-home '
        f'--bind {bind_str} '
        f'--env TEMPLATEFLOW_HOME="{tf_cont}" '
        f'{img} /bin/bash --noprofile --norc -lc \'{inner}\''
    )
    stderr('Skipping in-container BIDS validation (already validated on host).\n')
    stderr(cmd + '\n')
    os.system(cmd)
