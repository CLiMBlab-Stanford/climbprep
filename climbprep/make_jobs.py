import sys
import os
import argparse

from climbprep.constants import *

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dgb
#SBATCH --ntasks=%d
"""

JOB_TYPES = [
    'bidsify',
    'preprocess',
    'clean'
]

JOB_ORDER = [
    'bidsify',
    'preprocess',
    'clean',
    'firstlevels'
]

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run climbprep jobs
    ''')
    argparser.add_argument('participants', nargs='*', help='ID(s) of participant(s).')
    argparser.add_argument('-p', '--project', default='climblab', help='BIDS project name (default `climblab`)')
    argparser.add_argument('-j', '--job-types', nargs='+', default=JOB_TYPES, help=('Type(s) of job to run. One or '
        'more of ``["bidsify", "preprocess", "clean"]``'))
    argparser.add_argument('-t', '--time', type=int, default=72, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n-cores', type=int, default=8, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--partition', default='john', help=('Value for SLURM --partition setting, if '
                                                                      'applicable'))
    argparser.add_argument('-a', '--account', default='nlp', help='Value for SLURM --account setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts.')
    argparser.add_argument('--bidsify-config', default=None, help=('BIDSify config path. If not '
                                                                   'specified, the default config will be used.'))
    argparser.add_argument('--preprocess-config', default=None, help=('Preprocessing config (path or keyword). If not '
                                                                      'specified, the default config will be used.'))
    argparser.add_argument('--clean-config', default=None, help=('Cleaning config (path or keyword). If not '
                                                                 'specified, the default config will be used.'))
    args = argparser.parse_args()

    participants = args.participants
    project = args.project
    if not participants:
        participants = [x for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
    participants = [x.replace('sub-', '') for x in participants]
    job_types = args.job_types
    time = args.time
    n_cores = args.n_cores
    memory = args.memory
    partition = args.partition
    account = args.account
    if args.exclude:
        exclude = ','.join(args.exclude)
    else:
        exclude = []
    outdir = args.outdir
    if args.bidsify_config:
        bidsify_config_str = ' -c %s' % args.bidsify_config
    else:
        bidsify_config_str = ''
    if args.preprocess_config:
        preprocess_config_str = ' -c %s' % args.preprocess_config
    else:
        preprocess_config_str = ''
    if args.clean_config:
        clean_config_str = ' -c %s' % args.clean_config
    else:
        clean_config_str = ''

    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    for participant in participants:
        job_name = '_'.join([project, participant, '-'.join(job_types)])
        filename = os.path.normpath(os.path.join(outdir, job_name + '.pbs'))
        with open(filename, 'w') as f:
            f.write(base % (job_name, job_name, time, memory, n_cores))
            if partition:
                f.write('#SBATCH --partition=%s\n' % partition)
            if account:
                f.write('#SBATCH --account=%s\n' % account)
            if exclude:
                f.write('#SBATCH --exclude=%s\n' % exclude)
            wrapper = '%s'
            for job_type in JOB_ORDER:
                if job_type not in job_types:
                    continue
                if job_type.lower() == 'bidsify':
                    job_str = wrapper % ('python -m climbprep.bidsify %s -p %s%s' %
                                         (participant, project, bidsify_config_str))
                elif job_type.lower() == 'preprocess':
                    job_str = wrapper % ('python -m climbprep.preprocess %s -p %s%s' %
                                         (participant, project, preprocess_config_str))
                elif job_type.lower() == 'clean':
                    job_str = wrapper % ('python -m climbprep.preprocess %s -p %s%s' %
                                         (participant, project, clean_config_str))
                else:
                    raise ValueError('Unrecognized job type: %s.' % job_type)
                f.write(job_str)

