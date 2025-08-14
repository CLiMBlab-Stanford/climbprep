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
    'clean',
    'model',
    'plot',
    'parcellate'
]

JOB_ORDER = [
    'bidsify',
    'preprocess',
    'clean',
    'model',
    'plot',
    'parcellate'
]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run climbprep jobs
    ''')
    argparser.add_argument('participants', nargs='*', help='ID(s) of participant(s).')
    argparser.add_argument('-p', '--projects', nargs='+', default=['climblab'], help='BIDS project name(s) '
                                                                                     '(default `climblab`)')
    argparser.add_argument('-s', '--sessions', nargs='*', default=None, help='BIDS session name(s) (default `climblab`)')
    argparser.add_argument('-j', '--job-types', nargs='+', default=JOB_TYPES, help=('Type(s) of job to run. One or '
        'more of ``["bidsify", "preprocess", "clean", "model", "plot", "parcellate"]``'))
    argparser.add_argument('-t', '--time', type=int, default=72, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n-cores', type=int, default=8, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=128, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--partition', default='sphinx', help=('Value for SLURM --partition setting, if '
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
    argparser.add_argument('--model-config', default=None, help=('Modeling config (path or keyword). If not '
                                                                 'specified, the default config will be used.'))
    argparser.add_argument('--plot-config', default=None, help=('Plotting config (path or keyword). If not '
                                                                'specified, the default config will be used.'))
    argparser.add_argument('--parcellate-config', default=None, help=('Parcellation config (path or keyword). If not '
                                                                      'specified, the default config will be used.'))
    args = argparser.parse_args()

    projects = args.projects
    job_types = args.job_types
    sessions = args.sessions
    for job_type in job_types:
        assert not sessions or job_type not in ('preprocess', 'model', 'plot', 'parcellate'), \
                f'--sessions provided for a {job_type} job, which does not accept this argument'
    if not sessions:
        sessions = [None]
    for job_type in job_types:
        assert job_type in JOB_TYPES, 'Unrecognized job_type %s' % job_type
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
    if args.model_config:
        model_config_str = ' -c %s' % args.model_config
    else:
        model_config_str = ''
    if args.plot_config:
        plot_config_str = ' -c %s' % args.plot_config
    else:
        plot_config_str = ''
    if args.parcellate_config:
        parcellate_config_str = ' -c %s' % args.parcellate_config
    else:
        parcellate_config_str = ''

    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    for project in projects:
        sourcedata_path = os.path.join(BIDS_PATH, project, 'sourcedata')
        participants = args.participants
        if not participants:
            participants = set([x for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')])
            if os.path.exists(sourcedata_path):
                participants |= set([x for x in os.listdir(sourcedata_path) if x.startswith('sub-')])
        else:
            participants = set(participants)
        participants = [x.replace('sub-', '') for x in participants]
        if sessions == ['all']:
            sessions_ = []
            for participant_dir in [x for x in os.listdir(sourcedata_path) if x.startswith('sub-')]:
                participant = participant_dir[4:]
                if participant not in participants:
                    continue
                for session_dir in [
                        x for x in os.listdir(os.path.join(sourcedata_path, participant_dir)) if x.startswith('ses-')
                ]:
                    sessions_.append(session_dir[4:])
        else:
           sessions_ = sessions
        for session in sessions_:
            for participant in participants:
                has_this_session = False
                if not session:
                    has_this_session = True
                for path in os.listdir(os.path.join(sourcedata_path, f'sub-{participant}')):
                    if path == f'ses-{session}':
                        has_this_session = True
                if not has_this_session:
                    continue
                name_parts = [project, participant]
                if session:
                    name_parts.append(session)
                    session_str = ' -s %s' % session
                else:
                    session_str = ''
                name_parts.append('-'.join(job_types))
                job_name = '_'.join(name_parts)
                filename = os.path.normpath(os.path.join(outdir, job_name + '.pbs'))
                with open(filename, 'w') as f:
                    f.write(base % (job_name, job_name, time, memory, n_cores))
                    if partition:
                        f.write('#SBATCH --partition=%s\n' % partition)
                    if account:
                        f.write('#SBATCH --account=%s\n' % account)
                    if exclude:
                        f.write('#SBATCH --exclude=%s\n' % exclude)
                    f.write('\n\nset -e\n\n')
                    wrapper = '%s'
                    for job_type in JOB_ORDER:
                        if job_type not in job_types:
                            continue
                        if job_type.lower() == 'bidsify':
                            job_str = wrapper % ('python -m climbprep.bidsify %s -p %s%s%s\n' %
                                                 (participant, project, session_str, bidsify_config_str))
                        elif job_type.lower() == 'preprocess':
                            job_str = wrapper % ('python -m climbprep.preprocess %s -p %s%s\n' %
                                                 (participant, project, preprocess_config_str))
                        elif job_type.lower() == 'clean':
                            job_str = wrapper % ('python -m climbprep.clean %s -p %s%s%s\n' %
                                                 (participant, project, session_str, clean_config_str))
                        elif job_type.lower() == 'model':
                            job_str = wrapper % ('python -m climbprep.model %s -p %s%s\n' %
                                                 (participant, project, model_config_str))
                        elif job_type.lower() == 'plot':
                            job_str = wrapper % ('python -m climbprep.plot %s -p %s%s\n' %
                                                 (participant, project, plot_config_str))
                        elif job_type.lower() == 'parcellate':
                            job_str = wrapper % ('python -m climbprep.parcellate %s -p %s%s\n' %
                                                 (participant, project, parcellate_config_str))
                        else:
                            raise ValueError('Unrecognized job type: %s.' % job_type)
                        f.write(job_str)
    
