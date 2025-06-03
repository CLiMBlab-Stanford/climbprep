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

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run climbprep jobs
    ''')
    argparser.add_argument('participants', nargs='*', help='ID(s) of participant(s).')
    argparser.add_argument('-p', '--project', default='climblab', help='BIDS project name (default `climblab`)')
    argparser.add_argument('-j', '--job_types', nargs='+', default=['preprocess'], help='Type of job to run. One of ``["bidsify", "preprocess"]``')
    argparser.add_argument('-t', '--time', type=int, default=72, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=8, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--partition', default='john', help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-a', '--account', default='nlp', help='Value for SLURM --account setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-c', '--cli_args', default='', help='Command line arguments to pass into call')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts.')
    argparser.add_argument('--session', default=None, help='Specific session to bidsify (ignored for other job types)')
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
    cli_args = args.cli_args.replace('\\', '') # Delete escape characters
    outdir = args.outdir
    session = args.session
    if session:
        session_str = ' -s %s' % session
    else:
        session_str = ''

    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    for participant in participants:
        job_name = '_'.join([project, participant, ''.join(job_types)])
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
            for job_type in job_types:
                if job_type.lower() == 'bidsify':
                    job_str = wrapper % ('python -m climbprep.bidsify %s -p %s%s %s' % (participant, project, session_str, cli_args))
                elif job_type.lower() == 'preprocess':
                    job_str = wrapper % ('python -m climbprep.preprocess %s -p %s %s' % (participant, project, cli_args))
                else:
                    raise ValueError('Unrecognized job type: %s.' % job_type)
                f.write(job_str)

