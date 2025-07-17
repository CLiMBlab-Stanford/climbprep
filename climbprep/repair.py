import os
import json
import sys
import shutil
from tempfile import TemporaryDirectory
import pandas as pd
import bids
from bids.modeling import BIDSStatsModelsGraph
import argparse

from climbprep.constants import *
from climbprep.util import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(('Utility to repair BIDS directories. Currently just updates '
                                         '`participants.tsv`.'))
    argparser.add_argument('-p', '--projects', nargs='+', default=[], help=('Name of BIDS project (e.g., "climblab", '
        '"evlab", etc.). If unspecified, do all projects'))
    args = argparser.parse_args()

    if not args.projects:
        projects = os.listdir(BIDS_PATH)
    else:
        projects = args.projects
    for project in projects:
        stderr('Repairing project %s\n' % project)
        project_path = os.path.join(BIDS_PATH, project)
        stderr('  Updating participants.tsv\n')
        participants_path = os.path.join(project_path, 'participants.tsv')
        participants = pd.read_csv(participants_path, sep='\t')
        participants.participant_id = participants.participant_id.apply(lambda x: x if x.startswith('sub-') else f'sub-{x}')
        participants_BIDS = [x for x in os.listdir(project_path) if x.startswith('sub-')]
        participants = participants[participants.participant_id.isin(participants_BIDS)]
        participants_found = set(participants.participant_id.tolist())
        participants_missing = pd.DataFrame(dict(participant_id=list(set(participants_BIDS) - participants_found)))
        participants = pd.concat([participants, participants_missing], axis=0)
        participants = participants.sort_values('participant_id')
        for col in participants:
            if col.lower().endswith('_id') and pd.api.types.is_float_dtype(participants[col]):
                participants[col] = participants[col].astype('Int64')
        tasks = []
        for participant in participants.participant_id.tolist():
            path = os.path.join(project_path, participant)
            sessions = []
            for _path in os.listdir(path):
                if _path.startswith('ses-'):
                    sessions.append(os.path.join(path, _path))
            if not sessions:
                sessions = [path]
            _tasks = set()
            for path in sessions:
                func_path = os.path.join(path, 'func')
                if not os.path.exists(func_path):
                    continue
                for func in os.listdir(func_path):
                    parts = func.split('_')
                    for part in parts:
                        if part.startswith('task-'):
                            _tasks.add(part[5:])
                            break
            tasks.append(','.join(sorted(list(_tasks))))
        participants['tasks'] = tasks
        participants.to_csv(participants_path, index=False, sep='\t')

        stderr('  Updating pybids index\n')
        database_path = os.path.join(project_path, 'code', 'pybids_dbcache')
        indexer = bids.BIDSLayoutIndexer()
        derivatives_path = os.path.join(project_path, 'derivatives', 'preprocess')
        derivatives = [
            os.path.join(derivatives_path, x) for x in os.listdir(derivatives_path) if not x.startswith('.') and os.path.isdir(x)
        ]
        layout = bids.BIDSLayout(
            project_path,
            derivatives=derivatives,
            database_path=database_path,
            reset_database=True,
            indexer=indexer
        )

