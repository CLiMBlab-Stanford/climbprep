import os
import json
import sys
import shutil
from tempfile import TemporaryDirectory
import pandas as pd
import argparse

from climbprep.constants import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(('Convenience utility to convert fMRI source data to BIDS. '
                                         'The source data may be of any structure readable by dcm2bids, '
                                         'and it may optionally contain a user-written CSV at the '
                                         'directory root called `runs.csv` with run-level information. '
                                         'The table must at minimum contain the columns '
                                         '"SeriesNumber" and "TaskName". It may optionally '
                                         'contain a column "EventsFile" with a path to a '
                                         'BIDS-style table of event conditions and timings. '
                                         'If EventsFile is absent but the run used a task, '
                                         'this may cause errors downstream during firstlevel '
                                         'modeling. If the task for a run is missing from the table '
                                         'or the table is missing altogether, the taskname '
                                         'will be `UnknownTask`.'))
    argparser.add_argument('participant', help="BIDS participant ID.")
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-t', '--tmp-dir', default=None, help=('Temporary directory to use for dcm2bids. '
                                                    'Created automatically if left blank. '
                                                    'This is useful for saving the outputs of '
                                                    'dcm2niix if you are debugging things.'))
    argparser.add_argument('-c', '--config', default=None, help=("Path to config (json) for dcm2bids. Generated "
                                                                 "if unused."))
    argparser.add_argument('-O', '--overwrite', action='store_true', help="Overwrite existing outputs.")
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    tmp_dir_ = args.tmp_dir
    config_path = args.config
    overwrite = args.overwrite

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, args.project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    sourcedata_path = os.path.join(project_path, 'sourcedata')
    assert os.path.exists(sourcedata_path), 'Path not found: %s' % sourcedata_path

    sessions = set()
    for subdir in os.listdir(os.path.join(sourcedata_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}
    for session in sessions:
        subdir = 'sub-%s' % participant
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        src_path = os.path.join(sourcedata_path, subdir)
        assert os.path.exists(src_path), 'Path not found: %s' % src_path
        run_table_path = os.path.join(src_path, 'runs.csv')
        out_path = os.path.join(project_path, subdir)

        if os.path.exists(run_table_path):
            run_table = pd.read_csv(run_table_path)
        else:
            run_table = pd.DataFrame()
        run_map = {}
        for row in run_table.to_dict('records'):
            ix = int(row['SeriesNumber'])
            run_map[ix] = dict()
            if 'TaskName' in row:
                run_map[ix]['TaskName'] = row['TaskName']
            if 'EventsFile' in row and row['EventsFile'] and not pd.isna(row['EventsFile']):
                run_map[ix]['EventsFile'] = os.path.join(src_path, row['EventsFile'])

        with TemporaryDirectory() as tmp_dir:
            if tmp_dir_ is not None:
                tmp_dir = tmp_dir_
            os.system('dcm2bids_helper -d %s -o %s' % (src_path, tmp_dir))
            descriptions = []
            for jsonpath in sorted(
                    [x for x in os.listdir(os.path.join(tmp_dir, 'tmp_dcm2bids', 'helper')) if x.endswith('.json')]
            ):
                series_number = int(jsonpath.split('_')[0])
                with open(os.path.join(tmp_dir, 'tmp_dcm2bids', 'helper', jsonpath), 'r') as f:
                    meta = json.load(f)
                if 'BidsGuess' not in meta:
                    continue
                dtype, suffix = meta['BidsGuess']
                suffix = suffix.split('_')[-1]
                if dtype != 'discard':
                    if dtype == 'func' and len(descriptions) and \
                            'AcquisitionTime' in meta and 'AcquisitionTime' in descriptions[-1]['criteria'] and \
                            meta['AcquisitionTime'] == descriptions[-1]['criteria']['AcquisitionTime']:
                        continue
                    description = dict(
                        datatype=dtype,
                        suffix=suffix,
                        criteria=dict(
                            SeriesNumber=meta['SeriesNumber'],
                            SeriesDescription=meta['SeriesDescription']
                        )
                    )
                    if 'AcquisitionTime' in meta:
                        description['criteria']['AcquisitionTime'] = meta['AcquisitionTime']
                    if dtype == 'func':
                        task_name = run_map.get(series_number, {}).get('TaskName', DEFAULT_TASK)
                        description['custom_entities'] = 'task-%s' % task_name
                        description['sidecar_changes'] = dict(TaskName=task_name)
                        events_file = run_map.get(series_number, {}).get('EventsFile', None)
                        if events_file:
                            description['sidecar_changes']['EventsFile'] = events_file
                    descriptions.append(description)

            descriptions = dict(descriptions=descriptions)

            if config_path:
                config_path = config_path
            else:
                config_path = os.path.join(src_path, 'dcm2bids_config.json')
                with open(config_path, 'w') as f:
                    json.dump(descriptions, f, indent=2)

            if overwrite:
                overwrite_str = ' --clobber'
            else:
                overwrite_str = ''
            if session:
                session_str = ' -s %s' % session
            else:
                session_str = ''
            os.system('dcm2bids -d %s -p %s%s -c %s -o %s --auto_extract_entities%s --skip_dcm2niix --force_dcm2bids' %
                      (os.path.join(tmp_dir, 'tmp_dcm2bids', 'helper'), participant, session_str, config_path,
                       project_path, overwrite_str))

            dirnames = ('anat', 'func', 'derived')
            for dirname in dirnames:
                dir_path = os.path.join(out_path, dirname)
                if not os.path.exists(dir_path):
                    continue
                for path in os.listdir(dir_path):
                    if dirname == 'anat':
                        filepath = os.path.join(dir_path, path)
                        if filepath.endswith('.nii.gz'):
                            filepath_tmp = filepath.replace('.nii.gz', '_tmp.nii.gz')
                            os.system('synthstrip-singularity -i %s -o %s' % (filepath, filepath_tmp))
                            shutil.move(filepath_tmp, filepath)
                    else:
                        if path.endswith('.json'):
                            filepath = os.path.join(dir_path, path)
                            with open(filepath, 'r') as f:
                                sidecar = json.load(f)
                            if 'EventsFile' in sidecar:
                                events_path_new = os.path.join(dir_path, path.replace('bold.json', 'events.tsv'))
                                if events_path_new != sidecar['EventsFile']:
                                    shutil.copy2(sidecar['EventsFile'], events_path_new)
                                    sidecar['EventsFile'] = events_path_new
                            with open(filepath, 'w') as f:
                                json.dump(sidecar, f, indent=2)

            os.system('bids-validator %s' % project_path)

