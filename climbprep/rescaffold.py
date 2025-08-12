import pandas as pd
import argparse
import shutil
import json

from climbprep.constants import *
from climbprep.util import stderr

if __name__ ==  '__main__':
    argparser = argparse.ArgumentParser('Make a new multisession BIDS dataset from a subset of sessions in a '
                                        'project with sessions as subjects (e.g., climblab and evlab).')
    argparser.add_argument('project',  help='Name of BIDS project to create from the source data.')
    argparser.add_argument('-p', '--source_project', default='climblab',  help='Name of BIDS project to rescaffold '
                                                                               '(default `climblab`.')
    argparser.add_argument('participants', nargs='*', help=('Space-delimited list of participants or a `*.tsv` table '
                                                            'containing a list of participants. IMPORTANT: the '
                                                            'BIDS `participant` field in the source directory '
                                                            'indexes the *session*, the the participant. Participants '
                                                            'are indexed via the `<SOURCE>_id` column of '
                                                            '`participants.tsv`. Values passed to this argument '
                                                            'should be valid source IDs or they will be ignored. '
                                                            'If this argument is empty, all source participants '
                                                            'will be included.'))
    argparser.add_argument('-T', '--tasks', nargs='*', help=('Space-delimited list of tasks to include or a `*.tsv` '
                                                              'table containing a list of tasks. If this argument is '
                                                              'empty, all tasks will be included.'))
    args = argparser.parse_args()

    project = args.project
    source = args.source_project

    # Infer relevant participants and sessions
    participants = set(args.participants) or None
    if participants:
        participants_ = set()
        for participant in participants:
            if participant.endswith('.tsv'):
                participants_df = pd.read_csv(participant, sep='\t', header=None, names=['participant_id'])
                participants_ |= set(participants_df['participant_id'].str.replace('^sub-', '', regex=True).tolist())
        participants = participants_

    tasks = args.tasks or None
    if tasks:
        tasks_ = set()
        for task in tasks:
            if task.endswith('.tsv'):
                tasks_df = pd.read_csv(task, sep='\t', header=None, names=['task'])
                tasks_ |= set(tasks_df['task'].tolist())
        tasks = tasks_

    stderr(f'Rescaffolded project will be written to {os.path.join(BIDS_PATH, project)}\n')

    source_project_path = os.path.join(BIDS_PATH, source)
    participants_table = pd.read_csv(os.path.join(source_project_path, 'participants.tsv'), sep='\t')
    participants_table.participant_id = participants_table.participant_id.str.replace('^sub-', '', regex=True)
    session_to_id = {x: y for x, y in zip(participants_table['participant_id'], participants_table[f'{source}_id'])}
    id_to_sessions = {}
    for session in session_to_id:
        source_id = session_to_id[session]
        if source_id not in id_to_sessions:
            id_to_sessions[source_id] = set()
        id_to_sessions[source_id].add(session)
    available_participants = set(id_to_sessions.keys())
    if participants:
        missing = participants - available_participants
        if missing:
            stderr((f'WARNING: The following participants were not found in the {source} dataset: '
                    f"{' '.join(sorted(list(missing)))}. "
                    f'These participants will be ignored. If you are receiving this message in error, either the '
                    f'participant has no BIDSified sessions or `{BIDS_PATH}/{source}/participants.tsv` has not '
                    f'been updated to include their `{source}_id`. Check to make sure (1) that their session data can '
                    f'be found at `{BIDS_PATH}/{source}/sub-<SESSION_ID> (if not, run `python -m climbprep.bidsify '
                    f'<SESSION_ID> -p {source}) and (2) that `{BIDS_PATH}/{source}/participants.tsv` has been updated '
                    f'to include the `{source}_id` for the relevant sessions (if not, run `python -m climbprep.repair` '
                    f'and then manually add their `{source}_id` to the relevant rows of `{BIDS_PATH}/{source}/'
                    f'participants.tsv`. If you do not know their `{source}_id`, contact Cory or another lab member '
                    f'who has access to the high-risk database.\n'))
        participants &= available_participants
    else:
        participants = available_participants
    sessions = set()
    for participant in participants:
        sessions |= id_to_sessions[participant]
    task_to_sessions = {}
    session_to_tasks = {}
    for session in sessions:
        func_path = os.path.join(source_project_path, f'sub-{session}', 'func')
        for func in os.listdir(func_path):
            task = TASK_RE.match(func)
            if task:
                task = task.group(1)
                if task not in task_to_sessions:
                    task_to_sessions[task] = set()
                task_to_sessions[task].add(session)
                if session not in session_to_tasks:
                    session_to_tasks[session] = set()
                session_to_tasks[session].add(task)
    available_tasks = set(task_to_sessions.keys())
    if tasks:
        missing = set(tasks) - available_tasks
        if missing:
            stderr((f'WARNING: The following tasks were not found in the {source} dataset: '
                    f'{", ".join(sorted(list(missing)))}. These tasks will be ignored. If you are receiving this '
                    f'message in error, make sure that (1) you have requested an exact string match to the desired '
                    f'task name, and (2) that the relevant session data can be found at '
                    f'`{BIDS_PATH}/{source}/sub-<SESSION_ID> (if not, run `python -m climbprep.bidsify '
                    f'<SESSION_ID>).\n'))
        tasks = set(tasks) & available_tasks
    else:
        tasks = available_tasks
    task_sessions = set()
    for task in tasks:
        task_sessions |= task_to_sessions[task]
    sessions = sessions & task_sessions
    task_participants = set()
    for session in sessions:
        task_participants.add(session_to_id[session])
    participants = participants & task_participants

    if not participants:
        stderr('No matching participants found. Exiting.\n')
        exit()

    if not sessions:
        stderr('No matching sessions found. Exiting.\n')
        exit()

    # Minimally update new project's metadata to pass validation
    project_path = os.path.join(BIDS_PATH, project)
    if not os.path.exists(project_path):
        stderr(f'Project path {project_path} not found, automatically scaffolding.\n')
        cmd = f"dcm2bids_scaffold -o {project_path}"
        print(cmd)
        status = os.system(cmd)
        if status:
            stderr('Error creating BIDS scaffold. Exiting.\n')
            exit(status)

    if not os.path.exists(os.path.join(BIDS_PATH, project, 'participants.tsv')):
        participants_tsv = []
        for participant in participants:
            tasks_ = set()
            for session in id_to_sessions[participant]:
                tasks_ |= session_to_tasks[session]
            tasks_ = ','.join(sorted(list(tasks_)))
            participants_tsv.append(
                dict(participant_id='sub-' + str(participant), tasks=tasks_)
            )
        participants_tsv = pd.DataFrame(participants_tsv)
        participants_tsv.to_csv(os.path.join(BIDS_PATH, project, 'participants.tsv'), index=False, sep='\t')

        participants_json = {
            f"{source}_id" :{
                "LongName": "",
                "Description": "Unique participant-level identifier (BIDS `participant_id` used for sessions in "
                               "this project)"
            },
            "tasks": {
                "LongName": "",
                "Description": "Comma-delimited list of tasks. Auto-generated by `climbprep.repair`."
            }
        }
        with open(os.path.join(BIDS_PATH, project, 'participants.json'), 'w') as f:
            json.dump(participants_json, f, indent=2)

        readme_str = f'A multisession dataset for the {project} project.'
        with open(os.path.join(BIDS_PATH, project, 'README'), 'w') as f:
            f.write(readme_str)

    # Add symlinks to sourcedata
    for participant in participants:
        participant_path = os.path.join(project_path, 'sourcedata', f'sub-{participant}')
        if not os.path.exists(participant_path):
            stderr(f'Participant path {participant_path} not found, creating.\n')
            os.makedirs(participant_path)

        for session in id_to_sessions[participant]:
            session_path = os.path.join(participant_path, f'ses-{session}')
            source_path = os.path.join(source_project_path, 'sourcedata', f'sub-{session}')
            if not os.path.exists(session_path):
                stderr(f'Session path {session_path} not found, creating.\n')
                os.symlink(source_path, session_path, target_is_directory=True)

    # Clear incorrect BIDSified data
    for participant_path in os.listdir(project_path):
        if participant_path.startswith('sub-'):
            participant = participant_path[4:]
            if participant not in participants:
                shutil.rmtree(os.path.join(project_path, participant_path))
            else:
                for session_path in os.listdir(os.path.join(project_path, participant_path)):
                    session = session_path[4:]
                    if session not in sessions:
                        shutil.rmtree(os.path.join(project_path, participant_path, session_path))

    # Clear incorrect sourcedata links
    sourcedata_path = os.path.join(project_path, 'sourcedata')
    for participant_path in os.listdir(sourcedata_path):
        if participant_path.startswith('sub-'):
            participant = participant_path[4:]
            if participant not in participants:
                shutil.rmtree(os.path.join(sourcedata_path, participant_path))
            else:
                for session_path in os.listdir(os.path.join(sourcedata_path, participant_path)):
                    session = session_path[4:]
                    if session not in sessions:
                        os.remove(os.path.join(sourcedata_path, participant_path, session_path))
