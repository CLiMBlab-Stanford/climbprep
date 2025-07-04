import sys
import json
import yaml
import pandas as pd
import argparse

from climbprep.constants import *
from climbprep.util import *


def get_contrast_spec(name, weights):
    """
    Generate a contrast dictionary for the given name and conditions.

    :param name: ``str``; name of the contrast
    :param weights: ``dict``; map from condition names to weights
    :return: ``dict``; contrast dictionary
    """

    conditions = []
    condition_weights = []
    for condition in weights:
        conditions.append(condition)
        condition_weights.append(weights[condition])

    return {
        "Name": name,
        "ConditionList": conditions,
        "Weights": condition_weights,
        "Test": "t"
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(('Generate a model configuration file for fMRI analysis. Writes a file to '
                                         'the current working directory named `name`_model.json.'))
    argparser.add_argument('participant', help=('BIDS participant ID from which to initialize the model configuration '
                                                '(i.e., load *events.tsv to extract condition names).'))
    argparser.add_argument('tasks', nargs='+', help='List of tasks to include in the model.')
    argparser.add_argument('-n', '--name', help=('Name of the model configuration file to generate. Required if '
                                                 '`len(tasks) > `1, otherwise can be inferred as the task name.'))
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', help=('Path to YAML file containing any additional contrasts '
                                                   'to include beyond the default (one dummy contrast for '
                                                   'each condition). If provided, must be in the form of:\n'
                                                   '<NODE_NAME>:\n'
                                                   '  <CONTRAST_NAME>:\n'
                                                   '    <CONDITION>: <CONDITION_WEIGHT>\n\n'
                                                   'where <NODE_NAME> is one of "run", "session", or '
                                                   '"subject", any/all of which can be included as '
                                                   'top-level keys. For example, to add a sentences vs. '
                                                   'nonwords contrast called "SvN", the config should '
                                                   'be as follows:\n'
                                                   'run:\n'
                                                   '  SvN:\n'
                                                   '    S: 0.5\n'
                                                   '    N: -0.5\n\n'
                                                   'Note that the magnitude of the weights should sum to'
                                                   '1, for valid comparison between contrasts.'))
    args = argparser.parse_args()

    participant = args.participant
    tasks = args.tasks
    name = args.name
    if not name:
        if len(tasks) == 1:
            name = tasks[0]
        else:
            raise ValueError('Must provide a name for the model if more than one task is included.')
    tasks = set(tasks)
    project = args.project
    config = args.config
    if config:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path

    sessions = set()
    for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}

    conditions = set()
    for session in sessions:
        subdir = 'sub-%s' % participant
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        raw_path = os.path.join(project_path, subdir)
        assert os.path.exists(raw_path), 'Path not found: %s' % raw_path
        func_path = os.path.join(raw_path, 'func')
        assert os.path.exists(func_path), 'Path not found: %s' % func_path

        for event_file in os.listdir(func_path):
            if event_file.endswith('_events.tsv'):
                task = TASK_RE.match(event_file)
                if task:
                    task = task.group(1)
                    if task in tasks:
                        df = pd.read_csv(os.path.join(func_path, event_file), sep='\t')
                        conditions_ = set(df.trial_type.unique() if 'trial_type' in df.columns else [])
                        conditions |= conditions_

    second_level_in = conditions | set(config.get('run', {}).keys())

    model = MODEL_TEMPLATE.copy()
    model['Name'] = name
    model['Description'] = name
    model['Input']['task'] = sorted(list(tasks))
    model['Nodes'][0]['Model']['X'].insert(0, 'trial_type.*')
    model['Nodes'][0]['Contrasts'] = [
        get_contrast_spec(condition, {f"trial_type.{condition}": 1}) for condition in conditions
    ]
    model['Nodes'][1]['Model']['X'] = sorted(list(second_level_in))
    model['Nodes'][2]['Model']['X'] = sorted(list(second_level_in))

    for level in config:
        if level not in ['run', 'session', 'subject']:
            raise ValueError(f'Invalid node name: {level}. Must be one of "run", "session", or "subject".')
        if level == 'run':
            node = model['Nodes'][0]
        elif level == 'session':
            node = model['Nodes'][1]
        else:
            node = model['Nodes'][2]

        for contrast_name in config[level]:
            contrast_weights = config[level][contrast_name]
            if not isinstance(contrast_weights, dict):
                raise ValueError(f'Invalid contrast weights for {level} {contrast_name}: {contrast_weights}. '
                                 'Must be a dictionary mapping condition names to weights.')
            contrast_spec = get_contrast_spec(contrast_name, contrast_weights)
            if 'Contrasts' not in node:
                node['Contrasts'] = []
            node['Contrasts'].append(contrast_spec)

    out_path = os.path.join(os.getcwd(), f'{name}_model.json')
    stderr(f'Saving to {out_path}\n')
    with open(out_path, 'w') as f:
        json.dump(model, f, indent=2)
