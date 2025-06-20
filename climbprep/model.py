import os
import re
import json
import yaml
import pandas as pd
from nilearn import image, surface, maskers, signal, interfaces
import argparse

from climbprep.constants import *
from climbprep.util import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Model (run firstlevels on) a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-m', '--models', default=[], nargs='*', help=('List of models to run. '
                                                                          '(see `climbprep.constants.MODELFILES_PATH` '
                                                                          'If not specified, will run all models '
                                                                          'available for the participant.'))
    argparser.add_argument('-c', '--config', default=MODEL_DEFAULT_KEY, help=('Keyword (currently `main`) '
        'or YAML config file to used to parameterize firstlevels. If a keyword is provided, will use the default '
        'settings for that keyword. If not provided, will use default settings. '
        'To view the available config options, see `DEFAULTS["model"]["main"]` in `climbprep/constants.py`. '))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    models = args.models
    if models:
        models = set([x.replace('_model.json', '') for x in models])
        infer_models = False
    else:
        models = set()
        infer_models = True

    config = args.config
    if config in DEFAULTS['model']:
        config_default = DEFAULTS['model'][config]
        config = {}
    else:
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        config_default = DEFAULTS['model'][MODEL_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'model_label' in config, 'Required field `model_label` not found in config. ' \
                                    'Please provide a valid config file or keyword.'
    model_label = config.pop('model_label')
    assert 'preprocessing_label' in config, 'Required field `preprocessing_label` not found in config. ' \
                                            'Please provide a valid config file or keyword.'
    preprocessing_label = config.pop('preprocessing_label')

    # Set paths
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path
    fmriprep_path = os.path.join(derivatives_path, 'fmriprep', preprocessing_label)
    assert os.path.exists(fmriprep_path), 'Path not found: %s' % fmriprep_path
    modelfiles_path = MODELFILES_PATH
    model_library = set()
    task_to_models = {}
    for model in os.listdir(modelfiles_path):
        ix = max(0, len(model) - 11)
        name = model[:ix]
        suffix = model[ix:]
        assert suffix == '_model.json', 'Bad model file: %s (does not end with `_model.json`)' % model
        assert name.isidentifier(), 'Bad model file: %s (%s is not a valid Python identifier)' % (model, name)
        model_library.add(name)
        with open(os.path.join(modelfiles_path, model), 'r') as f:
            model_config = json.load(f)
        key1 = 'Input'
        if key1 not in model_config:
            key1 = 'input'
        assert key1 in model_config, 'Bad model file: %s (does not contain `%s` key)' % (model, key1)
        key2 = 'Task'
        if key2 not in model_config[key1]:
            key2 = 'task'
        assert key2 in model_config[key1], 'Bad model file: %s (does not contain `%s > %s` key)' % (model, key1, key2)
        tasks = model_config[key1][key2]
        if isinstance(tasks, str):
            tasks = [tasks]
        for task in tasks:
            if not task in task_to_models:
                task_to_models[task] = set()
            task_to_models[task].add(name)

    if infer_models:
        sessions = set()
        for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
            if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
                sessions.add(subdir[4:])
        if not sessions:
            sessions = {None}
        for session in sessions:
            # Set session-dependent paths
            subdir = 'sub-%s' % participant
            if session:
                subdir = os.path.join(subdir, 'ses-%s' % session)
                session_str = '_ses-%s' % session
            else:
                session_str = ''
            raw_path = os.path.join(project_path, subdir, 'func')
            assert os.path.exists(raw_path), 'Path not found: %s' % raw_path

            for path in os.listdir(raw_path):
                if path.endswith('_bold.nii.gz'):
                    task = TASK_RE.match(path)
                    if task:
                        task = task.group(1)
                        if task in task_to_models:
                            models |= task_to_models[task]
    else:
        missing = models - model_library
        if missing:
            stderr('The following models are not available in the model library: %s. Skipping.\n' % ', '.join(missing))
        models_ = models & model_library
        if not models_:
            stderr('No valid models specified. Exiting.\n')
            exit()
        models = models_

    for model in models:
        modelfile = os.path.join(modelfiles_path, '%s_model.json' % model)
        out_path = os.path.join(project_path, 'derivatives', 'firstlevels', model_label, model)
        work_path = os.path.join(WORK_PATH, project, participant, 'firstlevels', model_label, model)

        kwarg_strings = []
        for key in config:
            if key == 'smoothing_method':
                continue
            val = config[key]
            if key == 'smoothing_fwhm':
                key = 'smoothing'
                method = config.get('smoothing_method', 'isoblurto')
                val = f'{val}:run:{method}'
            if isinstance(val, list) or isinstance(val, tuple):
                val = ' '.join(val)
            key_str = '--%s' % key.replace('_', '-')
            if val is True:
                kwarg_strings.append(key_str)
            else:
                kwarg_strings.append(f'{key_str} {val}')

        args = [
                   project_path,
                   out_path,
                   'participant',
                   f'--participant-label {participant}',
                   f'--model {modelfile}',
                   f'--derivatives {fmriprep_path}',
                   f'--work-dir {work_path}',
               ] + kwarg_strings
        cmd = " ".join(args)
        cmd = f'''singularity run {FITLINS_IMG} {cmd}'''

        stderr(cmd + '\n')
        status = os.system(cmd)
        if status:
            stderr('Command failed with status %d. Exiting.\n' % status)
            exit(status)
