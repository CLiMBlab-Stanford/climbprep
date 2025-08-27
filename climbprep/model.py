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
    argparser.add_argument('-c', '--config', default=MODEL_DEFAULT_KEY, help=('Config name (default `T1w`) '
        'or YAML config file to used to parameterize model. '
        'See `climbprep.constants.CONFIG["model"]` for available config names and their settings.'))
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
    if config in CONFIG['model']:
        model_label = config
        config_default = CONFIG['model'][config]
        config = {}
    elif SMOOTHING_RE.match(config):
        config, fwhm = SMOOTHING_RE.match(config).groups()
        model_label = f'{config}{fwhm}mm'
        assert config in CONFIG['model'], 'Provided config (%s) does not match any known keyword.' % config
        config_default = CONFIG['model'][config]
        config_default['smoothing_fwhm'] = float(fwhm)
        config = {}
    else:
        assert config.endswith('_model.yml'), 'config must either be a known keyword or a file ending in ' \
                '_model.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        model_label = os.path.basename(config)[:-10]
        config_default = CONFIG['model'][MODEL_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'preprocessing_label' in config, 'Required field `preprocessing_label` not found in config. ' \
                                            'Please provide a valid config file or keyword.'
    preprocessing_label = config.pop('preprocessing_label')

    # Set paths
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path
    preprocess_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label)
    assert os.path.exists(preprocess_path), 'Path not found: %s' % preprocess_path
    modelfiles_path = MODELFILES_PATH
    assert os.path.exists(modelfiles_path), 'Path not found: %s' % modelfiles_path

    stderr(f'Modeling outputs will be written to {os.path.join(derivatives_path, "model", model_label)}\n')

    task_to_models = {}
    for model in os.listdir(modelfiles_path):
        ix = max(0, len(model) - 11)
        name = model[:ix]
        suffix = model[ix:]
        if not suffix == '_model.json':
            continue
        assert name.isidentifier(), 'Bad model file: %s (%s is not a valid Python identifier)' % (model, name)
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

    available_models = set()
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
        bids_path = os.path.join(project_path, subdir, 'func')
        assert os.path.exists(bids_path), 'Path not found: %s' % bids_path

        for path in os.listdir(bids_path):
            if path.endswith('_bold.nii.gz'):
                task = TASK_RE.match(path)
                if task:
                    task = task.group(1)
                    if task in task_to_models:
                        available_models |= task_to_models[task]
    if infer_models:
        models = available_models & set(task_to_models.keys())  # Run models whose names exactly match an available task
    else:
        missing = models - available_models
        if missing:
            stderr('The following models are not available in the model library: %s. Skipping.\n' % ', '.join(missing))
        models_ = models & available_models
        if not models_:
            stderr('No valid models specified. Exiting.\n')
            exit()
        models = models_

    for model in models:
        modelfile = os.path.join(modelfiles_path, '%s_model.json' % model)
        out_path = os.path.join(project_path, 'derivatives', 'model', model_label, model)
        work_path = os.path.join(WORK_PATH, project, 'derivatives', 'model', model_label, model, f'sub-{participant}')

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
                   f'--derivatives {preprocess_path}',
                   f'--work-dir {work_path}'
               ] + kwarg_strings
        cmd = " ".join(args)
        cmd = f'''singularity run {os.path.join(APPTAINER_PATH, "images", FITLINS_IMG)} {cmd}'''

        stderr(cmd + '\n')
        status = os.system(cmd)
        if status:
            stderr('Command failed with status %d. Exiting.\n' % status)
            exit(status)
