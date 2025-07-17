import yaml
from copy import deepcopy
import argparse

from climbprep.constants import *
from climbprep.util import *

IMG_CACHE = {}

def load_img(path, cache=IMG_CACHE):
    if path in cache:
        return cache[path]
    img = image.load_img(path)
    cache[path] = img
    return img


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Clean (denoise) a participant's functional data")
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default=PARCELLATE_DEFAULT_KEY, help=('Config name (default `T1w`) '
        'or YAML config file to used to parameterize parcellation. '
        'See `climbprep.constants.CONFIG["parcellate"]` for available config names and their settings. '))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    config = args.config
    if config in CONFIG['parcellate']:
        parcellation_label = config
        config_default = CONFIG['parcellate'][config]
        config = {}
    else:
        assert config.endswith('_parcellate.yml'), 'config must either be a known keyword or a file ending in ' \
                '_parcellate.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        parcellation_label = config[:-10]
        config_default = CONFIG['parcellate'][PARCELLATE_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'cleaning_label' in config, 'Required field `cleaning_label` not found in config. ' \
                                       'Please provide a valid config file or keyword.'
    cleaning_label = config.pop('cleaning_label')
    assert 'space' in config, 'Required field `space` not found in config. ' \
                              'Please provide a valid config file or keyword.'
    space = config.pop('space')

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, args.project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path

    stderr(f'Parcellation outputs will be written to '
           f'{os.path.join(derivatives_path, "parcellate", parcellation_label)}\n')

    sessions = set()
    for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}
    nodes = {}
    for session in sessions:
        # Set session-dependent paths
        subdir = 'sub-%s' % participant
        participant_dir = subdir
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        clean_path = os.path.join(derivatives_path, 'clean', cleaning_label, subdir)
        cleaning_config_path = os.path.join(clean_path, 'config.yml')
        assert os.path.exists(cleaning_config_path), 'Path not found: %s' % cleaning_config_path
        with open(cleaning_config_path, 'r') as f:
            cleaning_config = yaml.safe_load(f)
        preprocessing_label = cleaning_config['preprocessing_label']
        preprocess_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label, subdir)
        assert os.path.exists(preprocess_path), 'Path not found: %s' % preprocess_path
        anat_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label, participant_dir, 'anat')
        anat_by_session = False
        if not os.path.exists(anat_path):
            anat_path = os.path.join(preprocess_path, 'anat')
            anat_by_session = True
        assert os.path.exists(anat_path), 'Path not found: %s' % anat_path

        if session:
            ses_str = f'_ses-{session}'
        else:
            ses_str = ''

        functional_paths = []
        for path in os.listdir(clean_path):
            if path.endswith('desc-clean_bold.nii.gz'):
                space_ = SPACE_RE.match(path)
                if space_ and space_.group(1) == space:
                    functional_paths.append(os.path.join(clean_path, path))

        xfm_path = None
        if 'mni' in space.lower():
            surface = None
        else:  # Get transform from MNI to native space, and native surface data
            for path in os.listdir(anat_path):
                if path.endswith('_mode-image_xfm.h5'):
                    t = TO_RE.match(path)
                    f = FROM_RE.match(path)
                    if t and f:
                        t = t.group(1)
                        f = f.group(1)
                        if t == space and 'mni' in f.lower():
                            xfm_path = os.path.join(anat_path, path)
                            break
            assert xfm_path, f'Non-MNI space used but no matching transform (*_xfm.h5) found in {anat_path}.'

            if anat_by_session:
                ses_str_ = ses_str
            else:
                ses_str_ = ''
            surface = dict(
                pial=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_pial.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_pial.surf.gii')
                ),
                white=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_white.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_white.surf.gii')
                ),
                midthickness=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_midthickness.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_midthickness.surf.gii')
                ),
                sulc=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_sulc.shape.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_sulc.shape.gii')
                )
            )

        parcellation_config = deepcopy(config)
        if xfm_path:
            parcellation_config['xfm_path'] = xfm_path
        if surface:
            parcellation_config['surface'] = surface
        parcellation_config['sample']['main']['functional_paths'] = []

        if not 'session' in nodes:
            nodes['session'] = {}
        if not session in nodes['session']:
            nodes['session'][session] = deepcopy(parcellation_config)
        nodes['session'][session]['sample']['main']['functional_paths'] = functional_paths
        if not 'subject' in nodes:
            nodes['subject'] = deepcopy(parcellation_config)
        nodes['subject']['sample']['main']['functional_paths'] += functional_paths

    parcellation_dir = os.path.join(derivatives_path, 'parcellate', parcellation_label)
    cliargs = []
    for node in nodes:
        node_dir = os.path.join(parcellation_dir, f'node-{node}')
        participant_dir = os.path.join(node_dir, 'sub-%s' % participant)
        if node == 'subject':
            sessions = {None}
        else:
            sessions = set(nodes[node].keys())
        for session in sessions:
            if session:
                session_dir = os.path.join(participant_dir, f'ses-{session}')
                config_ = nodes[node][session]
            else:
                session_dir = participant_dir
                config_ = nodes[node]
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            config_['output_dir'] = session_dir
            config_path = os.path.join(session_dir, 'config.yml')
            with open(config_path, 'w') as f:
                yaml.dump(config_, f)
            cliargs.append(config_path)

    for cliarg in cliargs:
        cmd = f'python -m parcellate.bin.train {cliarg}'
        print(cmd)
        status = os.system(cmd)
        if status:
            stderr('Error during parcellation. Exiting.\n')
            exit(status)


