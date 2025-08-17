import sys
import os

from climbprep.constants import *

def stderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()

def get_preprocessed_anat_dir(project, participant, preprocessing_label='main'):
    project_path = os.path.join(BIDS_PATH, project)
    participant_path = os.path.join(
        project_path, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}'
    )
    anat_dir = None
    anat_dir_ = os.path.join(participant_path, 'anat')
    if os.path.exists(anat_dir_):
        anat_dir = anat_dir_
    else:
        for session in [x for x in os.listdir(participant_path) if x.startswith('ses-')]:
            if session.startswith('ses-'):
                anat_dir_ = os.path.join(participant_path, session, 'anat')
                if os.path.exists(anat_dir_):
                    assert not anat_dir, (
                        'Multiple preprocessed anat directories found in directory %s. '
                        'Please check the directory structure.' % participant_path
                    )
                    anat_dir = anat_dir_

    assert anat_dir, (
        'No preprocessed anat directory found in %s. '
        'Please check the directory structure.' % participant_path
    )

    return anat_dir
