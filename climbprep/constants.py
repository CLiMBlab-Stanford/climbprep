import os

LAB_PATH = os.path.normpath(os.path.join('/', 'juice2', 'scr2', 'nlp', 'climblab'))
BIDS_PATH = os.path.join(LAB_PATH, 'BIDS')
WORK_PATH = os.path.join(LAB_PATH, 'work')
FS_LICENSE_PATH = os.path.join(LAB_PATH, 'freesurfer', 'license.txt')
DEFAULT_TASK = 'UnknownTask'
FMRIPREP_IMG = os.path.join(LAB_PATH, 'apptainer', 'images', 'fmriprep.simg')

