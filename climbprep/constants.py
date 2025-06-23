import re
import os

LAB_PATH = os.path.normpath(os.path.join('/', 'juice6', 'u', 'nlp', 'climblab'))
BIDS_PATH = os.path.join(LAB_PATH, 'BIDS')
WORK_PATH = os.path.join(LAB_PATH, 'work')
EVENTFILES_PATH = os.path.join(LAB_PATH, 'eventfiles')
MODELFILES_PATH = os.path.join(LAB_PATH, 'modelfiles')
FS_LICENSE_PATH = os.path.join(LAB_PATH, 'freesurfer', 'license.txt')
DEFAULT_TASK = 'UnknownTask'
FMRIPREP_IMG = os.path.join(LAB_PATH, 'apptainer', 'images', 'fmriprep.simg')
FITLINS_IMG = os.path.join(LAB_PATH, 'apptainer', 'images', 'fitlins.simg')

SPACE_RE = re.compile('.+_space-([a-zA-Z0-9]+)_')
RUN_RE = re.compile('.+_run-([0-9]+)_')
TASK_RE = re.compile('.+_task-([a-zA-Z0-9]+)_')
HEMI_RE = re.compile('.+_task-([a-zA-Z0-9]+)_')

DEFAULTS = dict(
    preprocess=dict(
        main=dict(
            preprocessing_label='main',
            fs_license_file=FS_LICENSE_PATH,
            output_space=['MNI152NLin2009cAsym', 'T1w', 'fsnative'],
            skull_strip_t1w='skip',
            cifti_output='91k'
        )
    ),
    clean=dict(
        fc=dict(
            cleaning_label='fc',
            preprocessing_label='main',
            strategy=(
                'global_signal',
                'wm_csf',
                'motion',
                'scrub'
            ),
            global_signal='full',
            wm_csf='full',
            motion='full',
            smoothing_fwhm=4,
            std_dvars_threshold=1.5,
            fd_threshold=0.5,
            scrub=True,
            standardize=True,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            n_jobs=-1
        ),
        firstlevels_like=dict(
            cleaning_label='firstlevels',
            preprocessing_label='main',
            strategy=(
                'global_signal',
                'wm_csf',
                'motion',
                'scrub'
            ),
            global_signal='full',
            wm_csf='full',
            motion='full',
            smoothing_fwhm=4,
            std_dvars_threshold=1.5,
            fd_threshold=0.5,
            scrub=False,
            standardize=False,
            detrend=True,
            low_pass=None,
            high_pass=0.01,
            n_jobs=-1
        )
    ),
    model=dict(
        mni=dict(
            model_label='mni',
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='MNI152NLin2009cAsym',
            estimator='nilearn',
            drift_model='cosine',
            drop_missing=True
        ),
        T1w=dict(
            model_label='T1w',
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='T1w',
            estimator='nilearn',
            drift_model='cosine',
            drop_missing=True
        )
    )
)
DEFAULTS['model']['anat'] = DEFAULTS['model']['T1w'].copy()

PREPROCESS_DEFAULT_KEY = 'main'
CLEAN_DEFAULT_KEY = 'fc'
MODEL_DEFAULT_KEY = 'mni'
