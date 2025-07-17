import re
import os
import numpy as np

# Paths
LAB_PATH = os.path.normpath(os.path.join('/', 'juice6', 'u', 'nlp', 'climblab'))
APPTAINER_PATH = os.path.normpath(os.path.join(LAB_PATH, 'apptainer'))
CODE_PATH = os.path.normpath(os.path.join(LAB_PATH, 'code'))
BIDS_PATH = os.path.join(LAB_PATH, 'BIDS')
WORK_PATH = os.path.join(LAB_PATH, 'work')
EVENTFILES_PATH = os.path.join(LAB_PATH, 'eventfiles')
MODELFILES_PATH = os.path.join(LAB_PATH, 'modelfiles')
FS_LICENSE_PATH = os.path.join(LAB_PATH, 'freesurfer', 'license.txt')
FMRIPREP_IMG = os.path.join(LAB_PATH, 'apptainer', 'images', 'fmriprep.simg')
FITLINS_IMG = os.path.join(LAB_PATH, 'apptainer', 'images', 'fitlins-climbprep.sif')

# Defaults
DEFAULT_TASK = 'UnknownTask'
PREPROCESS_DEFAULT_KEY = 'main'
CLEAN_DEFAULT_KEY = 'fc'
MODEL_DEFAULT_KEY = 'T1w'
PLOT_DEFAULT_KEY = MODEL_DEFAULT_KEY
PARCELLATE_DEFAULT_KEY = 'T1w'
DEFAULT_MASK_SUFFIX = '_desc-ribbon_mask.nii.gz'
DEFAULT_MASK_FWHM = 1
DEFAULT_TARGET_AFFINE = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

# Regex
SPACE_RE = re.compile('.+_space-([a-zA-Z0-9]+)_')
RUN_RE = re.compile('.+_run-([0-9]+)_')
TASK_RE = re.compile('.+_task-([a-zA-Z0-9]+)_')
CONTRAST_RE = re.compile('.+_contrast-([a-zA-Z0-9]+)_')
HEMI_RE = re.compile('.+_hemi-([a-zA-Z0-9]+)_')
STAT_RE = re.compile('.+_stat-([a-zA-Z0-9]+)_')
FROM_RE = re.compile('.+_from-([a-zA-Z0-9]+)_')
TO_RE = re.compile('.+_to-([a-zA-Z0-9]+)_')

# Modeling
MODEL_CONFOUNDS = [
    "global_signal*",
    "csf*",
    "white_matter*",
    "rot_x*",
    "rot_y*",
    "rot_z*",
    "trans_x*",
    "trans_y*",
    "trans_z*",
    "motion_outlier*",
    1
]
MODEL_TEMPLATE = {
    "Name": None,
    "BIDSModelVersion": "1.0.0",
    "Description": None,
    "Input": {
        "task": []
    },
    "Nodes": [
        {
            "Level": "Run",
            "Name": "run",
            "GroupBy": ["run", "task", "session", "subject"],
            "Transformations": {
                "Transformer": "pybids-transforms-v1",
                "Instructions": [
                    {
                        "Name": "Factor",
                        "Input": "trial_type"
                    },
                    {
                        "Name": "Convolve",
                        "Input": [
                            "trial_type.*"
                        ]
                    }
                ]
            },
            "Model": {
                "X": MODEL_CONFOUNDS
            },
            "Contrasts": []
        },
        {
            "Level": "Session",
            "Name": "session",
            "GroupBy": ["session", "subject"],
            "Model": {
                "X": [],
                "Type": "meta"
            },
            "DummyContrasts": {"Test": "t"}
        },
        {
            "Level": "Subject",
            "Name": "subject",
            "GroupBy": ["subject"],
            "Model": {
                "X": [],
                "Type": "meta"
            },
            "DummyContrasts": {"Test": "t"}
        }
    ],
    "Edges": [
        {"Source": "run", "Destination": "session"},
        {"Source": "run", "Destination": "subject"}
    ]
}

# Plotting
PLOT_COLORS = [
    'Red',
    'Blue',
    'Lime',
    'Cyan',
    'Magenta',
    'Yellow'
    'Orange',
    'Pink',
    'Green',
    'aquamarine',
    'Violet',
    'Teal',
    'saddlebrown',
    'deeppink',
    'navy'
]
PLOT_STATMAP_SUFFIX = '.nii.gz'
PLOT_IMG_ORDER = [0, 1, 3, 2]
PLOT_LIGHTING = {
    'ambient': 0.65,
    'diffuse': 0.5,
    'fresnel': 0.25,
    'specular': 0.25,
    'roughness': 0.25,
    'facenormalsepsilon': 0,
    'vertexnormalsepsilon': 0
}
PLOT_LIGHTPOSITION = dict(x=100,
                          y=200,
                          z=100)
PLOT_VIEWS = {
    ('left', 'lateral'): (0, -90),
    ('left', 'medial'): (0, 90),
    ('right', 'lateral'): (0, 90),
    ('right', 'medial'): (0, -90),
}
PLOT_BG_BRIGHTNESS = 0.
PLOT_SULC_ALPHA = 0.8
PLOT_BG_ALPHA = 0.1
PLOT_AXIS_CONFIG = {
    'showgrid': False,
    'showline': False,
    'ticks': '',
    'title': '',
    'showticklabels': False,
    'zeroline': False,
    'showspikes': False,
    'spikesides': False,
    'showbackground': False,
    'visible': False
}

# Parcellation


# Configurations
CONFIG = dict(
    preprocess=dict(
        main=dict(
            fs_license_file=FS_LICENSE_PATH,
            output_space=['MNI152NLin2009cAsym', 'T1w', 'fsnative'],
            skull_strip_t1w='skip',
            cifti_output='91k'
        )
    ),
    clean=dict(
        fc=dict(
            clean_surf=False,
            preprocessing_label='main',
            # Matches global_signal, white, trans, rot, and motion_outlier confounds w/derivatives, excluding redundant csf_wm
            #confounds_regex=f'^(?!csf_wm)(global_signal|csf|white|trans|rot|motion_outlier).*',
            #confounds_regex=f'^(?!csf_wm)(global_signal|csf|white_matter|trans(_x|_y|_z)|rot(_x|_y|_z)|motion_outlier.*)(_derivative1)?$',
            confounds_regex=f'^(trans(_x|_y|_z)|rot(_x|_y|_z)|a_comp_cor_0[1-5]|cosine.*|motion_outlier.*)(_derivative1)?$',
            smoothing_fwhm=4,
            standardize=True,
            detrend=True,
            regress_out_task=True,
            low_pass=0.1,
            high_pass=0.01,
            n_jobs=-1
        ),
        firstlevels_like=dict(
            clean_surf=False,
            # Matches all global_signal, white, trans, rot, and motion_outlier confounds, excluding redundant csf_wm
            confounds_regex=f'^(?!csf_wm)(global_signal|csf|white|trans|rot|motion_outlier).*',
            smoothing_fwhm=4,
            standardize=False,
            detrend=True,
            regress_out_task=False,
            low_pass=None,
            high_pass=0.01,
            n_jobs=-1
        )
    ),
    model=dict(
        mni=dict(
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='MNI152NLin2009cAsym',
            estimator='nilearn',
            drift_model='cosine',
            drop_missing=True
        ),
        mni_afni=dict(
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='MNI152NLin2009cAsym',
            estimator='afni',
            drift_model='cosine',
            drop_missing=True
        ),
        T1w=dict(
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='T1w',
            estimator='nilearn',
            drift_model='cosine',
            drop_missing=True
        ),
        T1w_afni=dict(
            preprocessing_label='main',
            smoothing_fwhm=4,
            smoothing_method='iso',
            space='T1w',
            estimator='afni',
            drift_model='cosine',
            drop_missing=True
        )
    ),
    plot=dict(
        T1w=dict(),
        mni=dict()
    ),
    parcellate=dict(
        T1w=dict(
            cleaning_label='fc',
            space='T1w',
            sample=dict(
                main=dict(
                    n_networks=100,
                    n_samples=1000,
                    low_pass=None,
                    high_pass=None,
                    fwhm=None
                )
            ),
            align=dict(
                main=dict(
                    prealign=True,
                    minmax_normalize=False
                )
            )
        )
    )
)
for key in CONFIG['model']:
    CONFIG['plot'][key] = dict(
        model_label=key,
        vmax=dict(
            t=3,
            z=3,
        ),
        threshold=dict(
            t=0,
            z=0
        ),
        engine='plotly',
        vtrim=0.1,
        htrim=0.1,
        scale=1
    )


# Quickstart
PROFILE = '''# ~/.profile: executed by the command interpreter for login shells.
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi'''

PROFILE_CLIMBLAB = '''
if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc_climblab" ]; then
        . "$HOME/.bashrc_climblab"
    fi
fi'''

BASHRC = r'''
export SQUEUE_FORMAT="%.10i %9P %35j %.8u %.2t %.12M %.12L %.5C %.7m %.4D %b %R"

# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nlp/scr/{{USER}}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nlp/scr/{{USER}}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nlp/scr/{{USER}}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nlp/scr/{{USER}}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# helper env for nlprun to set environment
if [ ! $ANACONDA_ENV == '' ]; then
    conda activate $ANACONDA_ENV
fi

umask 002
'''

BASHRC_CLIMBLAB = r'''export APPTAINER_BIND={{LAB_PATH}},/juice6/u/{{USER}},/juice6/scr6/{{USER}},/afs/cs.stanford.edu/u/{{USER}}/BIDS,/afs/cs.stanford.edu/u/{{USER}}/{{USER}},/afs/cs.stanford.edu/u/{{USER}}/code
export APPTAINER_CACHEDIR=/juice6/u/{{USER}}/.apptainer/cache
export FREESURFER_HOME={{LAB_PATH}}/freesurfer
export PATH=$PATH:/u/nlp/bin:/usr/local/cuda:{{APPTAINER_PATH}}/bin:{{CODE_PATH}}/climbprep/bin:{{LAB_PATH}}/bin
export PYTHONPATH=$PYTHONPATH:{{CODE_PATH}}/climbprep:{{CODE_PATH}}/parcellate
export TMPDIR={{LAB_PATH}}/tmp
'''
