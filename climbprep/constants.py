import re
import os

LAB_PATH = os.path.normpath(os.path.join('/', 'juice6', 'u', 'nlp', 'climblab'))
APPTAINER_PATH = os.path.normpath(os.path.join(LAB_PATH, 'apptainer'))
CODE_PATH = os.path.normpath(os.path.join(LAB_PATH, 'code'))
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
CONTRAST_RE = re.compile('.+_contrast-([a-zA-Z0-9]+)_')
HEMI_RE = re.compile('.+_hemi-([a-zA-Z0-9]+)_')

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
            clean_surf=False,
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
            clean_surf=False,
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

BASHRC = r'''export APPTAINER_BIND={{LAB_PATH}},/juice6/u/{{USER}},/juice6/scr6/{{USER}},/afs/cs.stanford.edu/u/{{USER}}/BIDS,/afs/cs.stanford.edu/u/{{USER}}/{{USER}},/afs/cs.stanford.edu/u/{{USER}}/code
export APPTAINER_CACHEDIR=/juice6/u/{{USER}}/.apptainer/cache
export FREESURFER_HOME={{LAB_PATH}}/freesurfer
export PATH=$PATH:/u/nlp/bin:/usr/local/cuda:{{APPTAINER_PATH}}/bin:{{CODE_PATH}}/climbprep
export PYTHONPATH=$PYTHONPATH:{{CODE_PATH}}/climbprep
export TMPDIR={{LAB_PATH}}/tmp

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
