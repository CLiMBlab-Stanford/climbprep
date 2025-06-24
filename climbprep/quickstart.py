import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from climbprep import resources
import argparse

from climbprep.constants import *
from climbprep.util import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Set up a new CLiMB Lab user on the cluster.
    ''')

    username = os.popen('whoami').read().strip()

    # Set up environment
    home_path = os.path.join('/', 'sailhome', username)
    profile_str = PROFILE
    profile_path = os.path.join(home_path, '.profile')
    bashrc_str = BASHRC.replace(r'{{USER}}', username) \
                       .replace(r'{{LAB_PATH}}', LAB_PATH) \
                       .replace(r'{{BIDS_PATH}}', BIDS_PATH) \
                       .replace(r'{{APPTAINER_PATH}}', APPTAINER_PATH) \
                       .replace(r'{{CODE_PATH}}', CODE_PATH)
    bashrc_path = os.path.join(home_path, '.bashrc')
    bashrc_climblab_path = os.path.join(home_path, '.bashrc_climblab')
    bashrc_climblab_str = BASHRC_CLIMBLAB.replace(r'{{USER}}', username) \
                                         .replace(r'{{LAB_PATH}}', LAB_PATH) \
                                         .replace(r'{{BIDS_PATH}}', BIDS_PATH) \
                                         .replace(r'{{APPTAINER_PATH}}', APPTAINER_PATH) \
                                         .replace(r'{{CODE_PATH}}', CODE_PATH)

    for content, path in zip(
            (profile_str, bashrc_str, bashrc_climblab_str),
            (profile_path, bashrc_path, bashrc_climblab_path)
        ):
        if path != bashrc_climblab_path:
            if os.path.exists(path):
                ans = None
                while ans not in ('y', 'n', ''):
                    ans = input((f'File {path} already exists. Overwrite? (Not recommended if you '
                                 f'have customized this file, otherwise recommended) y/[n] >>> ')).strip()
                if ans == 'y':
                    with open(path, 'w') as f:
                        f.write(content + '\n')
            else:
                with open(path, 'w') as f:
                    f.write(content + '\n')
            if path == profile_path:
                with open(path, 'r') as f:
                    cur = f.read()
                if PROFILE_CLIMBLAB not in cur:
                    cur += f'\n{PROFILE_CLIMBLAB}\n'
                    with open(path, 'w') as f:
                        f.write(cur + '\n')
        else:
            with open(path, 'w') as f:
                f.write(content + '\n')

    # Install conda if missing
    conda_base_path = os.path.join('/', 'nlp', 'scr', username, 'miniconda3')
    conda_path = os.path.join(conda_base_path, 'bin', 'conda')
    if not os.path.exists(conda_path):
        stderr(f'Conda not found at {conda_path}. Installing.\n')

        cmd = ('mkdir -p ~/miniconda3\n'
            'wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh\n'
            f'bash ~/miniconda.sh -b -u -p {conda_base_path}\n'
            'rm ~/miniconda.sh\n'
        )
        print(cmd)
        status = os.system(cmd)
        if status:
            stderr('Error installing conda. Exiting.\n')
            exit()

    # Ensure if climbprep environment exists
    envs = os.popen(f'{conda_path} info --envs').readlines()
    climbprep_exists = False
    for env in envs:
        env = env.strip()
        if env:
            env = env.split()[0]
            if env == 'climbprep':
                climbprep_exists = True
                break
    if not climbprep_exists:
        stderr(f'`climbprep` environment not found. Installing.\n')
        with pkg_resources.as_file(pkg_resources.files(resources).joinpath('conda.yml')) as path:
            yml_path = path
        status = os.system(f'{conda_path} env create -f {yml_path}')
        if status:
            stderr('Error installing `climbprep` conda environment. Exiting.\n')
            exit()

    stderr('climbprep quickstart complete!\n')



