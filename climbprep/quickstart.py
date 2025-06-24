import os
import argparse

from climbprep.constants import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Set up a new CLiMB Lab user on the cluster.
    ''')

    username = os.popen('whoami').read().strip()

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
