import os
import argparse

from climbprep.constants import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Set up a new CLiMB Lab user on the cluster.
    ''')

    username = os.popen('whoami').read().strip()
    print(username)

    home_path = os.path.join('/', 'sailhome', username)
    profile_str = PROFILE
    profile_path = os.path.join(home_path, '.profile')
    bashrc_str = BASHRC.replace(r'{{USER}}', username) \
                       .replace(r'{{LAB_PATH}}', LAB_PATH) \
                       .replace(r'{{BIDS_PATH}}', BIDS_PATH) \
                       .replace(r'{{APPTAINER_PATH}}', APPTAINER_PATH) \
                       .replace(r'{{CODE_PATH}}', CODE_PATH)
    bashrc_path = os.path.join(home_path, '.bashrc')
    for content, path in zip((profile_str, bashrc_str), (profile_path, bashrc_path)):
        if os.path.exists(path):
            ans = None
            while ans not in ('y', 'n', ''):
                ans = input(f'File {path} already exists. Overwrite? y/[n] >>> ').strip()
            if ans == 'y':
                with open(path, 'w') as f:
                    f.write(content + '\n')
        else:
            with open(path, 'w') as f:
                f.write(content + '\n')
