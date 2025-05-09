import os
import argparse

from climbprep.constants import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Run fMRIprep on a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", "evlab", '
                                                                        'etc.). Default: "climblab"'))
    argparser.add_argument('-l', '--preprocessing-label', default='main', help='String identifier for fMRIprep settings.')
    argparser.add_argument('-f', '--fs-license-file', default=FS_LICENSE_PATH, help='Path to FreeSurfer license file.')
    args, kwargs = argparser.parse_known_args()

    participant = args.participant.replace('sub-')
    project_path = os.path.join(BIDS_PATH, args.project)
    out_path = os.path.join(project_path, 'derivatives', 'fmriprep', args.preprocessing_label)
    work_path = os.path.join(WORK_PATH, args.project, participant)

    cmd = f'fmriprep {project_path} {out_path} participant -w {work_path} --participant-label {args.participant} --fs-license-file {args.fs_license_file}'
    cmd = f'''singularity exec {FMRIPREP_IMG} bash -c "{' '.join([cmd] + kwargs)}"'''

    os.system(cmd)
