from nilearn import surface
import argparse

from climbprep.constants import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Convert fMRIprep anatomicals so that they render properly in SurfIce viewer.')
    argparser.add_argument('inpath', help='Path to directory containing anatomicals')
    argparser.add_argument('-o', '--outpath', default=None, help=('Path to output directory. If not provided, will '
                                                                  'write to the same directory as the input '
                                                                  'anatomicals.'))
    args = argparser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    if not outpath:
        outpath = inpath

    for path in os.listdir(inpath):
        is_mesh = is_data = False
        if path.endswith('surf.gii') or path.endswith('surf.gii.gz'):
            is_mesh = True
        elif path.endswith('shape.gii') or path.endswith('shape.gii.gz'):
            is_data = True
        if not (is_mesh or is_data):
            continue

        inpath_ = os.path.join(inpath, path)
        outpath_ = os.path.join(outpath, path.replace('.surf.gii', '_surfice.surf.gii'))
        hemi = HEMI_RE.match(path)
        if not hemi:
            continue
        hemi = hemi.group(1)
        if hemi == 'L':
            kwargs = dict(left=inpath_)
        elif hemi == 'R':
            kwargs = dict(right=inpath_)
        else:
            continue
        if is_mesh:
            obj = surface.PolyMesh(**kwargs)
        else:
            obj = surface.PolyData(**kwargs)
        print(outpath_)
        obj.to_filename(outpath_)
