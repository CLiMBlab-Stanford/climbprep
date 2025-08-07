import yaml
from copy import deepcopy
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from climbprep import resources
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from nilearn import image, surface
from nitransforms import resampling, manip, linear
import argparse

from climbprep.constants import *
from climbprep.util import *


NII_CACHE = {}  # Map from paths to NII objects
GII_CACHE = {}

def resample_to(nii, template):
    """
    Resample a Nifti image to match the shape of a template image.

    :param nii: ``nibabel.Nifti1Image``; Nifti image to resample.
    :param template: ``nibabel.Nifti1Image``; template image.
    :return: ``nibabel.Nifti1Image``; resampled Nifti image.
    """

    nii = image.math_img('nii * (1 + 1e-6)', nii=nii)  # Hack to force conversion to float
    return image.resample_to_img(nii, template, copy_header=True, force_resample=True)


def get_atlas(name, resampling_target_nii=None, xfm_path=None, nii_cache=NII_CACHE):
    """
    Load an atlas from a path or a dictionary containing a path and a value.

    :param atlas: ``str``, ``dict``, or ``tuple``; atlas name to retriev, dictionary containing atlas name and either
        a path or a value, or tuple containing atlas name and either a path or a value.
    :param resampling_target_nii: ``nibabel.Nifti1Image`` or ``None``; template image for resampling. If ``None``,
        no resampling is applied.
    :param xfm_path: ``str`` or ``None``; if the parcellation is not in MNI space, path to
        transformation from MNI to the parcellation space (e.g., native).
        If ``None``, parcellation is assumed to be in MNI space.
    :return: ``str``, ``str``, ``nibabel.Nifti1Image``; atlas name, atlas path, atlas Nifti image
    """

    filename = ATLAS_NAME_TO_FILE.get(name.lower(), None)
    if filename is None:
        raise ValueError('Unrecognized atlas name: %s' % name)
    if nii_cache is None:
        nii_cache = {}
    with pkg_resources.as_file(pkg_resources.files(resources).joinpath(filename)) as path:
        if path not in nii_cache:
            val = image.smooth_img(path, None)
            val = image.math_img('x * (1 + 1e-6)', x=val)  # Hack to force conversion to float

            if xfm_path is not None:
                assert resampling_target_nii is not None, \
                    'If xfm_path is provided, resampling_target_nii must also be provided.'
                if isinstance(xfm_path, str):
                    xfm = manip.load(xfm_path)
                else:
                    xfms = [manip.load(x) if x.endswith('.h5') else linear.load(x) for x in xfm_path]
                    xfm = manip.TransformChain(xfms)
                val = resampling.apply(
                    xfm,
                    val,
                    resampling_target_nii,
                )
            elif resampling_target_nii is not None:
                val = resample_to(val, resampling_target_nii)
            nii_cache[path] = val

        val = nii_cache[path]

    return val


def parcellate_surface(
        functional_paths,
        surface_left_path,
        surface_right_path,
        reference_image_path,
        xfm_path,
        output_dir,
        reference_target_affine=2,
        n_networks=100,
        n_components_pca=None,
        **ignored
):
    gii_cache = GII_CACHE
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = dict(
        functional_paths=functional_paths,
        surface_left=surface_left_path,
        surface_right=surface_right_path,
        reference_image_path=reference_image_path,
        xfm_path=xfm_path,
        output_dir=output_dir,
        reference_target_affine=reference_target_affine,
        n_networks=n_networks,
        n_components_pca=n_components_pca,
        ignored=ignored
    )
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    stderr('Loading atlases')
    reference = image.load_img(reference_image_path)
    if reference_target_affine is not None:
        if isinstance(reference_target_affine, int):
            reference_target_affine = np.eye(3) * reference_target_affine
        elif isinstance(reference_target_affine, tuple) or isinstance(reference_target_affine, list):
            reference_target_affine = np.diag(reference_target_affine)
        reference = image.resample_img(reference, target_affine=reference_target_affine)
    surf_anat = surface.PolyMesh(
        left=surface_left_path,
        right=surface_right_path
    )
    sub = SUB_RE.match(surface_left_path)
    assert sub, 'Surface data file name must contain a subject identifier (e.g., "sub-01").'
    sub = sub.group(1)
    for hemi in ('L', 'R'):
        surf_anat.to_filename(
            os.path.join(output_dir, f'sub-{sub}_hemi-{hemi}_surface.gii')
        )
    v_left = len(surf_anat.parts['left'].coordinates)
    v_right = len(surf_anat.parts['right'].coordinates)
    v = v_left + v_right
    atlas_surfaces = {}
    for atlas_name in ATLAS_NAME_TO_FILE:
        atlas = get_atlas(
            atlas_name,
            resampling_target_nii=reference,
            xfm_path=xfm_path
        )
        atlas_surface_L = surface.vol_to_surf(
            atlas,
            surf_anat.parts['left']
        )
        atlas_surface_R = surface.vol_to_surf(
            atlas,
            surf_anat.parts['right']
        )
        atlas_surfaces[atlas_name] = surface.PolyData(
            left=atlas_surface_L,
            right=atlas_surface_R
        )
        for hemi in ('L', 'R'):
            atlas_surfaces[atlas_name].to_filename(
                    os.path.join(output_dir, f'sub-{sub}_hemi-{hemi}_label-{atlas_name}REF.gii')
            )

    stderr('Loading timecourses\n')
    X = []
    ses = None
    for functional_path in functional_paths:
        assert functional_path.endswith('.gii') or functional_path.endswith('.gii.gz'), \
            'Functional data must be in GIFTI format.'
        if ses is None:
            ses = SES_RE.match(functional_path)
            if ses:
                ses = ses.group(1)
        hemi = HEMI_RE.match(functional_path)
        assert hemi, 'Functional data file name must contain a hemisphere identifier (e.g., "hemi-L" or "hemi-R").'
        hemi = hemi.group(1)
        if hemi == 'L':
            left = functional_path
            right = functional_path.replace('hemi-L', 'hemi-R')
        else:
            left = functional_path.replace('hemi-R', 'hemi-L')
            right = functional_path
        left_path = left
        if left_path not in gii_cache:
            left = surface.load_surf_data(left_path)
            assert len(left.shape) == 2, 'Functional data must be a 2D array (vertices x timepoints).'
            if left.shape[0] != v_left:
                left = left.T
            assert left.shape[0] == v_left, 'Left hemisphere functional data must have %d vertices, got %d.' % (v_left, left.shape[0])
            gii_cache[left_path] = left
        left = gii_cache[left_path]
        right_path = right
        if right_path not in gii_cache:
            right = surface.load_surf_data(right_path)
            assert len(right.shape) == 2, 'Functional data must be a 2D array (vertices x timepoints).'
            if right.shape[0] != v_right:
                right = right.T
            assert right.shape[0] == v_right, 'Right hemisphere functional data must have %d vertices, got %d.' % (v_right, right.shae[0])
            gii_cache[right_path] = right
        right = gii_cache[right_path]
        X_ = np.concatenate([left, right], axis=0)
        X.append(X_)
    X = np.concatenate(X, axis=1)

    if ses is None or ses == 'None':
        ses_str = ''
    else:
        ses_str = f'_ses-{ses}'

    if n_components_pca is not None:
        stderr('Applying PCA\n')
        if n_components_pca.lower() == 'auto':
            n_components_pca = n_networks
        pca = PCA(n_components=n_components_pca)
        X = pca.fit_transform(X)

    stderr('Parcellating\n')
    X = FastICA(n_components=n_networks).fit_transform(X)

    # Assume a network covers < half the mask volume, flip sign accordingly
    X = np.where(np.median(X, axis=0, keepdims=True) > 0, -X, X)
    # Clip and normalize (scale is arbitrary)
    uq = np.quantile(X, 0.99, axis=0, keepdims=True)
    X = np.clip(X, 0, uq) / uq
    parcellation = X.astype(np.float32)

    stderr('Labeling and saving results\n')
    df = []
    remaining = set(range(parcellation.shape[1]))
    for atlas_name in atlas_surfaces:
        atlas_surface = atlas_surfaces[atlas_name]
        atlas = np.concatenate([atlas_surface.parts['left'], atlas_surface.parts['right']], axis=0)
        max_r = -np.inf
        max_ix = None
        for ix in range(parcellation.shape[1]):
            network = parcellation[:, ix]
            r = np.corrcoef(atlas, network)[0, 1]
            if r > max_r:
                max_r = r
                max_ix = ix
        assert max_ix is not None, 'No matching network found for atlas %s.' % atlas_name
        df.append(dict(
            network=atlas_name,
            index=max_ix,
            score=max_r
        ))
        network = parcellation[:, max_ix]
        network = surface.PolyData(
            left=network[:v_left],
            right=network[v_left:]
        )
        for hemi in ('L', 'R'):
            network.to_filename(
                os.path.join(output_dir, f'sub-{sub}{ses_str}_hemi-{hemi}_network-{max_ix:03d}_label-{atlas_name}.gii')
            )

        if max_ix in remaining:
            remaining.remove(max_ix)

    for i, ix in enumerate(sorted(list(remaining))):
        df.append(dict(
            network='other',
            index=ix,
            score=np.nan
        ))
        network = parcellation[:, ix]
        network = surface.PolyData(
            left=network[:v_left],
            right=network[v_left:]
        )
        for hemi in ('L', 'R'):
            network.to_filename(
                    os.path.join(output_dir, f'sub-{sub}{ses_str}_hemi-{hemi}_network-{ix:03d}_label-other{i:03d}.gii')
            )

    df = pd.DataFrame(df)
    df = df.sort_values('index')
    df.to_csv(os.path.join(output_dir, f'sub-{sub}{ses_str}_parcellation.tsv'), index=False, sep='\t')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Clean (denoise) a participant's functional data")
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default=PARCELLATE_DEFAULT_KEY, help=('Config name (default `T1w`) '
        'or YAML config file to used to parameterize parcellation. '
        'See `climbprep.constants.CONFIG["parcellate"]` for available config names and their settings. '))
    args = argparser.parse_args()

    participant = args.participant.replace('sub-', '')
    config = args.config
    if config in CONFIG['parcellate']:
        parcellation_label = config
        config_default = CONFIG['parcellate'][config]
        config = {}
    else:
        assert config.endswith('_parcellate.yml'), 'config must either be a known keyword or a file ending in ' \
                '_parcellate.yml'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        parcellation_label = config[:-10]
        config_default = CONFIG['parcellate'][PARCELLATE_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    is_surface = config.get('surface', False)
    assert 'cleaning_label' in config, 'Required field `cleaning_label` not found in config. ' \
                                       'Please provide a valid config file or keyword.'
    cleaning_label = config.pop('cleaning_label')
    assert 'space' in config, 'Required field `space` not found in config. ' \
                              'Please provide a valid config file or keyword.'
    space = config.pop('space')

    # Set session-agnostic paths
    project_path = os.path.join(BIDS_PATH, args.project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path

    stderr(f'Parcellation outputs will be written to '
           f'{os.path.join(derivatives_path, "parcellate", parcellation_label)}\n')

    sessions = set()
    for subdir in os.listdir(os.path.join(project_path, 'sub-%s' % participant)):
        if subdir.startswith('ses-') and os.path.isdir(os.path.join(project_path, 'sub-%s' % participant, subdir)):
            sessions.add(subdir[4:])
    if not sessions:
        sessions = {None}
    nodes = {}
    for session in sessions:
        # Set session-dependent paths
        subdir = 'sub-%s' % participant
        participant_dir = subdir
        if session:
            subdir = os.path.join(subdir, 'ses-%s' % session)
        clean_path = os.path.join(derivatives_path, 'clean', cleaning_label, subdir)
        cleaning_config_path = os.path.join(clean_path, 'config.yml')
        assert os.path.exists(cleaning_config_path), 'Path not found: %s' % cleaning_config_path
        with open(cleaning_config_path, 'r') as f:
            cleaning_config = yaml.safe_load(f)
        preprocessing_label = cleaning_config['preprocessing_label']
        preprocess_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label, subdir)
        assert os.path.exists(preprocess_path), 'Path not found: %s' % preprocess_path
        anat_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label, participant_dir, 'anat')
        anat_by_session = False
        if not os.path.exists(anat_path):
            anat_path = os.path.join(preprocess_path, 'anat')
            anat_by_session = True
        assert os.path.exists(anat_path), 'Path not found: %s' % anat_path

        if session:
            ses_str = f'_ses-{session}'
        else:
            ses_str = ''

        functional_paths = []
        for path in os.listdir(clean_path):
            if not is_surface and path.endswith('desc-clean_bold.nii.gz'):
                space_ = SPACE_RE.match(path)
                if space_ and space_.group(1) == space:
                    functional_paths.append(os.path.join(clean_path, path))
            elif is_surface and path.endswith('desc-clean_bold.func.gii') or path.endswith('desc-clean_bold.func.gii.gz'):
                space_ = SPACE_RE.match(path)
                hemi = HEMI_RE.match(path)
                if space_ and space_.group(1) == space and hemi and hemi.group(1) == 'L':
                    functional_paths.append(os.path.join(clean_path, path))

        gii = set()
        nii = set()
        for path in functional_paths:
            if path.endswith('.gii') or path.endswith('.gii.gz'):
                gii.add(path)
            else:
                nii.add(path)
        assert not (len(nii) and len(gii)), 'Cannot mix GIFTI and NIfTI files in functional_paths.'

        mask_path = None
        mask_suffix = config.get('mask_suffix', DEFAULT_MASK_SUFFIX)
        xfm_path = None
        if 'mni' in space.lower():
            assert not is_surface, 'Surface parcellation in MNI space not currently supported'
            surfaces = None
            for path in os.listdir(anat_path):
                if space in path and path.endswith(mask_suffix):
                    mask_path = os.path.join(anat_path, path)
                    break
            T1 = None
        else:  # Get transform from MNI to native space, and native surface data
            xfm_path = []
            for path in os.listdir(anat_path):
                if space in path and path.endswith(mask_suffix):
                    mask_path = os.path.join(anat_path, path)
                elif '_space-' not in path and path.endswith(mask_suffix): 
                    mask_path = os.path.join(anat_path, path)
                elif path.endswith('_mode-image_xfm.h5') or path.endswith('_mode-image_xfm.txt'):
                    t = TO_RE.match(path)
                    f = FROM_RE.match(path)
                    if t and f:
                        t = t.group(1)
                        f = f.group(1)
                        if f == 'MNI152NLin2009cAsym' and (t == space or (t == 'T1w' and space == 'fsnative')):
                            xfm_path.append(os.path.join(anat_path, path))
                        elif f == 'T1w' and space == 'fsnative' and t == space:
                            xfm_path.append(os.path.join(anat_path, path))
            assert xfm_path, 'No transform found in %s' % anat_path
            if len(xfm_path) == 1:
                xfm_path = xfm_path[0]
            else:
                assert len(xfm_path) == 2, \
                    f'Expected one or two transforms from MNI to {space} space, found {len(xfm_path)}.'
                if space in xfm_path[0]:
                    # Reverse the order
                    xfm_path = xfm_path[::-1]
            assert xfm_path, f'Non-MNI space used but no matching transform (*_xfm.h5) found in {anat_path}.'

            if anat_by_session:
                ses_str_ = ses_str
            else:
                ses_str_ = ''
            surfaces = dict(
                pial=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_pial.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_pial.surf.gii')
                ),
                white=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_white.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_white.surf.gii')
                ),
                midthickness=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_midthickness.surf.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_midthickness.surf.gii')
                ),
                sulc=dict(
                    left=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-L_sulc.shape.gii'),
                    right=os.path.join(anat_path, f'sub-{participant}{ses_str_}_hemi-R_sulc.shape.gii')
                )
            )
            T1 = os.path.join(anat_path, f'sub-{participant}{ses_str_}_desc-preproc_T1w.nii.gz')

        parcellation_config = deepcopy(config)
        if mask_path:
            parcellation_config['mask_path'] = mask_path
        if xfm_path:
            parcellation_config['xfm_path'] = xfm_path
        if surfaces:
            parcellation_config['surface'] = surfaces
            if is_surface:
                parcellation_config['surface_left_path'] = surfaces['pial']['left']
                parcellation_config['surface_right_path'] = surfaces['pial']['right']
        if T1 is not None:
            parcellation_config['reference_image_path'] = T1
        if is_surface:
            parcellation_config['functional_paths'] = []
        else:
            parcellation_config['sample']['main']['functional_paths'] = []

        if not 'session' in nodes:
            nodes['session'] = {}
        if not session in nodes['session']:
            nodes['session'][session] = deepcopy(parcellation_config)
        if is_surface:
            nodes['session'][session]['functional_paths'] = functional_paths
        else:
            nodes['session'][session]['sample']['main']['functional_paths'] = functional_paths
        if not 'subject' in nodes:
            nodes['subject'] = deepcopy(parcellation_config)
        if is_surface:
            nodes['subject']['functional_paths'] += functional_paths
        else:
            nodes['subject']['sample']['main']['functional_paths'] += functional_paths

    parcellation_dir = os.path.join(derivatives_path, 'parcellate', parcellation_label)
    cliargs = []
    for node in nodes:
        node_dir = os.path.join(parcellation_dir, f'node-{node}')
        participant_dir = os.path.join(node_dir, 'sub-%s' % participant)
        if node == 'subject':
            sessions = {None}
        else:
            sessions = set(nodes[node].keys())
        for session in sessions:
            if node == 'session':
                if session:
                    session_dir = os.path.join(participant_dir, f'ses-{session}')
                else:
                    session_dir = participant_dir
                config_ = nodes[node][session]
            else:
                session_dir = participant_dir
                config_ = nodes[node]
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            config_['output_dir'] = session_dir
            if is_surface:
                del config_['surface']
                del config_['mask_path']

            config_path = os.path.join(session_dir, 'config.yml')
            with open(config_path, 'w') as f:
                yaml.dump(config_, f)
            cliargs.append(config_path)

    for cliarg in cliargs:
        if is_surface:
            # Surface parcellation
            with open(cliarg, 'r') as f:
                config_ = yaml.safe_load(f)
            parcellate_surface(**config_)
        else:
            cmd = f'python -m parcellate.bin.train {cliarg}'
            stderr(cmd + '\n')
            status = os.system(cmd)
            if status:
                stderr('Error during parcellation. Exiting.\n')
                exit(status)


