import json
import yaml
import multiprocessing
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from tempfile import TemporaryDirectory
from PIL import Image
from nilearn import image, surface, plotting, datasets
import argparse

from climbprep.constants import *
from climbprep.util import *


def plot(
        statmap_path,
        contrast_path,
        plot_path,
        mesh,
        white,
        midthickness,
        sulc,
        threshold=None,
        vmax=None,
        engine='matplotlib',
        scale=1.5,
        htrim=0.1,
        vtrim=0.1
):
    with TemporaryDirectory() as tmp_dir:
        stderr(f'Plotting statmap {statmap_path}\n')
        statmap_nii = image.load_img(os.path.join(contrast_path, statmap_path))

        cbar_img = None
        imgs = [None] * 4
        i = 0
        out_path_base = os.path.basename(statmap_path)[:-len(PLOT_STATMAP_SUFFIX)]
        for hemi in ('left', 'right'):
            statmap = surface.vol_to_surf(
                statmap_nii,
                mesh.parts[hemi],
                inner_mesh=white.parts[hemi],
                depth=np.linspace(0.0, 1.0, 10)
            )
            for view in ('lateral', 'medial'):
                colorbar = hemi == 'right' and view == 'lateral'
                fig = plotting.plot_surf(
                    surf_mesh=midthickness,
                    surf_map=statmap,
                    bg_map=sulc.parts[hemi],
                    hemi=hemi,
                    view=view if engine == 'matplotlib' else None,
                    bg_on_data=True,
                    threshold=threshold,
                    vmax=vmax,
                    colorbar=True,
                    cmap='coolwarm',
                    symmetric_cmap=True,
                    engine=engine
                )
                if engine == 'plotly':
                    fig = fig.figure
                    fig.update_traces(lighting=PLOT_LIGHTING, lightposition=PLOT_LIGHTPOSITION)
                    camera = fig.layout.scene.camera
                    if view == 'medial':
                        camera.eye.x = -camera.eye.x * 1.2
                    else:
                        camera.eye.x = camera.eye.x * 1.05
                if colorbar:
                    cbar_path = os.path.join(tmp_dir, out_path_base + f'_cbar.png')
                    if engine == 'plotly':
                        cbar = go.Figure(fig)
                        cbar.data = cbar.data[1:]
                        cbar.write_image(
                            cbar_path,
                            scale=scale
                        )
                    elif engine == 'matplotlib':
                        cbar = plotting.plot_surf(
                            surf_mesh=midthickness,
                            surf_map=statmap,
                            bg_map=sulc.parts[hemi],
                            hemi=hemi,
                            view=view if engine == 'matplotlib' else None,
                            bg_on_data=True,
                            threshold=threshold,
                            vmax=vmax,
                            colorbar=True,
                            cmap='coolwarm',
                            symmetric_cmap=True,
                            engine=engine
                        )
                        cbar.axes[0].remove()
                        cbar.savefig(
                            cbar_path,
                            dpi=300 * scale
                        )
                        plt.close(cbar)
                    else:
                        raise ValueError(f'Unknown plotting engine: {engine}')
                    cbar_img = Image.open(cbar_path)
                    w, h = cbar_img.size
                    l, t, r, b = w * 5 / 6, h * vtrim, w, h * (1 - vtrim)
                    cbar_img = cbar_img.crop((l, t, r, b))
                fig_path = os.path.join(tmp_dir, out_path_base + f'_hemi-{hemi}_view-{view}.png')
                if engine == 'plotly':
                    fig.data = fig.data[:1]
                    fig.write_image(
                        fig_path,
                        scale=scale
                    )
                elif engine == 'matplotlib':
                    fig.axes[1].remove()
                    fig.savefig(
                        fig_path,
                        dpi=300 * scale
                    )
                    plt.close(fig)
                else:
                    raise ValueError(f'Unknown plotting engine: {engine}')
                img = Image.open(fig_path)
                w, h = img.size
                l, t, r, b = w * htrim, h * vtrim, \
                             w * (1 - htrim), h * (1 - vtrim)
                img = img.crop((l, t, r, b))
                imgs[PLOT_IMG_ORDER[i]] = img
                i += 1

        if cbar_img:
            imgs.append(cbar_img)
        widths, heights = zip(*(i.size for i in imgs))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        img_path = os.path.join(plot_path, out_path_base + '.png')
        new_im.save(img_path)
        stderr(f'    Finished plotting statmap {statmap_path}\n')


def _plot(kwargs):
    try:
        plot(**kwargs)
    except Exception as e:
        stderr(f'Error plotting statmap {kwargs["statmap_path"]}\n')
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot firstlevels for a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-c', '--config', default=PLOT_DEFAULT_KEY, help=('Keyword (currently `main`) '
        'or YAML config file to used to parameterize preprocessing. If a keyword is provided, will '
        'the default settings for associated with that keyword. '
        'The possible config fields and values are just the `fmriprep` command-line arguments and their possible'
        'values. For details, see the `fmriprep` documentation.'))
    argparser.add_argument('--ncpus', type=int, default=8, help='Number of parallel processes to use.')
    args = argparser.parse_args()

    participant = args.participant
    ncpus = args.ncpus

    config = args.config
    if config in CONFIG['plot']:
        plot_label = config
        config_default = CONFIG['plot'][config]
        config = {}
    else:
        n = len(PLOT_STATMAP_SUFFIX)
        assert config.endswith(PLOT_STATMAP_SUFFIX), 'config must either be a known keyword or a file ending in ' \
                f'{PLOT_STATMAP_SUFFIX}'
        assert os.path.exists(config), ('Provided config (%s) does not match any known keyword or any existing '
                                        'filepath. Please provide a valid config.' % config)
        plot_label = config[:-n]
        config_default = CONFIG['plot'][PLOT_DEFAULT_KEY]
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    config = {x: config.get(x, config_default[x]) for x in config_default}
    assert 'model_label' in config, 'Required field `model_label` not found in config. ' \
                                    'Please provide a valid config file or keyword.'
    model_label = config.pop('model_label')

    # Set paths
    project = args.project
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path
    models_path = os.path.join(derivatives_path, 'firstlevels', model_label)
    assert os.path.exists(models_path), 'Path not found: %s' % models_path

    kwargs_all = []
    for model_subdir in os.listdir(models_path):
        model_path = os.path.join(models_path, model_subdir)
        if not os.path.isdir(model_path):
            continue
        dataset_description_path = os.path.join(model_path, 'dataset_description.json')
        assert os.path.exists(dataset_description_path), \
            'Dataset description file not found: %s' % dataset_description_path
        with open(dataset_description_path, 'r') as f:
            dataset_description = json.load(f)
        preprocessing_label = os.path.basename(
            dataset_description['PipelineDescription']['Parameters']['derivatives'][0]
        )
        space = dataset_description['PipelineDescription']['Parameters']['space']
        if not ('mni' in space.lower() or space == 'T1w' or space == 'anat'):
            raise ValueError(f'Plotting not supported for space {space}')
        if 'mni' in space.lower():
            mni_space = True
        else:
            mni_space = False
        fmriprep_path = os.path.join(derivatives_path, 'fmriprep', preprocessing_label)
        assert os.path.exists(fmriprep_path), 'Path not found: %s' % fmriprep_path
        anat_path = os.path.join(fmriprep_path, f'sub-{participant}', 'anat')
        assert os.path.exists(anat_path), 'Path not found: %s' % anat_path

        for node in ('subject', 'session', 'run'):
            node_dir = os.path.join(model_path, f'node-{node}')
            if not os.path.exists(node_dir):
                continue
            subdir = f'sub-{participant}'
            participant_dir = os.path.join(node_dir, subdir)
            sessions = set([x[4:] for x in os.listdir(os.path.join(participant_dir)) if x.startswith('ses-')])
            if not sessions:
                sessions = {None}
            for session in sessions:
                if session:
                    contrast_path = os.path.join(participant_dir, 'ses-%s' % session)
                    subdir_ = os.path.join(subdir, f'ses-{session}')
                else:
                    subdir_ = subdir
                    contrast_path = participant_dir

                plot_path = os.path.join(
                    derivatives_path,
                    'firstlevel_plots',
                    plot_label,
                    model_subdir,
                    f'node-{node}',
                    subdir_
                )
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                if mni_space:
                    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
                    mesh_in = (
                        fsaverage['pial_left'],
                        fsaverage['pial_right']
                    )
                    # Midthickness not supported by fsaverage, just use pial
                    midthickness_in = (
                        fsaverage['pial_left'],
                        fsaverage['pial_right']
                    )
                    white_in = (
                        fsaverage['white_left'],
                        fsaverage['white_right']
                    )
                    sulc_in = (
                        fsaverage['sulc_left'],
                        fsaverage['sulc_right']
                    )
                else:
                    mesh_in = (
                        os.path.join(anat_path, f'sub-{participant}_hemi-L_pial.surf.gii'),
                        os.path.join(anat_path, f'sub-{participant}_hemi-R_pial.surf.gii')
                    )
                    midthickness_in = (
                        os.path.join(anat_path, f'sub-{participant}_hemi-L_midthickness.surf.gii'),
                        os.path.join(anat_path, f'sub-{participant}_hemi-R_midthickness.surf.gii')
                    )
                    white_in = (
                        os.path.join(anat_path, f'sub-{participant}_hemi-L_white.surf.gii'),
                        os.path.join(anat_path, f'sub-{participant}_hemi-R_white.surf.gii')
                    )
                    sulc_in = (
                        os.path.join(anat_path, f'sub-{participant}_hemi-L_sulc.shape.gii'),
                        os.path.join(anat_path, f'sub-{participant}_hemi-R_sulc.shape.gii')
                    )
                mesh = surface.PolyMesh(*mesh_in)
                midthickness = surface.PolyMesh(*midthickness_in)
                white = surface.PolyMesh(*white_in)
                sulc = surface.PolyData(*sulc_in)

                statmap_paths = []
                for x in os.listdir(contrast_path):
                    if x.endswith('stat-t_statmap.nii.gz') or x.endswith(f'stat-z_statmap{PLOT_STATMAP_SUFFIX}'):
                        statmap_paths.append(os.path.join(contrast_path, x))

                for statmap_path in statmap_paths:
                    stat = STAT_RE.match(statmap_path)
                    if not stat:
                        continue
                    stat = stat.group(1)
                    kwargs = dict(
                        statmap_path=statmap_path,
                        contrast_path=contrast_path,
                        plot_path=plot_path,
                        mesh=mesh,
                        white=white,
                        midthickness=midthickness,
                        sulc=sulc,
                        vmax=config['vmax'][stat],
                        threshold=config['threshold'][stat],
                        engine=config['engine'],
                        htrim=config['htrim'],
                        vtrim=config['vtrim']
                    )
                    kwargs_all.append(kwargs)

    pool = multiprocessing.Pool(ncpus)
    pool.map(_plot, kwargs_all)