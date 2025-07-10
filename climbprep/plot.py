import time
import json
import yaml
import multiprocessing
from functools import lru_cache
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb, to_hex
from plotly import io as pio
from plotly import graph_objects as go
from tempfile import TemporaryDirectory
from PIL import Image
import pyvista
from nilearn import image, surface, plotting, datasets, maskers
import h5py
import argparse

from climbprep.constants import *
from climbprep.util import *


def plot_surface(
        pial,
        white,
        midthickness,
        sulc=None,
        statmaps=None,
        statmap_labels=None,
        cmaps=None,  # Defaults to single color per statmap, only makes sense if statmap_scales_alpha=True
        vmin=None,
        vmax=None,
        thresholds=None,
        statmap_scales_alpha=True,
        plot_path=None,
        display_surface='midthickness',
        colorbar=True,
        hide_min=None,
        hide_max=None,
        scale=1,
        htrim=0.1,
        vtrim=0.1,
        additive_color=False,
        turn_out_hemis=False,
        bg_brightness=0.95,
        sulc_alpha=0.9,
        medial_zoom=1.15
):
    with TemporaryDirectory() as tmp_dir:
        zoom_factor = dict(
            left=dict(
                lateral=1.5 if turn_out_hemis else 1.0,
                medial=-medial_zoom
            ),
            right=dict(
                lateral=-1.5 if turn_out_hemis else -1.0,
                medial=medial_zoom
            )
        )

        statmap_in = statmaps
        statmap_niis = []
        if statmap_in is None:
            statmap_in = []
        elif not hasattr(statmap_in, '__iter__') or isinstance(statmap_in, str):
            statmap_in = [statmap_in]
        seeds = []
        for statmap_in_ in statmap_in:
            if isinstance(statmap_in_, dict):
                assert 'functionals' in statmap_in_, \
                    'If `statmaps` is a dict, it must contain a key "functionals" with the paths to the timecourses.'
                assert 'masker' in statmap_in_, \
                    'If `statmaps` is a dict, it must contain a key "masker" with the NiftiMasker object.'
                assert 'seed' in statmap_in_, \
                    'If `statmaps` is a dict, it must contain a key "seed" with the seed coordinates (x, y, z).'
                functionals = statmap_in_['functionals']
                masker = statmap_in_['masker']
                x, y, z = statmap_in_['seed']
                fwhm = statmap_in_.get('fwhm', None)
                statmap_in_ = connectivity_from_seed(
                    x, y, z, functionals, masker, fwhm=fwhm
                )
                seeds.append(generate_sphere(
                    center=(x, y, z),
                    radius=1
                ))
            statmap_niis.append(image.load_img(statmap_in_))

        if statmap_labels is None:
            statmap_labels = [None] * len(statmap_in)
        elif not hasattr(statmap_labels, '__iter__') or isinstance(statmap_labels, str):
            statmap_labels = [statmap_labels]
        assert len(statmap_labels) == len(statmap_in), \
            '`statmap_labels` must either be `None` or a list of equal length to `statmaps`.'

        if cmaps is None:
            cmaps = [None] * len(statmap_in)
        elif not hasattr(cmaps, '__iter__') or isinstance(cmaps, str):
            cmaps = [cmaps]
        assert len(cmaps) == len(statmap_niis), \
            '`cmaps` must either be `None` or a list of equal length to `statmaps.'

        if vmin is None:
            vmin = [None] * len(statmap_in)
        elif not hasattr(vmin, '__iter__'):
            vmin = [vmin] * len(statmap_in)
        assert len(vmin) == len(statmap_niis), \
            '`vmin` must either be a single value or a list of equal length to `statmaps.`'
        
        if vmax is None:
            vmax = [None] * len(statmap_in)
        elif not hasattr(vmax, '__iter__'):
            vmax = [vmax] * len(statmap_in)
        assert len(vmax) == len(statmap_niis), \
            '`vmax` must either be a single value or a list of equal length to `statmaps.`'
        
        if thresholds is None:
            thresholds = [None] * len(statmap_in)
        elif not hasattr(thresholds, '__iter__'):
            thresholds = [thresholds] * len(statmap_in)
        assert len(thresholds) == len(statmap_niis), \
            '`threshold` must either be a single value or a list of equal length to `statmaps.`'

        if not hasattr(statmap_scales_alpha, '__iter__'):
            statmap_scales_alpha = [statmap_scales_alpha]
        statmap_scales_alpha = statmap_scales_alpha * len(statmap_in)

        mesh_kwargs = {}
        surf_types = dict(pial=pial, white=white, midthickness=midthickness, sulc=sulc)
        for surf_type in surf_types:
            if surf_type is None:
                mesh_kwargs[f'{surf_type}_{hemi}'] = None
            else:
                for hemi in ('left', 'right'):
                    if hasattr(surf_types[surf_type], 'parts'):
                        surf = surf_types[surf_type].parts[hemi]
                    else:
                        surf = surf_types[surf_type][hemi]
                    mesh_kwargs[f'{surf_type}_{hemi}'] = surf
        pial, white, midthickness, sulc = get_plot_meshes(**mesh_kwargs)

        cbar_img = None
        imgs = [None] * 4
        ix = 0
        if plot_path:
            out_path_base, ext = os.path.splitext(os.path.basename(plot_path))
        else:
            out_path_base = ext = None

        colorbar_written = False
        if plot_path:
            cbar_step = 0.2
        else:
            cbar_step = 0.1

        fig = go.Figure()
        fig.layout.paper_bgcolor = 'white'
        fig.layout.hovermode = 'closest'
        fig.layout.scene = {
            'dragmode': 'turntable',
            **{f"{dim}axis": PLOT_AXIS_CONFIG for dim in ("x", "y", "z")},
            'aspectmode': 'data'
        }
        fig.layout.margin = dict(l=0, r=0, b=0, t=0, pad=0)
        fig.layout.scene.camera = dict(
            eye=dict(x=-1.575, y=0, z=0),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
        )

        camera = fig.layout.scene.camera

        traces = {}
        for hemi in ('left', 'right'):

            # Mesh3d
            if display_surface == 'midthickness':
                surface_ = midthickness
            elif display_surface == 'white':
                surface_ = white
            elif display_surface == 'pial':
                surface_ = pial
            else:
                raise ValueError(f'Invalid display_surface: {display_surface}. '
                                 'Must be one of "midthickness", "white", or "pial".')
            trace, customdata = get_plot_Mesh3d(surface_=surface_, hemi=hemi, turn_out_hemis=turn_out_hemis)

            # Background colors
            bgcolors = get_plot_bgcolors(
                len(trace.x),
                hemi,
                sulc=sulc,
                bg_brightness=bg_brightness,
                sulc_alpha=sulc_alpha
            )

            # Statmap colors
            plot_colors = iter(PLOT_COLORS)
            vertexcolor = None
            vertexalpha = None
            vertexscale = None
            cbars = []
            cbar_x = 1
            for statmap_nii, cmap, label, vmin_, vmax_, threshold_, statmap_scales_alpha_ in \
                        zip(statmap_niis, cmaps, statmap_labels, vmin, vmax, thresholds, statmap_scales_alpha):
                statmap = get_statmap_surface(statmap_nii=statmap_nii, pial=pial, white=white, hemi=hemi)
                col = label if label else 'val'
                customdata[col] = statmap
                if vmin_ is None:
                    vmin_ = np.nanmin(statmap)
                if vmax_ is None:
                    vmax_ = np.nanmax(statmap)
                cmin = vmin_
                cmax = vmax_
                if cmin < 0 < cmax: # If both positive and negative values are present, center around 0
                    mag = max(np.abs(cmin), np.abs(cmax))
                    cmin = -mag
                    cmax = mag
                    if hide_min is None:
                        hide_min = False
                    if hide_max is None:
                        hide_max = False
                elif cmin >= 0:  # If only positive values are present
                    if hide_min is None:
                        hide_min = True
                    if hide_max is None:
                        hide_max = False
                elif cmax <= 0:  # If only negative values are present
                    if hide_min is None:
                        hide_min = False
                    if hide_max is None:
                        hide_max = True
                if hide_min:
                    statmap = np.where(statmap < vmin_, np.nan, statmap)
                if hide_max:
                    statmap = np.where(statmap > vmax_, np.nan, statmap)
                statmap_abs = np.abs(statmap)
                if threshold_ is not None:
                    assert threshold_ >= 0, 'Threshold must be non-negative'
                    statmap = np.where(statmap_abs < threshold_, np.nan, statmap)
                statmap = np.clip(statmap, cmin, cmax)
                statmap /= cmax - cmin
                statmap_abs = np.abs(statmap)
                statmap -= cmin / (cmax - cmin)

                if cmap is None:
                    try:
                        color = next(plot_colors)
                    except StopIteration:
                        raise ValueError('Not enough colors in `climbprep.constants.PLOT_COLORS` to plot all statmaps. '
                                         'Please provide a list of colors for `cmaps`.')
                    color = to_rgb(color)
                    cmap = LinearSegmentedColormap.from_list(f'{color}', [color, color])
                elif isinstance(cmap, str):
                    try:
                        cmap = plt.get_cmap(cmap)
                    except ValueError:
                        if isinstance(cmap, str):
                            color = to_rgb(cmap)
                        else:
                            color = cmap
                        cmap = LinearSegmentedColormap.from_list(f'{cmap}', [color, color])
                cmap.set_extremes(bad=[np.nan]*4, over=[np.nan]*4, under=[np.nan]*4)

                vertexcolor_ = cmap(statmap)[..., :3]
                if statmap_scales_alpha_:
                    vertexcolor_ = vertexcolor_ * statmap_abs[..., None]
                if vertexcolor is None:
                    vertexcolor = vertexcolor_
                else:
                    vertexcolor = np.where(np.isnan(vertexcolor), vertexcolor_,
                                           np.where(np.isnan(vertexcolor_), vertexcolor, vertexcolor + vertexcolor_))
                if statmap_scales_alpha_:
                    vertexalpha_ = statmap_abs[..., None]
                else:
                    vertexalpha_ = np.isfinite(statmap_abs)[..., None].astype(statmap_abs.dtype)
                if vertexalpha is None:
                    vertexalpha = vertexalpha_
                else:
                    vertexalpha = np.fmax(vertexalpha, vertexalpha_)

                if vertexscale is None:
                    vertexscale = vertexalpha_
                else:
                    vertexscale = np.where(np.isnan(vertexscale), vertexalpha_,
                                           np.where(np.isnan(vertexalpha_), vertexscale, vertexscale + vertexalpha_))

                if colorbar:
                    cbar = go.Mesh3d()
                    cbar.cmax = cmax
                    cbar.cmin = cmin
                    cbar.colorbar = dict(len=0.5, tickfont=dict(color='black', size=25), tickformat='.1f', x=cbar_x)
                    if label:
                        cbar.colorbar.title = dict(text=label, side='right')
                    colorscale_ix = np.linspace(cmin, cmax, 256) / (cmax - cmin)
                    if statmap_scales_alpha_:
                        colorscale_alpha = np.abs(colorscale_ix)
                    else:
                        colorscale_alpha = np.ones_like(colorscale_ix)
                    if threshold_ is not None:
                        colorscale_alpha = np.where(colorscale_alpha < threshold_ / (cmax - cmin), 0., colorscale_alpha)
                    colorscale_ix -= cmin  / (cmax - cmin)
                    colorscale = cmap(colorscale_ix)
                    colorscale[..., 3] = colorscale_alpha
                    colorscale_ = []
                    for colorscale_ix_, color in zip(colorscale_ix, colorscale):
                        if np.all(np.isfinite(color)):
                            color = 'rgba(' + ', '.join([f'{round(c*255)}' for c in color[:3]] + [f'{color[3]}']) + ')'
                            colorscale_.append([colorscale_ix_, color])
                    colorscale = colorscale_
                    cbar.colorscale = colorscale
                    cbar.i = [0]
                    cbar.j = [0]
                    cbar.k = [0]
                    cbar.intensity = [0]
                    cbar.opacity = 0
                    cbar.x = [1, 0, 0]
                    cbar.y = [0, 1, 0]
                    cbar.z = [0, 0, 1]
                    cbars.append(cbar)
                    cbar_x += cbar_step

            if vertexcolor is None:
                vertexcolor = bgcolors
            else:
                if additive_color:
                    vertexcolor = np.clip(vertexcolor, 0, 1)
                else:
                    vertexcolor /= vertexscale

                if vertexcolor is None:
                    vertexcolor = bgcolors
                else:
                    vertexalpha *= 1 - PLOT_BG_ALPHA
                    vertexalpha = np.where(np.isnan(vertexalpha), 0, vertexalpha)
                    vertexcolor = np.where(np.isnan(vertexcolor), 0, vertexcolor)
                    vertexcolor = bgcolors * (1 - vertexalpha) + vertexcolor * vertexalpha

            trace.vertexcolor = tuple([to_hex(c) for c in vertexcolor])
            trace.customdata = customdata
            trace.hovertemplate = ''.join(['<b>' + col + ':</b> %{customdata[' + str(i) + ']:.2f}<br>'
                                           for i, col in enumerate(customdata.columns)]) + '<extra></extra>'
            traces[hemi] = trace

            if plot_path:
                fig.add_traces(traces[hemi])
                for view in ('lateral', 'medial'):
                    zoom = zoom_factor[hemi][view]
                    camera.eye.x = camera.eye.x * zoom
                    fig_path = os.path.join(tmp_dir, out_path_base + f'_hemi-{hemi}_view-{view}{ext}')
                    fig.write_image(
                        fig_path,
                        scale=scale
                    )
                    img = Image.open(fig_path)
                    w, h = img.size
                    l, t, r, b = w * htrim, h * vtrim, \
                                 w * (1 - htrim), h * (1 - vtrim)
                    img = img.crop((l, t, r, b))
                    imgs[PLOT_IMG_ORDER[ix]] = img
                    ix += 1
                    camera.eye.x = camera.eye.x / zoom

                fig.data = []

                if colorbar and not colorbar_written:
                    cbar_path = os.path.join(tmp_dir, out_path_base + f'_cbar{ext}')
                    cbar_fig = go.Figure(fig)
                    cbar_fig.data = cbar_fig.data[:0]
                    cbar_fig.add_traces(cbars)
                    cbar_fig.layout.width = pio.kaleido.scope.default_width * cbar_x
                    cbar_fig.write_image(
                        cbar_path,
                        scale=scale
                    )
                    cbar_img = Image.open(cbar_path)
                    w, h = cbar_img.size
                    l, t, r, b = w * 1 / cbar_x, h * vtrim, w, h * (1 - vtrim)
                    cbar_img = cbar_img.crop((l, t, r, b))
                    colorbar_written = True

        if plot_path:
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
            img_path = plot_path
            new_im.save(img_path)

        fig.add_traces([traces['left'], traces['right']] + cbars + seeds)
        fig.update_traces(lighting=PLOT_LIGHTING, lightposition=PLOT_LIGHTPOSITION)

        return fig

@lru_cache(maxsize=1)
def get_plot_meshes(
        pial_left=None,
        pial_right=None,
        white_left=None,
        white_right=None,
        midthickness_left=None,
        midthickness_right=None,
        sulc_left=None,
        sulc_right=None,
        display_surface='midthickness',
        percent=None,
        face_count=None
):
    if sulc_left is not None or sulc_right is not None:
        sulc = surface.PolyData(left=sulc_left, right=sulc_right)
    else:
        sulc = None
    if pial_left is not None or pial_right is not None:
        pial, sulc_ = resample_mesh(
            surface.PolyMesh(left=pial_left, right=pial_right),
            surf_data=sulc if sulc is not None and display_surface == 'pial' else None,
            percent=percent,
            face_count=face_count
        )
        if sulc_ is not None:
            sulc = sulc_
    else:
        pial = None
    if white_left is not None or white_right is not None:
        white, sulc_ = resample_mesh(
            surface.PolyMesh(left=white_left, right=white_right),
            surf_data=sulc if sulc is not None and display_surface == 'white' else None,
            percent=percent,
            face_count=face_count
        )
        if sulc_ is not None:
            sulc = sulc_
    else:
        white = None
    if midthickness_left is not None or midthickness_right is not None:
        midthickness, sulc_ = resample_mesh(
            surface.PolyMesh(left=midthickness_left, right=midthickness_right),
            surf_data=sulc if sulc is not None and display_surface == 'midthickness' else None,
            percent=percent,
            face_count=face_count
        )
        if sulc_ is not None:
            sulc = sulc_
    else:
        midthickness = None

    return pial, white, midthickness, sulc


def resample_mesh(
        mesh,
        surf_data=None,
        percent=None,
        face_count=None
):
    '''Doesn't work good, just pass through with `None` for all kwargs'''

    assert not (percent and face_count), 'Either percent or face_count must be specified, not both.'

    if not (percent or face_count):
        return mesh, surf_data

    has_surf_data = surf_data is not None
    out_mesh = {}
    out_surf = {}
    for hemi in ('left', 'right'):
        mesh_ = mesh.parts[hemi]
        if percent is not None:
            assert 0 < percent < 100, 'percent_downsample must be between 0 and 100'
            target_reduction = percent / 100
            resample = True
        elif face_count is not None:
            assert face_count > 0, 'face_count must be greater than 0'
            target_reduction = face_count / mesh_.coordinates.shape[0]
            resample = True
        else:
            target_reduction = 1
            resample = False

        if resample:
            faces_pyvista_format = np.hstack(
                (
                    np.full((mesh_.faces.shape[0], 1), 3),
                    mesh_.faces
                )
            ).flatten()
            mesh_ = pyvista.PolyData(
                mesh_.coordinates,
                faces=faces_pyvista_format,
            )
            mesh_.clear_data()
            if surf_data is not None:
                mesh_.point_data['surf_data'] = surf_data.parts[hemi]
            mesh_ = mesh_.decimate(target_reduction=target_reduction, scalars=True)

            faces_nilearn_format = mesh_.faces.reshape((-1, 4))[:, 1:]
            out_mesh[hemi] = surface.InMemoryMesh(
                coordinates=mesh_.points,
                faces=faces_nilearn_format
            )
            if has_surf_data:
                out_surf[hemi] = np.array(mesh_.point_data['surf_data'])
        else:
            out_mesh[hemi] = mesh_
            if has_surf_data:
                out_surf[hemi] = surf_data.parts[hemi]

    out_mesh = surface.PolyMesh(**out_mesh)
    if has_surf_data:
        out_surf = surface.PolyData(**out_surf)
    else:
        out_surf = None

    return out_mesh, out_surf


def get_plot_Mesh3d(surface_, hemi, turn_out_hemis=False):
    x, y, z = surface_.parts[hemi].coordinates.T
    i, j, k = surface_.parts[hemi].faces.T
    customdata = pd.DataFrame(dict(  # Store true coordinates for hover before messing with them
        x=x,
        y=y,
        z=z,
    ))

    # Turn out hemispheres
    if turn_out_hemis:
        x = np.abs(x)
        y -= y.min() - 10
        if hemi == 'right':
            y *= -1
    trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
    trace.lighting = PLOT_LIGHTING
    trace.lightposition = PLOT_LIGHTPOSITION

    return trace, customdata


def get_plot_bgcolors(nvertices, hemi, sulc=None, bg_brightness=0.95, sulc_alpha=0.9):
    bgcolors = np.ones((nvertices, 3)) * bg_brightness
    if sulc is not None:
        sulc_data = sulc.parts[hemi]
        sulc_vmin, sulc_vmax = np.min(sulc_data), np.max(sulc_data)
        if sulc_vmin < 0 or sulc_vmax > 1:
            bg_norm = Normalize(vmin=sulc_vmin, vmax=sulc_vmax)
            sulc_data = bg_norm(sulc_data)
        sulccolors = plt.get_cmap("Greys")(sulc_data)[..., :3]
        bgcolors = bgcolors * (1 - sulc_alpha) + sulccolors * sulc_alpha

    return bgcolors


def get_statmap_surface(statmap_nii, pial, white=None, hemi='left'):
    statmap = surface.vol_to_surf(
        statmap_nii,
        pial.parts[hemi],
        inner_mesh=white.parts[hemi],
        depth=np.linspace(0.0, 1.0, 10)
    )
    return statmap


@lru_cache(maxsize=1)
def get_functionals_and_masker(
        participant,
        project='climblab',
        session=None,
        cleaning_label=CLEAN_DEFAULT_KEY,
        space=PARCELLATE_DEFAULT_KEY,
        debug=False
):
    if session:
        sessions = {session}
    else:
        sessions = set(
            [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project, f'sub-{participant}'))
             if x.startswith('ses-')]
        )
        if not sessions:
            sessions = {None}
    functionals = []
    for session in sessions:
        cleaned_dir = os.path.join(
            BIDS_PATH, project, 'derivatives', 'cleaned', cleaning_label, f'sub-{participant}'
        )
        if session:
            cleaned_dir = os.path.join(cleaned_dir, f'ses-{session}')
        functionals += sorted([os.path.join(cleaned_dir, x) for x in os.listdir(cleaned_dir) if
                               x.endswith(f'_space-{space}_desc-clean_bold.nii.gz')])

    if not functionals:
        return [], None

    if debug:
        functionals = functionals[:1]

    masker = maskers.NiftiMasker()
    boldref = image.load_img(functionals[0].replace('_desc-preproc_bold', '_boldref'))
    masker.fit(boldref)
    functionals_ = []
    for functional in functionals:
        functionals_.append(masker.transform(image.resample_to_img(functional, masker.mask_img_)))
    functionals = np.concatenate(functionals_, axis=0)  # shape (n_timepoints, n_kept_voxels)

    return functionals, masker


def standardize_timecourse(arr, axis=0):
    return (arr - np.mean(arr, axis=axis, keepdims=True)) / np.std(arr, axis=axis, keepdims=True)


def unmask(arr, mask_nii):
    # arr.shape -> (n_timepoints, n_kept_voxels) OR (n_kept_voxels,)
    if arr.ndim == 2:
        T = arr.shape[0]
    else:
        T = 0
    mask = image.get_data(mask_nii).astype(bool)
    shape = tuple(mask.shape)
    if T:
        shape = shape + (T,)
        arr = arr.T
    out = np.zeros(shape, dtype=arr.dtype)
    out[mask] = arr
    nii = image.new_img_like(mask_nii, out)

    return nii


def connectivity_from_seed(x, y, z, functionals, masker, fwhm=None):
    mask_img = masker.mask_img_
    ix = np.where(image.get_data(mask_img))
    X, Y, Z = image.coord_transform(*ix, mask_img.affine)
    seed = np.array([x, y, z])
    distances = np.sqrt(np.sum((np.array([X, Y, Z]).T - seed) ** 2, axis=1))
    if fwhm:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        seed_timecourse = functionals @ weights
    else:
        ix = np.argmin(distances)
        seed_timecourse = functionals[:, ix]
    seed_timecourse = standardize_timecourse(seed_timecourse)
    other_timecourses = standardize_timecourse(functionals)
    n = len(seed_timecourse)

    connectivity = other_timecourses.T @ seed_timecourse / n
    connectivity = unmask(connectivity, mask_img)

    return connectivity


def generate_sphere(center, radius, opacity=0.7):
    sphere_pv = pyvista.Sphere(radius=radius, center=center)
    x, y, z = sphere_pv.points.T
    i, j, k = sphere_pv.faces.reshape((-1, 4))[:, 1:].T
    sphere = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='white', opacity=opacity)

    return sphere





def _plot(kwargs):
    try:
        plot_surface(**kwargs)
    except Exception as e:
        stderr(f'Error plotting statmap {kwargs["statmap_path"]}\n')
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot firstlevels for a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-m', '--models', default=[], nargs='*', help=('List of models to plot. '
                                                                          '(see `climbprep.constants.MODELFILES_PATH` '
                                                                          'If not specified, will plot all models '
                                                                          'available for the participant.'))
    argparser.add_argument('-c', '--config', default=PLOT_DEFAULT_KEY, help=('Keyword (currently `main`) '
        'or YAML config file to used to parameterize preprocessing. If a keyword is provided, will '
        'the default settings for associated with that keyword. '
        'The possible config fields and values are just the `fmriprep` command-line arguments and their possible'
        'values. For details, see the `fmriprep` documentation.'))
    argparser.add_argument('--ncpus', type=int, default=8, help='Number of parallel processes to use.')
    args = argparser.parse_args()

    participant = args.participant
    project = args.project
    models = args.models
    config = args.config
    ncpus = args.ncpus

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
    project_path = os.path.join(BIDS_PATH, project)
    assert os.path.exists(project_path), 'Path not found: %s' % project_path
    derivatives_path = os.path.join(project_path, 'derivatives')
    assert os.path.exists(derivatives_path), 'Path not found: %s' % derivatives_path
    models_path = os.path.join(derivatives_path, 'firstlevels', model_label)
    assert os.path.exists(models_path), 'Path not found: %s' % models_path

    stderr(f'Plotting outputs will be written to {os.path.join(derivatives_path, "firstlevel_plots", plot_label)}\n')

    kwargs_all = []
    for model_subdir in os.listdir(models_path):
        if models and model_subdir not in models:
            continue
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
                    pial = dict(
                        left=fsaverage['pial_left'],
                        right=fsaverage['pial_right']
                    )
                    # Midthickness not supported by fsaverage, just use pial
                    midthickness_in = dict(
                        left=fsaverage['pial_left'],
                        right=fsaverage['pial_right']
                    )
                    white = dict(
                        left=fsaverage['white_left'],
                        right=fsaverage['white_right']
                    )
                    sulc = dict(
                        left=fsaverage['sulc_left'],
                        right=fsaverage['sulc_right']
                    )
                else:
                    pial = dict(
                        left=os.path.join(anat_path, f'sub-{participant}_hemi-L_pial.surf.gii'),
                        right=os.path.join(anat_path, f'sub-{participant}_hemi-R_pial.surf.gii')
                    )
                    midthickness = dict(
                        left=os.path.join(anat_path, f'sub-{participant}_hemi-L_midthickness.surf.gii'),
                        right=os.path.join(anat_path, f'sub-{participant}_hemi-R_midthickness.surf.gii')
                    )
                    white = dict(
                        left=os.path.join(anat_path, f'sub-{participant}_hemi-L_white.surf.gii'),
                        right=os.path.join(anat_path, f'sub-{participant}_hemi-R_white.surf.gii')
                    )
                    sulc = dict(
                        left=os.path.join(anat_path, f'sub-{participant}_hemi-L_sulc.shape.gii'),
                        right=os.path.join(anat_path, f'sub-{participant}_hemi-R_sulc.shape.gii')
                    )

                statmap_paths = []
                for x in os.listdir(contrast_path):
                    if x.endswith('stat-t_statmap.nii.gz') or x.endswith(f'stat-z_statmap{PLOT_STATMAP_SUFFIX}'):
                        statmap_paths.append(os.path.join(contrast_path, x))

                for statmap_path in statmap_paths:
                    stat = STAT_RE.match(statmap_path)
                    contrast = CONTRAST_RE.match(statmap_path)
                    if not stat or not contrast:
                        continue
                    stat = stat.group(1)
                    contrast = contrast.group(1)
                    kwargs = dict(
                        pial=pial,
                        white=white,
                        midthickness=midthickness,
                        sulc=sulc,
                        statmaps=os.path.join(contrast_path, statmap_path),
                        statmap_labels=f'{contrast} ({stat})',
                        cmaps='coolwarm',
                        vmin=-5,
                        vmax=5,
                        thresholds=None,
                        statmap_scales_alpha=False,
                        plot_path=os.path.join(
                            plot_path,
                            os.path.basename(statmap_path)[:-len(PLOT_STATMAP_SUFFIX)] + '.png'
                        ),
                        display_surface='midthickness',
                        colorbar=True,
                        hide_min=True,
                        hide_max=False,
                        scale=config['scale'],
                        htrim=config['htrim'],
                        vtrim=config['vtrim'],
                    )
                    stderr(f'Plotting statmap {os.path.join(contrast_path, statmap_path)}\n')
                    plot_surface(**kwargs)
                    stderr(f'  Finished plotting statmap {os.path.join(contrast_path, statmap_path)}\n')
                    kwargs_all.append(kwargs)

    # pool = multiprocessing.Pool(ncpus)
    # pool.map(_plot, kwargs_all)