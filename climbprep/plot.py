import time
import json
import yaml
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
from nilearn import image, surface, plotting, datasets, masking, maskers
import diskcache
import argparse

from climbprep.constants import *
from climbprep.util import *


class PlotLib:

    CACHABLE = {
        'get_statmap_surface_and_color',
        'get_surface_mesh_hemi',
        'infer_midthickness_mesh_hemi',
        'get_surface_data',
        'get_plot_bgcolors',
        'get_mask',
        'get_functional',
        'get_connectivity_from_seed',
        'make_sphere'
    }

    def get_fig(
            self,
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
            hide_min=None,
            hide_max=None,
            display_surface='midthickness',
            colorbar=True,
            bg_brightness=PLOT_BG_BRIGHTNESS,
            sulc_alpha=PLOT_SULC_ALPHA,
            bg_alpha=PLOT_BG_ALPHA,
            additive_color=False,
            turn_out_hemis=False,
            cbar_step=0.1
    ):
        plot_data = self.get_plot_data(
            pial,
            white,
            midthickness,
            sulc=sulc,
            statmaps=statmaps,
            statmap_labels=statmap_labels,
            colors=cmaps,
            vmin=vmin,
            vmax=vmax,
            thresholds=thresholds,
            statmap_scales_alpha=statmap_scales_alpha,
            hide_min=hide_min,
            hide_max=hide_max,
            display_surface=display_surface,
            colorbar=colorbar,
            bg_brightness=bg_brightness,
            sulc_alpha=sulc_alpha,
            bg_alpha=bg_alpha,
            additive_color=additive_color,
            turn_out_hemis=turn_out_hemis
        )

        fig = self.plot_data_to_fig(
            **plot_data,
            cbar_step=cbar_step
        )

        return fig

    def get_plot_data(
            self,
            pial,
            white,
            midthickness,
            sulc=None,
            statmaps=None,
            statmap_labels=None,
            colors=None,  # Defaults to single color per statmap, only makes sense if statmap_scales_alpha=True
            vmin=None,
            vmax=None,
            thresholds=None,
            skip=False,
            statmap_scales_alpha=True,
            hide_min=None,
            hide_max=None,
            display_surface='midthickness',
            colorbar=True,
            bg_brightness=PLOT_BG_BRIGHTNESS,
            sulc_alpha=PLOT_SULC_ALPHA,
            bg_alpha=PLOT_BG_ALPHA,
            additive_color=False,
            turn_out_hemis=False,
            progress_fn=None
    ):
        statmaps_in = statmaps
        if statmaps_in is None:
            statmaps_in = []
        elif not hasattr(statmaps_in, '__iter__') or isinstance(statmaps_in, str):
            statmaps_in = [statmaps_in]
        seeds = []
        for statmap_in in statmaps_in:
            if isinstance(statmap_in, dict) and 'functionals' in statmap_in:
                assert 'seed' in statmap_in, \
                    'If statmap is connectivity, it must contain a key "seed" with the seed coordinates (x, y, z).'
                seeds.append(statmap_in['seed'])

        if statmap_labels is None:
            statmap_labels = [None] * len(statmaps_in)
        elif not hasattr(statmap_labels, '__iter__') or isinstance(statmap_labels, str):
            statmap_labels = [statmap_labels]
        assert len(statmap_labels) == len(statmaps_in), \
            '`statmap_labels` must either be `None` or a list of equal length to `statmaps`.'

        if colors is None:
            colors = [None] * len(statmaps_in)
        elif not hasattr(colors, '__iter__') or isinstance(colors, str):
            colors = [colors]
        assert len(colors) == len(statmaps_in), \
            '`cmaps` must either be `None` or a list of equal length to `statmaps.'

        if vmin is None:
            vmin = [None] * len(statmaps_in)
        elif not hasattr(vmin, '__iter__'):
            vmin = [vmin] * len(statmaps_in)
        assert len(vmin) == len(statmaps_in), \
            '`vmin` must either be a single value or a list of equal length to `statmaps.`'

        if vmax is None:
            vmax = [None] * len(statmaps_in)
        elif not hasattr(vmax, '__iter__'):
            vmax = [vmax] * len(statmaps_in)
        assert len(vmax) == len(statmaps_in), \
            '`vmax` must either be a single value or a list of equal length to `statmaps.`'

        if thresholds is None:
            thresholds = [None] * len(statmaps_in)
        elif not hasattr(thresholds, '__iter__'):
            thresholds = [thresholds] * len(statmaps_in)
        assert len(thresholds) == len(statmaps_in), \
            '`threshold` must either be a single value or a list of equal length to `statmaps.`'

        if not hasattr(statmap_scales_alpha, '__iter__'):
            statmap_scales_alpha = [statmap_scales_alpha] * len(statmaps_in)
        assert len(statmap_scales_alpha) == len(statmaps_in), \
            '`statmap_scales_alpha` must either be a single value or a list of equal length to `statmaps.`'

        if not hasattr(skip, '__iter__'):
            skip = [skip] * len(statmaps_in)
        assert len(skip) == len(statmaps_in), \
            '`skip` must either be a single value or a list of equal length to `statmaps.`'

        C = 0.8
        T = 0.9
        incr_meshes = 1 / (3 + len(statmaps_in) * C) * T
        incr_statmaps = (C / (3 + len(statmaps_in) * C) * T) / 2

        mesh_kwargs = {}
        surf_types = dict(pial=pial, white=white, midthickness=midthickness, sulc=sulc)
        for surf_type in surf_types:
            if surf_types[surf_type] is None:
                for hemi in ('left', 'right'):
                    if surf_type == 'midthickness':
                        mesh_kwargs[f'{surf_type}_{hemi}'] = 'infer'
                    else:
                        mesh_kwargs[f'{surf_type}_{hemi}'] = None
            else:
                for hemi in ('left', 'right'):
                    if hasattr(surf_types[surf_type], 'parts'):
                        surf = surf_types[surf_type].parts[hemi]
                    else:
                        surf = surf_types[surf_type][hemi]
                    mesh_kwargs[f'{surf_type}_{hemi}'] = surf
        if progress_fn is not None:
            progress_fn.incr = incr_meshes
            progress_fn('Loading surface meshes', 0)
        mesh_kwargs['progress_fn'] = progress_fn
        pial, white, midthickness, sulc = self.get_surface_meshes(**mesh_kwargs)

        if display_surface == 'midthickness':
            display_surface = midthickness
        elif display_surface == 'white':
            display_surface = white
        elif display_surface == 'pial':
            display_surface = pial
        else:
            raise ValueError(f'Invalid display_surface: {display_surface}. '
                             'Must be one of "midthickness", "white", or "pial".')

        if progress_fn is not None:
            progress_fn.incr = incr_statmaps
        out = {}
        for hemi in ('left', 'right'):
            # Background colors
            if progress_fn is not None:
                progress_fn(f'Loading {hemi} background', 0)
            bgcolors = self.get_plot_bgcolors(
                len(pial.parts[hemi].coordinates),
                hemi,
                sulc_left=surf_types['sulc']['left'],
                sulc_right=surf_types['sulc']['right'],
                bg_brightness=bg_brightness,
                sulc_alpha=sulc_alpha
            )

            # Source coordinates
            x, y, z = display_surface.parts[hemi].coordinates.T
            customdata = pd.DataFrame(dict(  # Store true coordinates for hover before messing with them
                x=x,
                y=y,
                z=z,
            ))
            coord_map = self.get_coord_map(x, y, z, hemi, turn_out_hemis=turn_out_hemis)
            x, y, z = coord_map(x, y, z)
            display_surface.parts[hemi].coordinates = np.column_stack((x, y, z))

            # Statmap colors
            plot_colors = iter(PLOT_COLORS)
            vertexcolor = None
            vertexalpha = None
            vertexscale = None
            colorbars = []

            for statmap_in, color, label, vmin_, vmax_, threshold_, statmap_scales_alpha_, skip_ in \
                    zip(statmaps_in, colors, statmap_labels, vmin, vmax, thresholds, statmap_scales_alpha, skip):
                if progress_fn is not None:
                    progress_fn(f'Computing {hemi} vertex colors for statmap {label}', 0)
                if isinstance(statmap_in, dict):
                    if 'functionals' in statmap_in:
                        assert 'seed' in statmap_in, \
                            'If statmap is connectivity, it must contain a key "seed" with the seed coordinates ' \
                            '(x, y, z).'
                        functional_paths = tuple(sorted(statmap_in['functionals']))
                        assert len(functional_paths) > 0, \
                            'If statmap is connectivity, it must contain a key "functionals" with a list of ' \
                            'at least one functional path.'
                        mask = statmap_in.get('mask', functional_paths[0])
                        seed = statmap_in['seed']
                        statmap_kwargs = dict(
                            functional_paths=functional_paths,
                            mask=mask,
                            seed=seed,
                            fwhm=statmap_in.get('fwhm', None)
                        )
                    elif 'path' in statmap_in:
                        statmap_kwargs = dict(path=statmap_in['path'], mask=statmap_in.get('mask', None))
                    else:
                        raise ValueError('`statmaps` dict must contain either "functionals" or "path" key.')
                elif isinstance(statmap_in, str):
                    statmap_kwargs = dict(path=statmap_in)
                else:
                    raise ValueError('`statmaps` must be a list of strings or dicts, not %s.' % type(statmap_in))
                if surf_types['pial'] is None:
                    statmap_kwargs['pial'] = None
                else:
                    statmap_kwargs['pial'] = surf_types['pial'][hemi]
                if surf_types['white'] is None:
                    statmap_kwargs['white'] = None
                else:
                    statmap_kwargs['white'] = surf_types['white'][hemi]
                statmap_kwargs['hemi'] = hemi

                if progress_fn is not None:
                    progress_fn(f'Projecting statmap {label} to {hemi} surface', 0)

                if color is None:
                    try:
                        color = next(plot_colors)
                    except StopIteration:
                        raise ValueError('Not enough colors in `climbprep.constants.PLOT_COLORS` to plot all statmaps. '
                                         'Please provide a list of colors for `cmaps`.')

                if skip_:
                    continue

                statmap, vertexcolor_, vertexalpha_, cmap, cmin, cmax = self.get_statmap_surface_and_color(
                    color,
                    vmin_=vmin_,
                    vmax_=vmax_,
                    threshold_=threshold_,
                    statmap_scales_alpha_=statmap_scales_alpha_,
                    hide_min=hide_min,
                    hide_max=hide_max,
                    progress_fn=progress_fn,
                    **statmap_kwargs
                )

                col = label if label else 'val'
                customdata[col] = statmap

                if vertexcolor is None:
                    vertexcolor = vertexcolor_
                else:
                    vertexcolor = np.where(np.isnan(vertexcolor), vertexcolor_,
                                           np.where(np.isnan(vertexcolor_), vertexcolor, vertexcolor + vertexcolor_))
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
                    colorbars.append(dict(
                        cmap=cmap,
                        cmin=cmin,
                        cmax=cmax,
                        label=label,
                        statmap_scales_alpha=statmap_scales_alpha_,
                        threshold=threshold_
                    ))

            if progress_fn is not None:
                progress_fn(f'Merging {hemi} vertex colors', 0)
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
                    vertexalpha *= 1 - bg_alpha
                    vertexalpha = np.where(np.isnan(vertexalpha), 0, vertexalpha)
                    vertexcolor = np.where(np.isnan(vertexcolor), 0, vertexcolor)
                    vertexcolor = bgcolors * (1 - vertexalpha) + vertexcolor * vertexalpha

            out[hemi] = dict(
                mesh=display_surface.parts[hemi],
                # vertexcolor=tuple([to_hex(c) for c in vertexcolor]),
                vertexcolor=vertexcolor,
                customdata=customdata
            )

        out['colorbars'] = colorbars
        out['seeds'] = seeds
        out['turn_out_hemis'] = turn_out_hemis

        return out

    def plot_data_to_fig(
            self,
            left,
            right,
            colorbars=None,
            seeds=None,
            turn_out_hemis=False,
            cbar_step=0.1
    ):
        traces = []
        for hemi in ('left', 'right'):
            data = left if hemi == 'left' else right
            surface_ = data['mesh']
            trace = self.make_plot_Mesh3d(surface=surface_)
            trace.vertexcolor = data['vertexcolor']
            trace.customdata = data['customdata']
            trace.hovertemplate = ''.join(['<b>' + col + ':</b> %{customdata[' + str(i) + ']:.2f}<br>'
                                           for i, col in enumerate(data['customdata'].columns)]) + '<extra></extra>'
            traces.append(trace)

        if not colorbars:
            colorbars = []
        cbar_x = 1
        for colorbar in colorbars:
            colorbar['cbar_x'] = cbar_x
            traces.append(self.make_colorbar(**colorbar))
            cbar_x += cbar_step

        if not seeds:
            seeds = []
        for seed in seeds:
            x, y, z = seed
            if turn_out_hemis:
                if x > 0:
                    x = -x
                    y = -y
                    trace = traces[1]
                else:
                    trace = traces[0]
                y_delta = trace.y[0] - data['customdata'].y.values[0]
                y += y_delta
            traces.append(self.make_sphere((x, y, z)))

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

        fig.add_traces(traces)

        return fig

    def fig_to_image(
            self,
            fig,
            plot_path,
            scale=1,
            htrim=0.,
            vtrim=0.,
            medial_zoom=1.15,
            cbar_step=0.2,
            turn_out_hemis=False
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
            camera = fig.layout.scene.camera
            out_path_base, ext = os.path.splitext(os.path.basename(plot_path))

            hemis = dict(left=fig.data[0], right= fig.data[1])
            colorbars = [x for x in fig.data[2:] if len(x.i) == 1]  # Colorbars have only one dummy point
            seeds = [x for x in fig.data[2:] if len(x.i) > 1]
            has_colorbar = len(colorbars) > 0

            imgs = [None] * 4
            ix = 0
            colorbar_written = False

            cbar_x = 1
            for cbar in colorbars:
                cbar.colorbar.x = cbar_x
                cbar_x += cbar_step

            cbar_img = None
            for hemi in ('left', 'right'):
                for view in ('lateral', 'medial'):
                    fig.data = []
                    fig.add_traces([hemis[hemi]] + seeds)
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

                    if has_colorbar and not colorbar_written:
                        cbar_path = os.path.join(tmp_dir, out_path_base + f'_cbar{ext}')
                        cbar_fig = go.Figure(fig)
                        cbar_fig.data = cbar_fig.data[:0]
                        cbar_fig.add_traces(colorbars)
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

    def get_surface_meshes(
            self,
            pial_left=None,
            pial_right=None,
            white_left=None,
            white_right=None,
            midthickness_left='infer',
            midthickness_right='infer',
            sulc_left=None,
            sulc_right=None,
            progress_fn=None
    ):
        if progress_fn is not None:
            progress_fn('Loading pial meshes')
        if pial_left is not None or pial_right is not None:
            pial = self.get_surface_mesh(left=pial_left, right=pial_right)
        else:
            pial = None

        if progress_fn is not None:
            progress_fn('Loading white meshes')
        if white_left is not None or white_right is not None:
            white = self.get_surface_mesh(left=white_left, right=white_right)
        else:
            white = None

        if progress_fn is not None:
            progress_fn('Loading midthickness meshes')
        midthickness = self.get_midthickness_mesh(
            left=midthickness_left,
            right=midthickness_right,
            pial_left=pial_left,
            pial_right=pial_right,
            white_left=white_left,
            white_right=white_right
        )

        if progress_fn is not None:
            progress_fn('Loading sulcal depth', 0)
        if sulc_left is not None or sulc_right is not None:
            sulc = self.get_surface_data(left=sulc_left, right=sulc_right)
        else:
            sulc = None

        return pial, white, midthickness, sulc

    def get_surface_mesh(
            self,
            left,
            right
    ):
        return surface.PolyMesh(
            left=self.get_surface_mesh_hemi(left),
            right=self.get_surface_mesh_hemi(right)
        )

    # Cachable
    def get_surface_mesh_hemi(
            self,
            path,
    ):
        if path is None:
            return path
        return surface.PolyMesh(left=path).parts['left']

    def get_midthickness_mesh(
            self,
            left=None,
            right=None,
            pial_left=None,
            pial_right=None,
            white_left=None,
            white_right=None
    ):
        if left is None and right is None:
            return None

        assert (pial_left is not None or pial_right is not None) and \
            (white_left is not None or white_right is not None), \
            'If no left or right mesh is provided, both pial and white meshes must be provided.'

        midthickness = dict(
            left=left,
            right=right
        )
        pial = dict(
            left=pial_left,
            right=pial_right
        )
        white = dict(
            left=white_left,
            right=white_right
        )

        mesh_kwargs = {}
        for hemi in ('left', 'right'):
            if midthickness[hemi] is None:
                mesh_kwargs[hemi] = None
            elif midthickness[hemi] == 'infer':
                # Average pial and white coords
                assert pial[hemi] is not None, \
                    f'Pial mesh must be provided for {hemi} hemisphere if midthickness is `infer`.'
                assert white[hemi] is not None, \
                    f'White mesh must be provided for {hemi} hemisphere if midthickness is `infer`.'
                mesh_kwargs[hemi] = self.infer_midthickness_mesh_hemi(pial[hemi], white[hemi])
            else:
                mesh_kwargs[hemi] = self.get_surface_mesh_hemi(midthickness[hemi])
        return surface.PolyMesh(**mesh_kwargs)

    # Cachable
    def infer_midthickness_mesh_hemi(
            self,
            pial,
            white
    ):
        pial = self.get_surface_mesh_hemi(pial)
        white = self.get_surface_mesh_hemi(white)
        midthickness_coords = ((pial.coordinates + white.coordinates) / 2) \
            .astype(np.float32)
        # Use pial faces (arbitrary, faces are the same for all meshes)
        midthickness_faces = pial.faces

        return surface.InMemoryMesh(
            coordinates=midthickness_coords,
            faces=midthickness_faces
        )

    # Cachable
    def get_surface_data(
            self,
            left,
            right
    ):
        return surface.PolyData(left=left, right=right)

    # Cachable
    def get_plot_bgcolors(self, nvertices, hemi, sulc_left=None, sulc_right=None, bg_brightness=0.5, sulc_alpha=0.9):
        bgcolors = np.ones((nvertices, 3)) * bg_brightness
        sulc = sulc_left if hemi == 'left' else sulc_right
        if sulc is not None:
            sulc_data = self.get_surface_data(left=sulc_left, right=sulc_right).parts[hemi]
            sulc_vmin, sulc_vmax = np.min(sulc_data), np.max(sulc_data)
            if sulc_vmin < 0 or sulc_vmax > 1:
                bg_norm = Normalize(vmin=sulc_vmin, vmax=sulc_vmax)
                sulc_data = bg_norm(sulc_data)
            sulccolors = plt.get_cmap("Greys")(sulc_data)[..., :3]
            bgcolors = bgcolors * (1 - sulc_alpha) + sulccolors * sulc_alpha

        bgcolors = bgcolors.astype(np.float32)

        return bgcolors

    def get_statmap(self, path=None, functional_paths=None, mask=None, seed=None, fwhm=None, progress_fn=None):
        if path is None and functional_paths is None:
            raise ValueError('Either `path` or `functional_paths` must be provided.')
        if path is not None:
            if progress_fn is not None:
                progress_fn(f'Loading statmap {path}')
            statmap = self.load_nii(path)
        else:
            assert functional_paths is not None, 'If `path` is not provided, `functional_paths` must be ' \
                                     'a tuple of paths to timecourses.'
            assert seed and len(seed) == 3, 'If `path` is not provided, `seed` must be a tuple of (x, y, z) ' \
                                            'coordinates.'
            x, y, z = seed
            statmap = self.get_connectivity_from_seed(
                x=x, y=y, z=z,
                functional_paths=functional_paths,
                mask_path=mask,
                fwhm=fwhm,
                progress_fn=progress_fn
            )

        return statmap

    def get_functional_paths(
            self,
            participant,
            project='climblab',
            session=None,
            cleaning_label=CLEAN_DEFAULT_KEY,
            space=PARCELLATE_DEFAULT_KEY,
            regex_filter=None
    ):
        if regex_filter is not None:
            regex_filter = re.compile(regex_filter)
        if session:
            sessions = {session}
        else:
            sessions = set(
                [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project, f'sub-{participant}'))
                 if x.startswith('ses-')]
            )
            if not sessions:
                sessions = {None}
        functional_paths = []
        for session in sessions:
            clean_dir = os.path.join(
                BIDS_PATH, project, 'derivatives', 'clean', cleaning_label, f'sub-{participant}'
            )
            if session:
                clean_dir = os.path.join(clean_dir, f'ses-{session}')
            if os.path.exists(clean_dir):
                for x in os.listdir(clean_dir):
                    path_ = os.path.join(clean_dir, x)
                    if not x.endswith(f'_space-{space}_desc-clean_bold.nii.gz'):
                        continue
                    if regex_filter is not None and not regex_filter.search(x):
                        continue
                    functional_paths.append(path_)

        return tuple(sorted(functional_paths))

    # Cachable
    def get_functionals_and_mask(
            self,
            functional_paths,
            mask_path,
            debug=False,
            progress_fn=None
    ):
        if not functional_paths:
            return [], None

        if debug:
            functional_paths = functional_paths[:1]

        if progress_fn:
            progress_fn('Getting mask', 0)
        mask = self.get_mask(mask_path, nii_ref=functional_paths[0])
        functionals_ = []
        incr = 0
        if progress_fn:
            incr = progress_fn.incr
            progress_fn.incr = incr / (len(functional_paths) + 1)
            progress_fn(f'Loading {len(functional_paths)} timecourses', 0)
        for i, functional in enumerate(functional_paths):
            if len(functional) > 75:
                functional_str = '...' + functional[-75:]
            else:
                functional_str = functional
            msg = f'Loading and resampling functional image {i+1}/{len(functional_paths)}: {functional_str}'
            print('  ' + msg)
            if progress_fn:
                progress_fn(msg)
            functionals_.append(self.get_functional(functional, mask_path))
        if progress_fn:
            progress_fn(f'Concatenating {len(functional_paths)} timecourses', 0)
        t0 = time.time()
        for functional_ in functionals_:
            std = np.std(functional_, axis=0)
        functionals_ = np.concatenate(functionals_, axis=0)  # shape (n_timepoints, n_kept_voxels)
        t1 = time.time()
        print(f'   Concatenating runs into shape {functionals_.shape} took {t1 - t0:.2f} seconds.' % ())
        if progress_fn:
            progress_fn.incr = incr

        return functionals_, mask

    # Cachable
    def get_mask(self, mask_path, target_affine=DEFAULT_TARGET_AFFINE, mask_fwhm=DEFAULT_MASK_FWHM, nii_ref=None):
        if mask_path is None:
            assert nii_ref is not None, \
                'If `mask_path` is None, `nii_ref` must be provided to compute the mask.'
            mask = masking.compute_brain_mask(nii_ref, connected=False, opening=False, mask_type='gm')
        else:
            mask = self.load_nii(mask_path)
        mask = image.new_img_like(mask, image.get_data(mask).astype(np.float32))
        if mask_fwhm:
            mask = image.smooth_img(mask, fwhm=mask_fwhm)
        if target_affine is not None:
            mask = image.resample_img(mask, target_affine=target_affine, interpolation='linear')
        mask = image.math_img('x > 0', x=mask)

        return mask

    # Cachable
    def get_functional(self, path, mask_path, target_affine=DEFAULT_TARGET_AFFINE):
        t0 = time.time()
        functional = self.load_nii(path)
        mask = self.get_mask(mask_path, target_affine=target_affine, nii_ref=path)
        t1 = time.time()
        print(f'    Loading data {path} took {t1 - t0:.2f} seconds.')
        t0 = time.time()
        functional = image.resample_to_img(functional, mask)
        functional = self.apply_mask(functional, mask)
        t1 = time.time()
        print(f'    Resampling functional image {path} took {t1 - t0:.2f} seconds.')

        return functional

    def load_nii(self, path, fwhm=None):
        if fwhm:
            return image.smooth_img(path, fwhm=fwhm)
        return image.load_img(path)

    # Cachable
    def get_connectivity_from_seed(self, x, y, z, functional_paths, mask_path, fwhm=None, progress_fn=None):
        incr = load_incr = conn_incr = 0
        if progress_fn is not None:
            incr = progress_fn.incr
            load_incr = incr * 0.9
            conn_incr = incr - load_incr
            progress_fn.incr = load_incr
        functionals, mask_img = self.get_functionals_and_mask(
            functional_paths, mask_path=mask_path, progress_fn=progress_fn
        )
        if progress_fn is not None:
            progress_fn.incr = conn_incr
            progress_fn(f'Calculating connectivity from seed ({x}, {y}, {z})')
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
        seed_timecourse = self.standardize_timecourse(seed_timecourse)
        other_timecourses = self.standardize_timecourse(functionals)
        n = len(seed_timecourse)

        connectivity = (other_timecourses.T @ seed_timecourse / n).astype(np.float32)
        connectivity = self.invert_mask(connectivity, mask_img)

        if progress_fn is not None:
            progress_fn.incr = incr

        return connectivity


    # Cachable
    def get_statmap_surface(
            self,
            path=None,
            functional_paths=None,
            mask=None,
            seed=None,
            fwhm=None,
            pial=None,
            white=None,
            hemi='left',
            progress_fn=None
    ):
        assert hemi in ('left', 'right'), 'Hemispheres must be "left" or "right".'
        statmap_nii = self.get_statmap(
            path=path, functional_paths=functional_paths, mask=mask, seed=seed, fwhm=fwhm, progress_fn=progress_fn
        )
        mesh_kwargs = {
            f'pial_{hemi}': pial,
        }
        if white is not None:
            mesh_kwargs[f'white_{hemi}'] = white
        mesh_kwargs[f'midthickness_left'] = None
        mesh_kwargs[f'midthickness_right'] = None
        pial, white, _, _ = self.get_surface_meshes(**mesh_kwargs)

        if progress_fn is not None:
            progress_fn(f'Projecting statmap to {hemi} surface', 0)
        if mask:
            nii_ref = None
        elif path:
            nii_ref = path
        else:
            assert functional_paths is not None and len(functional_paths) > 0, \
                'If `mask` is not provided, `path` or `functional_paths` must be provided to get the reference NIfTI.'
            assert len(functional_paths), \
                'If `mask` is not provided, `functional_paths` must be a list of at least one functional path.'
            nii_ref = functional_paths[0]
        statmap = surface.vol_to_surf(
            statmap_nii,
            pial.parts[hemi],
            mask_img=self.get_mask(mask, nii_ref=nii_ref),
            inner_mesh=white.parts[hemi],
            depth=np.linspace(0.0, 1.0, 10)
        ).astype(np.float32)

        return statmap

    # Cachable
    def get_statmap_surface_and_color(
            self,
            color,
            vmin_=None,
            vmax_=None,
            threshold_=None,
            statmap_scales_alpha_=True,
            hide_min=None,
            hide_max=None,
            progress_fn=None,
            **statmap_kwargs
    ):
        statmap_src = self.get_statmap_surface(**statmap_kwargs, progress_fn=progress_fn)
        statmap = statmap_src.copy()
        if vmin_ is None:
            vmin_ = np.nanmin(statmap)
        if vmax_ is None:
            vmax_ = np.nanmax(statmap)
        cmin = vmin_
        cmax = vmax_
        if cmin < 0 < cmax:  # If both positive and negative values are present, center around 0
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

        assert isinstance(color, str), '`cmap` must be a string.'
        try:
            color = to_rgb(color)
            cmap = LinearSegmentedColormap.from_list(f'{color}', [color, color])
        except ValueError:
            cmap = plt.get_cmap(color)
        cmap.set_extremes(bad=[np.nan] * 4, over=[np.nan] * 4, under=[np.nan] * 4)

        vertexcolor_ = cmap(statmap)[..., :3]
        if statmap_scales_alpha_:
            vertexcolor_ = vertexcolor_ * statmap_abs[..., None]
        if statmap_scales_alpha_:
            vertexalpha_ = statmap_abs[..., None]
        else:
            vertexalpha_ = np.isfinite(statmap_abs)[..., None].astype(statmap_abs.dtype)

        return statmap_src, vertexcolor_, vertexalpha_, cmap, cmin, cmax

    def make_plot_Mesh3d(self, surface):
        x, y, z = surface.coordinates.T
        i, j, k = surface.faces.T

        trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
        trace.lighting = PLOT_LIGHTING
        trace.lightposition = PLOT_LIGHTPOSITION

        return trace

    def make_colorbar(
            self,
            cmap,
            cmin,
            cmax,
            label,
            statmap_scales_alpha,
            threshold,
            cbar_x=1.0
    ):
        cbar = go.Mesh3d()
        cbar.cmax = cmax
        cbar.cmin = cmin
        cbar.colorbar = dict(len=0.5, tickfont=dict(color='black', size=25), tickformat='.1f', x=cbar_x)
        if label:
            cbar.colorbar.title = dict(text=label, side='right')
        colorscale_ix = np.linspace(cmin, cmax, 256) / (cmax - cmin)
        if statmap_scales_alpha:
            colorscale_alpha = np.abs(colorscale_ix)
        else:
            colorscale_alpha = np.ones_like(colorscale_ix)
        if threshold is not None:
            colorscale_alpha = np.where(colorscale_alpha < threshold / (cmax - cmin), 0., colorscale_alpha)
        colorscale_ix -= cmin / (cmax - cmin)
        colorscale = cmap(colorscale_ix)
        colorscale[..., 3] = colorscale_alpha
        colorscale_ = []
        for colorscale_ix_, color in zip(colorscale_ix, colorscale):
            if np.all(np.isfinite(color)):
                color = 'rgba(' + ', '.join([f'{round(c * 255)}' for c in color[:3]] + [f'{color[3]}']) + ')'
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

        return cbar

    # Cachable
    def make_sphere(self, center, radius=1, opacity=0.7):
        sphere_pv = pyvista.Sphere(radius=radius, center=center)
        x, y, z = sphere_pv.points.T.astype(np.float32)
        i, j, k = sphere_pv.faces.reshape((-1, 4))[:, 1:].T
        sphere = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='white', opacity=opacity)

        return sphere

    def get_coord_map(self, x, y, z, hemi, spacer=5, turn_out_hemis=False):
        if turn_out_hemis:
            if hemi == 'left':
                x_a = 1
                x_b = 0
                y_a = 1
                y_b = -y.min() + spacer
            else:
                x_a = -1
                x_b = 0
                y_a = -1
                y_b = -y.max() - spacer
            def coord_map (x, y, z):
                return x * x_a + x_b, y * y_a + y_b, z
        else:
            def coord_map(x, y, z):
                return x, y, z

        return coord_map

    def map_coords(self, x, y, z, hemi, spacer=5, turn_out_hemis=False):
        if turn_out_hemis:
            if hemi == 'right':
                x = -x
                y = -y
            x -= x.min()
            if hemi == 'left':
                y -= y.min() - spacer
            else:
                y -= y.max() + spacer

        return x, y, z

    def standardize_timecourse(self, arr, axis=0):
        return (arr - np.mean(arr, axis=axis, keepdims=True)) / np.std(arr, axis=axis, keepdims=True)

    def apply_mask(self, arr, mask_nii):
        # arr and mask_nii are NIFTI images
        arr = image.get_data(arr)
        mask = image.get_data(mask_nii).astype(bool)
        # arr.shape -> (n_timepoints, n_kept_voxels) OR (n_kept_voxels,)
        return arr[mask].T

    def invert_mask(self, arr, mask_nii):
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


class PlotLibMemoized(PlotLib):

    def __init__(
            self,
            cache_fns=None
    ):
        if cache_fns is None:
            self.cache_fn_default = lambda x: x
            self.cache_fns = {}
        elif isinstance(cache_fns, dict):
            self.cache_fn_default = lambda x: x
            self.cache_fns = cache_fns
        else:
            self.cache_fn_default = cache_fns
            self.cache_fns = {}

        for cachable in self.CACHABLE:
            cache_fn = self.cache_fns.get(cachable, self.cache_fn_default)
            setattr(self, cachable, cache_fn(getattr(self, cachable)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot firstlevel contrasts for a participant')
    argparser.add_argument('participant', help='BIDS participant ID')
    argparser.add_argument('-p', '--project', default='climblab', help=('Name of BIDS project (e.g., "climblab", '
                                                                        '"evlab", etc.). Default: "climblab"'))
    argparser.add_argument('-m', '--models', default=[], nargs='*', help=('List of models to plot. '
                                                                          '(see `climbprep.constants.MODELFILES_PATH` '
                                                                          'If not specified, will plot all models '
                                                                          'available for the participant.'))
    argparser.add_argument('-c', '--config', default=PLOT_DEFAULT_KEY, help=('Keyword (currently `main`) '
        'or YAML config file to used to parameterize plotting. If a keyword is provided, will '
        'the default settings for associated with that keyword. '
        'See `climbprep.constants.CONFIG["plot"]` for available config names and their settings.'))
    argparser.add_argument('--ncpus', type=int, default=8, help='Number of parallel processes to use.')
    args = argparser.parse_args()

    cache = diskcache.Cache(
        os.path.join(os.getcwd(), '.cache', 'viz'),
        size_limit=1024 * 1024 * 1024 * 16,  # 16GB
    )

    pl = PlotLibMemoized(
        cache.memoize(ignore={'progress_fn', 'masker'})
    )

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
    models_path = os.path.join(derivatives_path, 'model', model_label)
    assert os.path.exists(models_path), 'Path not found: %s' % models_path

    stderr(f'Plotting outputs will be written to {os.path.join(derivatives_path, "plot", plot_label)}\n')

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
        bids_path = os.path.join(derivatives_path, 'preprocess', preprocessing_label)
        assert os.path.exists(bids_path), 'Path not found: %s' % bids_path
        anat_path = os.path.join(bids_path, f'sub-{participant}', 'anat')
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
                    'plot',
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
                    stderr(f'Plotting statmap {os.path.join(contrast_path, statmap_path)}\n')
                    fig = pl.get_fig(
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
                        display_surface='midthickness',
                        colorbar=True
                    )
                    plot_path_ = os.path.join(
                        plot_path,
                        os.path.basename(statmap_path)[:-len(PLOT_STATMAP_SUFFIX)] + '.png'
                    )
                    pl.fig_to_image(
                        fig,
                        plot_path=plot_path_,
                        scale=config['scale'],
                        htrim=config['htrim'],
                        vtrim=config['vtrim']
                    )
                    stderr(f'  Finished plotting: {plot_path_}\n')
