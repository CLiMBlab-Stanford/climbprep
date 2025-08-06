from nilearn import datasets
from plotly import graph_objects as go
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import Dash, DiskcacheManager, html, dcc, Input, Output, State, Patch, callback_context
import diskcache
import argparse
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from climbprep.plot import PlotLibMemoized
from climbprep.constants import *
from climbprep import resources


CACHE_PATH = os.path.join(os.getcwd(), '.cache', 'viz')
CACHE_SIZE = 1024 * 1024 * 1024 * 16  # 16GB
OPTION_HEIGHT = 50


class Progress:
    def __init__(self, progress_fn):
        self.progress_fn = progress_fn
        self.value = 0
        self.incr = 0

    def __call__(self, message, incr=None):
        if incr is None:
            incr = self.incr
        self.progress_fn((message, self.value * 100))
        self.value += incr


def viewport():
    fig = go.Figure(go.Mesh3d())
    fig.layout.paper_bgcolor = 'white'
    fig.layout.scene = {
        **{f'{dim}axis': PLOT_AXIS_CONFIG for dim in ('x', 'y')},
    }

    graph = dcc.Graph(
        id='main',
        responsive=True,
        style=dict(visibility='hidden')
    )

    div = html.Div(
        id='viewport',
        children=[
            graph
        ]
    )

    # loading = dcc.Loading(
    #     id='viewport-loader',
    #     children=[div],
    #     target_components={'main': '*'}
    # )

    return div


def menu():
    menu = html.Div([
        dmc.Affix(
            dmc.Button(
                DashIconify(icon='material-symbols:menu-rounded', color='#fff',
                            style={'width': '3rem', 'height': '3rem'}),
                id='drawer-toggle',
                style=dict(
                    height='4rem',
                    width='4rem'
                )
            ),
            position=dict(top='1rem', left='1rem'),
            zIndex=2000
        ),
        dmc.Affix(
            dmc.Button(
                children=[
                    DashIconify(icon='mdi:brain', color='#fff',
                            style={'width': '3rem', 'height': '3rem'}),
                    html.Span(className='compiling')
                ],
                id='compile',
                style=dict(
                    height='4rem',
                    width='4rem'
                )
            ),
            position=dict(top='1rem', left='6rem'),
            zIndex=2000
        ),
        dmc.Affix(
            dmc.Button(
                children=[
                    DashIconify(icon='material-symbols:stop', color='#fff',
                            style={'width': '3rem', 'height': '3rem'})
                ],
                id='cancel',
                style=dict(
                    height='4rem',
                    width='4rem',
                    display='none',
                    background='red'
                )
            ),
            position=dict(top='1rem', left='11rem'),
            zIndex=2000
        ),
        dmc.Drawer(
            [
                html.Div(
                    children=[
                        html.Label('Project'),
                        dcc.Dropdown(
                            [],
                            None,
                            id='project-dropdown',
                            clearable=False
                        )
                    ]
                ),
                html.Div(
                    children=[
                        html.Label('Participant'),
                        dcc.Dropdown(
                            [],
                            None,
                            id='participant-dropdown',
                            clearable=False,
                        )
                    ],
                    id='participant-dropdown-wrapper'
                ),
                html.Div(
                    children=[
                        html.Label('Display surface'),
                        dcc.Dropdown(
                            [
                                {'label': 'Pial', 'value': 'pial'},
                                {'label': 'White', 'value': 'white'},
                                {'label': 'Midthickness', 'value': 'midthickness'},
                                {'label': 'Inflated', 'value': 'inflated'}
                            ],
                            'midthickness',
                            id='display-surface-dropdown',
                            clearable=False,
                        )
                    ],
                    id='display-surface-dropdown-wrapper'
                ),
                dmc.Switch(id='template-surface',
                           label="Use MNI surface", checked=False),
                html.Div(
                    children=[
                        dmc.TextInput(
                            id='local-directory',
                            placeholder=f'Path to local directory',
                            style={'width': '100%', 'paddingTop': '0.25rem'},
                            label='Local directory',
                            required=True
                        ),
                        dmc.Flex(
                            [
                                html.Div(
                                    [
                                        html.Label('Pial left'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='pial-left',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                                html.Div(
                                    [
                                        html.Label('Pial right'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='pial-right',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                            ]
                        ),
                        dmc.Flex(
                            [
                                html.Div(
                                    [
                                        html.Label('White left'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='white-left',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                                html.Div(
                                    [
                                        html.Label('White right'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='white-right',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                            ]
                        ),
                        dmc.Flex(
                            [
                                html.Div(
                                    [
                                        html.Label('Midthickness left'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='midthickness-left',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                                html.Div(
                                    [
                                        html.Label('Midthickness right'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='midthickness-right',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                            ]
                        ),
                        dmc.Flex(
                            [
                                html.Div(
                                    [
                                        html.Label('Inflated left'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='inflated-left',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                                html.Div(
                                    [
                                        html.Label('Inflated right'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='inflated-right',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                            ]
                        ),
                        dmc.Flex(
                            [
                                html.Div(
                                    [
                                        html.Label('Sulc left'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='sulc-left',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                                html.Div(
                                    [
                                        html.Label('Sulc right'),
                                        dcc.Dropdown(
                                            [],
                                            'None',
                                            id='sulc-right',
                                            optionHeight=OPTION_HEIGHT,
                                            clearable=True
                                        )
                                    ],
                                    style={'width': '49%'}
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label('Mask'),
                                dmc.TextInput(
                                    id='mask',
                                    placeholder='Path to mask file (optional)',
                                )
                            ],
                            style={'width': '98%'}
                        ),
                    ],
                    id='local-directory-wrapper',
                    style=dict(display='none')
                ),
                dmc.TextInput(
                    id='preprocessing-label',
                    placeholder=f'Preprocessing label (default: {PREPROCESS_DEFAULT_KEY})',
                    style={'width': '100%', 'marginTop': '0.25rem'}
                ),
                dmc.Switch(
                    id='additive-color',
                    label='Additive color',
                    checked=True,
                    style={'width': '48%', 'marginTop': '0.25rem'}
                ),
                dmc.Switch(
                    id='turn-out-hemis',
                    label='Turn out hemispheres',
                    checked=False,
                    style={'width': '48%', 'marginTop': '0.25rem'}
                ),
                html.Div(
                    children=[
                        html.H3('Overlays'),
                        dmc.Group(
                            children=[
                                dmc.Button(
                                    '+',
                                    id='add-statmap'
                                ),
                                dcc.Dropdown(
                                    [],
                                    id='statmap-type',
                                    value=None,
                                    clearable=False,
                                    style=dict(width='80%')
                                )
                            ]
                        ),
                        html.Div(
                            children=[],
                            id='statmap-list'
                        )
                    ],
                    id='statmap-wrapper'
                )
            ],
            id='menu',
            opened=True
        )
    ])

    return menu


def layout():
    return html.Div(
        children=[
            dcc.Store(id='store', storage_type='memory'),
            html.Div(
                [
                    dmc.Progress(
                        id='progress',
                        value=0,
                        size='lg',
                        radius='5px',
                        striped=False,
                        animated=False,
                        style={
                            'width': '20%',
                            'display': 'block',
                            'border': '1.5px solid #777',
                        }
                    ),
                    html.Span(
                        'Compiling figure',
                        id='progress-text',
                        style={'display': 'block'}
                    )
                ],
                id='progress-wrapper',
                style={'display': 'none'}
            ),
            menu(),
            viewport()
        ]
    )


def assign_callbacks(app, cache):
    pl = PlotLibMemoized(
        cache.memoize(ignore={'progress_fn', 'masker'})
    )

    @app.callback(
        Output('main', 'figure'),
        Output('main', 'style'),
        Input('compile', 'n_clicks'),
        State('project-dropdown', 'value'),
        State('participant-dropdown', 'value'),
        State('display-surface-dropdown', 'value'),
        State('preprocessing-label', 'value'),
        State('additive-color', 'checked'),
        State('turn-out-hemis', 'checked'),
        State('local-directory', 'value'),
        State('mask', 'value'),
        State('template-surface', 'checked'),
        State('pial-left', 'value'),
        State('pial-right', 'value'),
        State('white-left', 'value'),
        State('white-right', 'value'),
        State('midthickness-left', 'value'),
        State('midthickness-right', 'value'),
        State('inflated-left', 'value'),
        State('inflated-right', 'value'),
        State('sulc-left', 'value'),
        State('sulc-right', 'value'),
        State('statmap-list', 'children'),
        State('main', 'figure'),
        prevent_initial_call=True,
        background=True,
        running=[
            (Output('compile', 'disabled'), True, False),
            (
                    Output('cancel', 'style'),
                    dict(
                        height='4rem',
                        width='4rem',
                        display='block',
                        background='red'
                    ),
                    dict(
                        height='4rem',
                        width='4rem',
                        display='none',
                        background='red'
                    )
            ),
            (
                Output("progress-wrapper", "style"),
                {
                    'display': 'block',
                    'width': '100%',
                    'textAlign': 'left',
                    'position': 'fixed',
                    'bottom': 0,
                    'left': 0,
                    'padding': '1rem',
                    'zIndex': 2000,
                    'backgroundColor': '#f0f0f0',
                    'opacity': 0.9,
                },
                {
                    'display': 'none'
                }
            )
        ],
        progress=[
            Output("progress-text", "children"),
            Output("progress", "value")
        ],
        cancel=[Input("cancel", "n_clicks")]
    )
    def update_graph(
            progress_fn,
            n_clicks,
            project,
            participant,
            display_surface,
            preprocessing_label,
            additive_color,
            turn_out_hemis,
            local_directory,
            local_mask,
            use_template_surface,
            pial_left,
            pial_right,
            white_left,
            white_right,
            midthickness_left,
            midthickness_right,
            inflated_left,
            inflated_right,
            sulc_left,
            sulc_right,
            statmap_list,
            fig_prev
    ):
        if progress_fn is not None:
            progress_fn = Progress(progress_fn)
            progress_fn('Initializing...', 0)

        if use_template_surface:
            pial_left = pkg_resources.files(resources).joinpath('mni_lh.pial.gii')
            pial_right = pkg_resources.files(resources).joinpath('mni_rh.pial.gii')
            white_left = pkg_resources.files(resources).joinpath('mni_lh.white.gii')
            white_right = pkg_resources.files(resources).joinpath('mni_rh.white.gii')
            inflated_left = pkg_resources.files(resources).joinpath('mni_lh.inflated.gii')
            inflated_right = pkg_resources.files(resources).joinpath('mni_rh.inflated.gii')
            sulc_left = pkg_resources.files(resources).joinpath('mni_lh.sulc.gii')
            sulc_right = pkg_resources.files(resources).joinpath('mni_rh.sulc.gii')
            surfaces = dict(
                pial=dict(
                    left=pial_left,
                    right=pial_right
                ),
                white=dict(
                    left=white_left,
                    right=white_right
                ),
                midthickness=dict(left='infer', right='infer'),
                inflated=dict(
                    left=inflated_left,
                    right=inflated_right
                ),
                sulc=dict(
                    left=sulc_left,
                    right=sulc_right
                )
            )
        else:
            surfaces = dict(
                pial=dict(left=pial_left, right=pial_right),
                white=dict(left=white_left, right=white_right),
                midthickness=dict(
                    left='infer' if midthickness_left is None else midthickness_left,
                    right='infer' if midthickness_right is None else midthickness_right
                ),
                inflated=dict(left=inflated_left, right=inflated_right),
                sulc=dict(left=sulc_left, right=sulc_right)
            )

        if project == 'local':
            anat = {'mask': None}
            for surf_type in ('pial', 'white', 'midthickness', 'inflated', 'sulc'):
                for hemi in ('left', 'right'):
                    surf_path = surfaces[surf_type][hemi]
                    if local_directory is not None and (surf_path is not None and surf_path != 'infer'):
                        surf_path = os.path.join(local_directory, surf_path)
                        assert os.path.exists(surf_path), f'Surface {surf_type} for {hemi} hemisphere does not ' \
                                                          f'exist: {surf_path}'
                    if surf_type not in anat:
                        anat[surf_type] = {}
                    anat[surf_type][hemi] = surf_path
            if local_mask and not use_template_surface:
                anat['mask'] = os.path.join(local_directory, local_mask)
        else:
            projects = sorted([x for x in os.listdir(BIDS_PATH) if os.path.isdir(os.path.join(BIDS_PATH, x))])
            if project is None or project not in projects:
                project = 'climblab'
            participants = sorted(
                [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
            )
            if participant is None or participant not in participants:
                participant = participants[0] if participants else None

            preprocessing_label = preprocessing_label or PREPROCESS_DEFAULT_KEY
            session_subdirs = [
                x for x in os.listdir(os.path.join(
                    BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}'
                )) if x.startswith('ses-')
            ]
            anat = {}
            mask_path = os.path.join(
                BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}', 'anat',
                f'sub-{participant}{DEFAULT_MASK_SUFFIX}'
            )
            if not os.path.exists(mask_path) and session_subdirs:
                session = session_subdirs[0][4:]
                mask_path = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}',
                    f'ses-{session}', 'anat', f'sub-{participant}_ses-{session}{DEFAULT_MASK_SUFFIX}'
                )
            anat['mask'] = None if use_template_surface else mask_path
            for surf_type in ('pial', 'white', 'midthickness', 'sulc'):
                if surf_type == 'sulc':
                    suffix = '.shape.gii'
                else:
                    suffix = '.surf.gii'
                for hemi in ('left', 'right'):
                    if use_template_surface:
                        surf_path = surfaces[surf_type][hemi]
                    else:
                        surf_path = os.path.join(
                            BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}',
                            'anat', f'sub-{participant}_hemi-{hemi[0].upper()}_{surf_type}{suffix}'
                        )
                        if not os.path.exists(surf_path) and session_subdirs:
                            session = session_subdirs[0][4:]
                            surf_path = os.path.join(
                                BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, f'sub-{participant}',
                                f'ses-{session}', 'anat',
                                f'sub-{participant}_ses-{session}_hemi-{hemi[0].upper()}_{surf_type}{suffix}'
                            )
                    if not surf_type in anat:
                        anat[surf_type] = dict()
                    anat[surf_type][hemi] = surf_path
            surf_type = 'inflated'
            for hemi in ('left', 'right'):
                if use_template_surface:
                    surf_path = surfaces[surf_type][hemi]
                else:
                    surf_path = os.path.join(
                        BIDS_PATH, project, 'derivatives', 'preprocess', preprocessing_label, 'sourcedata',
                        'freesurfer', f'sub-{participant}', 'surf', f'{hemi[0]}h.inflated'
                    )
                if not surf_type in anat:
                    anat[surf_type] = dict()
                anat[surf_type][hemi] = surf_path

        statmaps = statmap_list
        statmap_paths = []
        statmap_labels = []
        colors = []
        vmin = []
        vmax = []
        thresholds = []
        statmap_scales_alpha = []
        skip = []
        for statmap in statmaps:
            stat_type = get_value(statmap, 'type')
            session = get_value(statmap, 'session')
            if session == 'all':
                session = None

            subdir = f'sub-{participant}'
            session_str = ''
            if session:
                subdir = os.path.join(subdir, f'ses-{session}')
                session_str = f'_ses-{session}'
                node = 'session'
            else:
                node = 'subject'

            if stat_type == 'contrast':
                task = get_value(statmap, 'task') or None
                contrast = get_value(statmap, 'contrast') or None
                model_label = get_value(statmap, 'model_label') or MODEL_DEFAULT_KEY
                if task is None or contrast is None:
                    continue
                statmap_path = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'model', model_label, task, f'node-{node}', subdir,
                    f'sub-{participant}{session_str}_contrast-{contrast}_stat-t_statmap.nii.gz'
                )
                if not os.path.exists(statmap_path):
                    continue
                statmap_in = dict(
                    path=statmap_path,
                    mask=anat['mask'],
                )
                statmap_label_default = f'{contrast} (t)' if not session else f'{contrast} (t), {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 5
            elif stat_type == 'network':
                network = get_value(statmap, 'network') or None
                if network is None:
                    continue
                parcellation_label = get_value(statmap, 'parcellation_label') or PARCELLATE_DEFAULT_KEY
                node = 'session' if session else 'subject'
                statmap_dir = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'parcellate', parcellation_label, f'node-{node}', subdir,
                )
                statmap_path = None
                for x in os.listdir(statmap_dir):
                    if ('_hemi-L' in x and x.endswith(f'{network}.gii')) or x.endswith(f'{network}.nii.gz'):
                        statmap_path = os.path.join(statmap_dir, x)
                        break
                if not statmap_path or not os.path.exists(statmap_path):
                    continue
                statmap_in = dict(
                    path=statmap_path,
                    mask=anat['mask'],
                )
                statmap_label_default = f'p({network})' if not session else f'p({network}), {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 1
            elif stat_type == 'file':
                statmap_file = get_value(statmap, 'local_file') or None
                if not statmap_file:
                    continue
                statmap_path = os.path.join(local_directory, statmap_file)
                if not os.path.exists(statmap_path):
                    continue
                statmap_in = dict(
                    path=statmap_path,
                    mask=anat.get('mask', None)
                )
                statmap_label_default = f'{statmap_file.replace("nii.gz", "").replace("nii", "")}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin')
                vmax_ = get_value(statmap, 'vmax')
            elif stat_type == 'connectivity':
                seed = (
                    get_value(statmap, 'seed_x'),
                    get_value(statmap, 'seed_y'),
                    get_value(statmap, 'seed_z'),
                )
                if seed[0] is None or seed[1] is None or seed[2] is None:
                    continue
                fwhm = get_value(statmap, 'fwhm') or None
                regex_filter = get_value(statmap, 'regex_filter') or None
                cleaning_label = get_value(statmap, 'cleaning_label') or CLEAN_DEFAULT_KEY
                space = PARCELLATE_DEFAULT_KEY

                functional_paths = pl.get_functional_paths(
                    participant,
                    project=project,
                    session=session,
                    cleaning_label=cleaning_label,
                    space=space,
                    regex_filter=regex_filter
                )

                if not functional_paths:
                    continue

                statmap_in = dict(
                    functionals=functional_paths,
                    mask=anat['mask'],
                    seed=seed,
                    fwhm=fwhm
                )
                statmap_label_default = f'Connectivity @ {round(seed[0])}, {round(seed[1])}, {round(seed[2])} (r)'
                if session:
                    statmap_label_default += f', {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 0.5
            elif stat_type == 'connectivity_local':
                seed = (
                    get_value(statmap, 'seed_x'),
                    get_value(statmap, 'seed_y'),
                    get_value(statmap, 'seed_z'),
                )
                if seed[0] is None or seed[1] is None or seed[2] is None:
                    continue
                fwhm = get_value(statmap, 'fwhm') or None
                functional_paths = get_value(statmap, 'functional_files') or None

                if not functional_paths:
                    continue

                functional_paths = [os.path.join(local_directory, x) for x in functional_paths]
                statmap_in = dict(
                    functionals=functional_paths,
                    mask=anat['mask'],
                    seed=seed,
                    fwhm=fwhm
                )
                statmap_label_default = f'Connectivity @ {round(seed[0])}, {round(seed[1])}, {round(seed[2])} (r)'
                if session:
                    statmap_label_default += f', {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 0.5
            else:
                raise ValueError(f'Unknown statmap type: {stat_type}. Must be one of contrast or network.')
            statmap_paths.append(statmap_in)
            statmap_labels.append(statmap_label)
            colors.append(get_value(statmap, 'color') or None)
            vmin.append(vmin_)
            vmax.append(vmax_)
            thresholds.append(get_value(statmap, 'threshold') or None)
            statmap_scales_alpha.append(get_value(statmap, 'statmap_scales_alpha'))
            skip.append((not get_value(statmap, 'show')) or False)

        plot_kwargs = dict(
            statmaps=statmap_paths,
            statmap_labels=statmap_labels,
            colors=colors,
            vmin=vmin,
            vmax=vmax,
            thresholds=thresholds,
            statmap_scales_alpha=statmap_scales_alpha,
            skip=skip,
            display_surface=display_surface,
            additive_color=additive_color,
            turn_out_hemis=turn_out_hemis
        )
        for surf_type in ('pial', 'white', 'midthickness', 'inflated' ,'sulc'):
            for hemi in ('left', 'right'):
                if not surf_type in plot_kwargs:
                    plot_kwargs[surf_type] = dict()
                plot_kwargs[surf_type][hemi] = anat[surf_type][hemi]
        plot_kwargs['progress_fn'] = progress_fn

        plot_data = pl.get_plot_data(**plot_kwargs)

        if progress_fn is not None:
            progress_fn('Rendering figure...', 0.1)

        if fig_prev is None:
            fig = pl.plot_data_to_fig(**plot_data)
            fig.layout.uirevision = True
            fig.layout.meta = dict(
                project=project,
                participant=participant,
                display_surface=display_surface,
                turn_out_hemis=turn_out_hemis
            )
        else:
            fig_ = Patch()
            for trace in fig_prev['data'][2:]:
                fig_['data'].remove(trace)
            project_prev = fig_prev.get('layout', {}).get('meta', {}).get('project', None)
            participant_prev = fig_prev.get('layout', {}).get('meta', {}).get('participant', None)
            display_surface_prev = fig_prev.get('layout', {}).get('meta', {}).get('display_surface', None)
            turn_out_hemis_prev = fig_prev.get('layout', {}).get('meta', {}).get('turn_out_hemis', None)
            use_template_surface_prev = fig_prev.get('layout', {}).get('meta', {}).get('use_template_surface', None)
            update_meshes = project != project_prev or participant != participant_prev or \
                            display_surface != display_surface_prev or turn_out_hemis_prev != turn_out_hemis or \
                            use_template_surface != use_template_surface_prev
            left = None
            right = None
            if update_meshes:
                for hemi in ('left', 'right'):
                    data = plot_data['left'] if hemi == 'left' else plot_data['right']
                    surface_ = data['mesh']
                    trace = pl.make_plot_Mesh3d(surface=surface_)
                    trace.vertexcolor = data['vertexcolor']
                    trace.customdata = data['customdata']
                    trace.hovertemplate = ''.join(
                        ['<b>' + col + ':</b> %{customdata[' + str(i) + ']:.2f}<br>'
                               for i, col in enumerate(data['customdata'].columns)]) + '<extra></extra>'
                    if hemi == 'left':
                        ix = 0
                        left = trace
                    else:
                        ix = 1
                        right = trace
                    fig_['data'][ix] = trace
            else:
                for hemi in ('left', 'right'):
                    data = plot_data['left'] if hemi == 'left' else plot_data['right']
                    if hemi == 'left':
                        ix = 0
                        left = fig_prev['data'][ix]
                    else:
                        ix = 1
                        right = fig_prev['data'][ix]
                    fig_['data'][ix]['vertexcolor'] = data['vertexcolor']
                    fig_['data'][ix]['customdata'] = data['customdata'].values

            cbar_x = 1
            cbar_step = 0.1
            extra_traces = []
            for colorbar in plot_data['colorbars']:
                # Add any colorbar traces
                extra_traces.append(pl.make_colorbar(**colorbar, cbar_x=cbar_x))
                cbar_x += cbar_step
            for seed in plot_data['seeds']:
                seed_hemi = seed['hemi']
                seed_ix = seed['ix']
                if seed_hemi == 'left':
                    trace = left
                    x, y, z = trace['x'][seed_ix], trace['y'][seed_ix], trace['z'][seed_ix]
                elif seed_hemi == 'right':
                    trace = right
                    x, y, z = trace['x'][seed_ix], trace['y'][seed_ix], trace['z'][seed_ix]
                else:
                    raise ValueError(f'Invalid seed hemisphere: {seed_hemi}. Must be "left" or "right".')
                extra_traces.append(pl.make_sphere((x, y, z)))
            fig_['data'].extend(extra_traces)

            # fig_['layout'] = fig_prev['layout']
            fig_['layout']['uirevision'] = True
            fig_['layout']['meta']['project'] = project
            fig_['layout']['meta']['participant'] = participant
            fig_['layout']['meta']['display_surface'] = display_surface
            fig_['layout']['meta']['turn_out_hemis'] = turn_out_hemis
            fig_['layout']['meta']['use_template_surface'] = use_template_surface
            fig = fig_

        if progress_fn is not None:
            progress_fn.value = 0
            progress_fn('Compiling figure', 0)

        return fig, {}

    @app.callback(Output('project-dropdown', 'options'),
                  Output('project-dropdown', 'value'),
                  Output('participant-dropdown', 'options'),
                  Output('participant-dropdown', 'value'),
                  Output('pial-left', 'value'),
                  Output('pial-right', 'value'),
                  Output('white-left', 'value'),
                  Output('white-right', 'value'),
                  Output('midthickness-left', 'value'),
                  Output('midthickness-right', 'value'),
                  Output('inflated-left', 'value'),
                  Output('inflated-right', 'value'),
                  Output('sulc-left', 'value'),
                  Output('sulc-right', 'value'),
                  Output('mask', 'value'),
                  Output('pial-left', 'options'),
                  Output('pial-right', 'options'),
                  Output('white-left', 'options'),
                  Output('white-right', 'options'),
                  Output('midthickness-left', 'options'),
                  Output('midthickness-right', 'options'),
                  Output('sulc-left', 'options'),
                  Output('sulc-right', 'options'),
                  Output('statmap-type', 'options'),
                  Output('statmap-type', 'value'),
                  Output('statmap-list', 'children'),
                  Output('participant-dropdown-wrapper', 'style'),
                  Output('preprocessing-label', 'style'),
                  Output('local-directory-wrapper', 'style'),
                  Input('project-dropdown', 'value'),
                  Input('participant-dropdown', 'value'),
                  Input('add-statmap', 'n_clicks'),
                  Input('statmap-list', 'n_clicks'),
                  Input('main', 'clickData'),
                  Input('store', 'data'),
                  Input('local-directory', 'value'),
                  Input('template-surface', 'n_clicks'),
                  Input('template-surface', 'checked'),
                  Input('pial-left', 'value'),
                  Input('pial-right', 'value'),
                  Input('white-left', 'value'),
                  Input('white-right', 'value'),
                  Input('midthickness-left', 'value'),
                  Input('midthickness-right', 'value'),
                  Input('inflated-left', 'value'),
                  Input('inflated-right', 'value'),
                  Input('sulc-left', 'value'),
                  Input('sulc-right', 'value'),
                  Input('mask', 'value'),
                  State('statmap-type', 'value'),
                  State('statmap-list', 'children'))
    def update_menu(
            project,
            participant,
            add_n_clicks,
            statmap_list_n_clicks,
            click_data,
            store,
            local_directory,
            template_surface_n_clicks,
            use_template_surface,
            pial_left,
            pial_right,
            white_left,
            white_right,
            midthickness_left,
            midthickness_right,
            inflated_left,
            inflated_right,
            sulc_left,
            sulc_right,
            mask,
            statmap_type,
            statmap_list
    ):
        if store is None:
            store = {}

        surfaces = dict(
            pial=dict(left=pial_left, right=pial_right),
            white=dict(left=white_left, right=white_right),
            midthickness=dict(left=midthickness_left, right=midthickness_right),
            inflated=dict(left=inflated_left, right=inflated_right),
            sulc=dict(left=sulc_left, right=sulc_right)
        )

        if use_template_surface:
            model_default = 'mni'
            parcellate_default = 'mni'
        else:
            model_default = MODEL_DEFAULT_KEY
            parcellate_default = PARCELLATE_DEFAULT_KEY

        if os.path.exists(BIDS_PATH):
            projects = sorted([x for x in os.listdir(BIDS_PATH) if os.path.isdir(os.path.join(BIDS_PATH, x))])
        else:
            projects = []
        projects.append('local')
        if project is None or project not in projects:
            if projects and 'climblab' in projects:
                project = 'climblab'
            else:
                project = 'local'
        local_directory_files = []
        local_directory_options = []
        mask_path = None
        if project == 'local':
            participants = []
            statmap_type_data = [
                {'label': 'File', 'value': 'file'},
                {'label': 'Connectivity', 'value': 'connectivity_local'}
            ]
            statmap_type_value = statmap_type if statmap_type else 'file'
            if local_directory and os.path.exists(local_directory):
                local_directory = os.path.abspath(local_directory)
                if not local_directory.endswith(os.sep):
                    local_directory += os.sep
                if os.path.exists(local_directory):
                    local_directory_files = os.listdir(local_directory)
                    local_directory_options = [
                        {'label': x, 'value': x} for x in local_directory_files
                    ]
            for surf_type in surfaces:
                for hemi in surfaces[surf_type]:
                    if surfaces[surf_type][hemi] not in local_directory_files:
                        if surf_type == 'sulc':
                            suffix = '.shape.gii'
                        else:
                            suffix = '.surf.gii'
                        for x in local_directory_files:
                            if x.endswith(f'_hemi-{hemi[0].upper()}_{surf_type}{suffix}'):
                                surfaces[surf_type][hemi] = x
                                break
            for x in local_directory_files:
                if x.endswith(DEFAULT_MASK_SUFFIX):
                    mask_path = x
                    break
        else:
            participants = sorted(
                [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
            )
            statmap_type_data = [
                {'label': 'Contrast', 'value': 'contrast'},
                {'label': 'Network', 'value': 'network'},
                {'label': 'Connectivity', 'value': 'connectivity'}
            ]
            statmap_type_value = statmap_type if statmap_type else 'contrast'
        if participant is None or participant not in participants:
            participant = participants[0] if participants else None
        if project == 'local':
            sessions = []
        else:
            sessions = sorted(
                [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project, f'sub-{participant}'))
                             if x.startswith('ses-')]
            )

        statmap_list_ = []
        for statmap in statmap_list:
            deleted = False
            for child in statmap['props']['children']:
                if 'id' in child['props'] and child['props']['id']['type'] == 'remove-statmap':
                    if 'n_clicks' in child['props'] and child['props']['n_clicks']:
                        deleted = True
                        break
            if not deleted:
                statmap_list_.append(statmap)
        statmap_list = statmap_list_

        if callback_context.triggered_id == 'add-statmap':
            statmap_ix = store['statmap_ix']
            children = [
                html.Label(
                    statmap_type[0].upper() + statmap_type[1:], style=dict(width='100%'),
                    id={'type': 'statmap-type', 'index': statmap_ix}
                ),
            ]
            if statmap_type == 'contrast':
                children += [
                    dmc.TextInput(
                        id={'type': 'statmap-task', 'index': statmap_ix},
                        placeholder='Task',
                        style=dict(width='49%')
                    ),
                    dmc.TextInput(
                        id={'type': 'statmap-contrast', 'index': statmap_ix},
                        placeholder='Contrast',
                        style=dict(width='49%')
                    ),
                    dmc.TextInput(
                        id={'type': 'statmap-model-label', 'index': statmap_ix},
                        placeholder=f'Model label (default: {model_default})',
                        value=model_default,
                        style=dict(width='98%')
                    ),
                ]
            elif statmap_type == 'network':
                children += [
                    dmc.TextInput(
                        id={'type': 'statmap-network', 'index': statmap_ix},
                        placeholder='Network',
                        style=dict(width='98%')
                    ),
                    dmc.TextInput(
                        id={'type': 'statmap-parcellation-label', 'index': statmap_ix},
                        placeholder=f'Parcellation label (default: {parcellate_default})',
                        value=parcellate_default,
                        style=dict(width='98%')
                    ),
                ]
            elif statmap_type == 'file':
                children += [
                    dcc.Dropdown(
                        id={'type': 'statmap-local-file', 'index': statmap_ix},
                        options=local_directory_options,
                        value=None,
                        clearable=False,
                        optionHeight=OPTION_HEIGHT,
                        style=dict(width='98%')
                    )
                ]
            elif statmap_type == 'connectivity':
                x = y = z = None
                if store is not None:
                    seed = store.get('seed', {}).get('points', [None])[0]
                    if seed is not None and 'customdata' in seed:
                        x, y, z = round(seed['customdata'][0], 2), round(seed['customdata'][1], 2), \
                                  round(seed['customdata'][2], 2)
                children += [
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-x', 'index': statmap_ix},
                        placeholder='x',
                        value=x,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-y', 'index': statmap_ix},
                        placeholder='y',
                        value=y,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-z', 'index': statmap_ix},
                        placeholder='z',
                        value=z,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-fwhm', 'index': statmap_ix},
                        placeholder='FWHM (mm)',
                        style=dict(width='24%')
                    ),
                    dmc.TextInput(
                        id={'type': 'statmap-regex-filter', 'index': statmap_ix},
                        placeholder=f'Regex filter (optional)',
                        style=dict(width='49%')
                    ),
                    dmc.TextInput(
                        id={'type': 'statmap-cleaning-label', 'index': statmap_ix},
                        placeholder=f'Cleaning label (default: {CLEAN_DEFAULT_KEY})',
                        style=dict(width='49%')
                    ),
                ]
            elif statmap_type == 'connectivity_local':
                x = y = z = None
                if store is not None:
                    seed = store.get('seed', {}).get('points', [None])[0]
                    if seed is not None and 'customdata' in seed:
                        x, y, z = round(seed['customdata'][0], 2), round(seed['customdata'][1], 2), \
                                  round(seed['customdata'][2], 2)
                children += [
                    html.Div(
                        [
                            html.Label('Functional files'),
                            dcc.Dropdown(
                                id={'type': 'statmap-functional-files', 'index': statmap_ix},
                                options=[x for x in local_directory_options if x['value'].endswith('.nii.gz') or
                                         x['value'].endswith('.nii')],
                                value=None,
                                clearable=False,
                                multi=True,
                                closeOnSelect=False,
                                optionHeight=OPTION_HEIGHT,
                                style=dict(width='98%')
                            )
                        ],
                        id={'type': 'statmap-functional-files-wrapper', 'index': statmap_ix},
                        style=dict(width='98%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-x', 'index': statmap_ix},
                        placeholder='x',
                        value=x,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-y', 'index': statmap_ix},
                        placeholder='y',
                        value=y,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-seed-z', 'index': statmap_ix},
                        placeholder='z',
                        value=z,
                        style=dict(width='24%')
                    ),
                    dmc.NumberInput(
                        id={'type': 'statmap-fwhm', 'index': statmap_ix},
                        placeholder='FWHM (mm)',
                        style=dict(width='24%')
                    ),
                ]
            children += [
                dmc.TextInput(
                    id={'type': 'statmap-text', 'index': statmap_ix},
                    placeholder='Axis text',
                    style=dict(width='49%')
                ),
                dmc.TextInput(
                    id={'type': 'statmap-color', 'index': statmap_ix},
                    placeholder='Color',
                    style=dict(width='49%')
                ),
                dmc.NumberInput(
                    id={'type': 'statmap-vmin', 'index': statmap_ix},
                    placeholder='vmin',
                    style=dict(width='24%')
                ),
                dmc.NumberInput(
                    id={'type': 'statmap-vmax', 'index': statmap_ix},
                    placeholder='vmax',
                    style=dict(width='24%')
                ),
                dmc.NumberInput(
                    id={'type': 'statmap-thresh', 'index': statmap_ix},
                    placeholder='thresh',
                    style=dict(width='24%')
                ),
                dmc.Switch(id={'type': 'statmap-statmap-scales-alpha', 'index': statmap_ix},
                           label="Statmap scales transparency", checked=True),
                dmc.Switch(id={'type': 'statmap-show', 'index': statmap_ix},
                           label="Show", checked=True),
                dmc.Button(
                    DashIconify(icon='mdi:delete-outline'),
                    id={'type': 'remove-statmap', 'index': statmap_ix},
                    variant='subtle',
                    color='red',
                    style=dict(width='24%', border='1px solid #ccc', background='#eee')
                ),
            ]

            new_statmap = dmc.Group(
                children=children,
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '0.2rem',
                    'padding': '0.5rem',
                    'margin': '0.5rem 0',
                    'border': '2px solid #ccc',
                    'borderRadius': '0.5rem'
                }
            )
            statmap_list.append(new_statmap)

        delete = not sessions

        for statmap in statmap_list:
            is_dict = isinstance(statmap, dict)
            if is_dict:
                statmap_ix = statmap['props']['children'][0]['props']['id']['index']
            else:
                statmap_ix = statmap.children[0].id['index']
            children = []
            if is_dict:
                children_src = statmap['props']['children']
            else:
                children_src = statmap.children
            found = False
            for child in children_src:
                # Update default model/parcellate labels if needed
                if callback_context.triggered_id == 'template-surface':
                    if is_dict:
                        if child['props']['id']['type'] == 'statmap-model-label':
                            child['props']['value'] = model_default
                        elif child['props']['id']['type'] == 'statmap-parcellation-label':
                            child['props']['value'] = parcellate_default
                    else:
                        if child.id['type'] == 'statmap-model-label':
                            child.value = model_default
                        elif child.id['type'] == 'statmap-parcellation-label':
                            child.value = parcellate_default
                # Remove the session selector if needed
                if is_dict:
                    found_ = 'id' in child['props'] and child['props']['id']['type'] == 'statmap-session-wrapper'
                else:
                    found_ = hasattr(child, 'id') and 'type' in child.id and \
                            child.id['type'] == 'statmap-session-wrapper'
                found = found or found_
                if found_ and delete:
                    continue
                children.append(child)
            if not delete and not found:
                session_select = html.Div(
                    [
                        html.Label('Session'),
                        dcc.Dropdown(
                            id={'type': 'statmap-session', 'index': statmap_ix},
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': s, 'value': s} for s in sessions],
                            value='all',
                            clearable=False,
                        )
                    ],
                    id={'type': 'statmap-session-wrapper', 'index': statmap_ix},
                    style=dict(width='98%')
                )

                if is_dict:
                    children = statmap['props']['children']
                else:
                    children = statmap.children
                children.insert(1, session_select)
            if is_dict:
                statmap['props']['children'] = children
            else:
                statmap.children = children

        if callback_context.triggered_id == 'local-directory':
            pial_left = surfaces['pial']['left']
            pial_right = surfaces['pial']['right']
            white_left = surfaces['white']['left']
            white_right = surfaces['white']['right']
            midthickness_left = surfaces['midthickness']['left']
            midthickness_right = surfaces['midthickness']['right']
            inflated_left = surfaces['inflated']['left']
            inflated_right = surfaces['inflated']['right']
            sulc_left = surfaces['sulc']['left']
            sulc_right = surfaces['sulc']['right']
            mask = mask_path

        return  [
            [{'label': p, 'value': p} for p in projects],
            project,
            [{'label': p, 'value': p} for p in participants],
            participant,
            pial_left,
            pial_right,
            white_left,
            white_right,
            midthickness_left,
            midthickness_right,
            inflated_left,
            inflated_right,
            sulc_left,
            sulc_right,
            mask,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            local_directory_options,
            statmap_type_data,
            statmap_type_value,
            statmap_list,
            {'display': 'none'} if project == 'local' else {},
            {'display': 'none'} if project == 'local' else {},
            {} if project == 'local' else {'display': 'none'}
        ]

    @app.callback(
        Output('menu', 'opened'),
        Input('drawer-toggle', 'n_clicks'),
        State('menu', 'opened'),
        prevent_initial_call=True
    )
    def menu_toggle(n_clicks, opened):
        if opened:
            return False
        return True

    @app.callback(Output('store', 'data'),
                  Input('main', 'clickData'),
                  Input('add-statmap', 'n_clicks'),
                  State('store', 'data'))
    def update_store(click_data, n_clicks, store):
        if store is None:
            store = {}
        if 'statmap_ix' not in store:
            store['statmap_ix'] = 0
        if click_data is not None:
            store['seed'] = click_data
        if n_clicks is not None:
            store['statmap_ix'] = n_clicks

        return store


def get_value(statmap, key):
    key_child = None
    for child in statmap['props']['children']:
        if child['props']['id']['type'][8:].replace('-', '_') == key:
            key_child = child
            break
        elif child['props']['id']['type'][8:].replace('-', '_') == f'{key}_wrapper':
            key_child = child['props']['children'][1]
            break

    if key_child is None:
        return None

    if key == 'type':
        val = key_child['props']['children'].lower()
    elif 'checked' in key_child['props']:
        val = key_child['props']['checked']
    else:
        val = key_child['props'].get('value', None)

    return val


def initialize_app(config_path):
    cache = diskcache.Cache(
        CACHE_PATH,
        size_limit=CACHE_SIZE
    )
    background_callback_manager = DiskcacheManager(cache)

    app = Dash(__name__, background_callback_manager=background_callback_manager)
    app.config_path = config_path
    app.layout = dmc.MantineProvider(layout())
    app.title = 'Cortical Surface Viewer'

    return app, cache


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('CLiMBprep Cortical Surface Viewer')
    argparser.add_argument('-c', '--config_path', help=('Path to the configuration file (YAML) used to initialize '
                                                        'the app.'))
    argparser.add_argument('-d', '--debug', action='store_true', help='Run the app in debug mode.')
    argparser.add_argument('--port', default=8050, help='Port.')
    args = argparser.parse_args()

    config_path = args.config_path
    debug = args.debug
    port = args.port

    app, cache = initialize_app(config_path)
    assign_callbacks(app, cache)
    app.run(port=port, debug=debug, dev_tools_hot_reload=False)