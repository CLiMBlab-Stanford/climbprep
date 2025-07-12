from uuid import uuid4
from plotly import graph_objects as go
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import Dash, DiskcacheManager, html, dcc, Input, Output, State, Patch, callback_context
import diskcache
import argparse

from climbprep.plot import PlotLibMemoized
from climbprep.constants import *


class Progress:
    def __init__(self, progress_fn):
        self.progress_fn = progress_fn
        self.max = 1
        self.value = 0
        self.incr = 0

    def __call__(self, message, incr=None):
        if incr is None:
            incr = self.incr
        self.progress_fn((message, str(self.value), str(self.max)))
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
                id='button1',
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
                    ],
                    style={
                        'marginTop': '2rem',
                    }
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
                        html.Label('Anatomical directory'),
                        dmc.TextInput(
                            id='anatomical-directory',
                            placeholder=f'Path to anatomical directory',
                            style={'width': '100%', 'paddingTop': '0.25rem'}
                        ),
                    ],
                    id='anatomical-directory-wrapper',
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
                                dmc.Select(
                                    id='statmap-type',
                                    data=[
                                        {'label': 'Contrast', 'value': 'contrast'},
                                        {'label': 'Network', 'value': 'network'},
                                        {'label': 'Connectivity', 'value': 'connectivity'}
                                    ],
                                    value='contrast',
                                    clearable=False,
                                    allowDeselect=False,
                                    size='sm',
                                    style=dict(width='40%')
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
                    html.Progress(
                        id='progress',
                        value='0'
                    ),
                    html.Span(
                        id='progress-text',
                        style={'marginLeft': '1rem'}
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
        cache.memoize(ignore={'progress_fn'})
    )

    @app.callback(
        Output('main', 'figure'),
        Output('main', 'style'),
        Input('compile', 'n_clicks'),
        State('project-dropdown', 'value'),
        State('participant-dropdown', 'value'),
        State('preprocessing-label', 'value'),
        State('additive-color', 'checked'),
        State('turn-out-hemis', 'checked'),
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
                    'opacity': 0.7,
                },
                {
                    'display': 'none'
                }
            )
        ],
        progress=[
            Output("progress-text", "children"),
            Output("progress", "value"),
            Output("progress", "max")
        ],
        cancel=[Input("cancel", "n_clicks")]
    )
    def update_graph(
            progress_fn,
            n_clicks,
            project,
            participant,
            preprocessing_label,
            additive_color,
            turn_out_hemis,
            statmap_list,
            fig_prev
    ):
        if progress_fn is not None:
            progress_fn = Progress(progress_fn)
        progress_fn('Initializing...', 0)
        projects = sorted([x for x in os.listdir(BIDS_PATH) if os.path.isdir(os.path.join(BIDS_PATH, x))])
        if project is None or project not in projects:
            project = 'climblab'
        participants = sorted(
            [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
        )
        if participant is None or participant not in participants:
            participant = participants[0] if participants else None

        preprocessing_label = preprocessing_label or PREPROCESS_DEFAULT_KEY
        anat = {}
        for surf_type in ('pial', 'white', 'midthickness', 'sulc'):
            if surf_type == 'sulc':
                suffix = '.shape.gii'
            else:
                suffix = '.surf.gii'
            for hemi in ('left', 'right'):
                surf_path = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'fmriprep', preprocessing_label, f'sub-{participant}', 'anat',
                    f'sub-{participant}_hemi-{hemi[0].upper()}_{surf_type}{suffix}'
                )
                if not os.path.exists(surf_path):
                    session_subdirs = [
                        x for x in os.listdir(os.path.join(
                            BIDS_PATH, project, 'derivatives', 'fmriprep', preprocessing_label, f'sub-{participant}'
                        )) if x.startswith('ses-')
                    ]
                    if session_subdirs:
                        session = session_subdirs[0][4:]
                        surf_path = os.path.join(
                            BIDS_PATH, project, 'derivatives', 'fmriprep', preprocessing_label, f'sub-{participant}',
                            f'ses-{session}', 'anat',
                            f'sub-{participant}_ses-{session}_hemi-{hemi[0].upper()}_{surf_type}{suffix}'
                        )
                if not surf_type in anat:
                    anat[surf_type] = dict()
                anat[surf_type][hemi] = surf_path

        statmaps = statmap_list
        statmap_paths = []
        statmap_labels = []
        cmaps = []
        vmin = []
        vmax = []
        thresholds = []
        statmap_scales_alpha = []
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
                statmap_img = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'firstlevels', model_label, task, f'node-{node}', subdir,
                    f'sub-{participant}{session_str}_contrast-{contrast}_stat-t_statmap.nii.gz'
                )
                if not os.path.exists(statmap_img):
                    continue
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
                statmap_img = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'parcellate', parcellation_label, f'node-{node}', subdir,
                    'parcellation', 'main', f'{network}.nii.gz'
                )
                if not os.path.exists(statmap_img):
                    continue
                statmap_label_default = f'p({network})' if not session else f'p({network}), {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 0.5
            elif stat_type == 'connectivity':
                seed = (
                    get_value(statmap, 'seed_x'),
                    get_value(statmap, 'seed_y'),
                    get_value(statmap, 'seed_z'),
                )
                if seed[0] is None or seed[1] is None or seed[2] is None:
                    continue
                fwhm = get_value(statmap, 'fwhm') or None
                cleaning_label = get_value(statmap, 'cleaning_label') or CLEAN_DEFAULT_KEY
                space = PARCELLATE_DEFAULT_KEY

                functional_paths = pl.get_functional_paths(
                    participant,
                    project=project,
                    session=session,
                    cleaning_label=cleaning_label,
                    space=space
                )

                if not functional_paths:
                    continue

                # Not really an image, just a parameterization for seed-based connectivity,
                # but this is what the plotting function expects.
                statmap_img = dict(
                    functionals=functional_paths,
                    seed=seed,
                    fwhm=fwhm
                )
                statmap_label_default = f'Connectivity @ {round(seed[0])}, {round(seed[1])}, {round(seed[2])} (r)'
                if session:
                    statmap_label_default += f', {session}'
                statmap_label = get_value(statmap, 'text') or statmap_label_default
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 0.3
            else:
                raise ValueError(f'Unknown statmap type: {stat_type}. Must be one of contrast or network.')
            statmap_paths.append(statmap_img)
            statmap_labels.append(statmap_label)
            cmaps.append(get_value(statmap, 'color') or None)
            vmin.append(vmin_)
            vmax.append(vmax_)
            thresholds.append(get_value(statmap, 'threshold') or None)
            statmap_scales_alpha.append(get_value(statmap, 'statmap_scales_alpha'))

        plot_kwargs = dict(
            statmaps=statmap_paths,
            statmap_labels=statmap_labels,
            cmaps=cmaps,
            vmin=vmin,
            vmax=vmax,
            thresholds=thresholds,
            statmap_scales_alpha=statmap_scales_alpha,
            additive_color=additive_color,
            turn_out_hemis=turn_out_hemis
        )
        for surf_type in ('pial', 'white', 'midthickness', 'sulc'):
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
            fig.layout.meta = dict(project=project, participant=participant)
        else:
            fig_ = Patch()
            for trace in fig_prev['data'][2:]:
                fig_['data'].remove(trace)
            project_prev = fig_prev.get('layout', {}).get('meta', {}).get('project', None)
            participant_prev = fig_prev.get('layout', {}).get('meta', {}).get('participant', None)
            same_participant = project == project_prev and participant == participant_prev
            if not same_participant:
                for hemi in ('left', 'right'):
                    data = plot_data['left'] if hemi == 'left' else plot_data['right']
                    ix = 0 if hemi == 'left' else 1
                    surface_ = data['mesh']
                    trace = pl.make_plot_Mesh3d(surface=surface_)
                    trace.vertexcolor = data['vertexcolor']
                    trace.customdata = data['customdata']
                    trace.hovertemplate = ''.join(['<b>' + col + ':</b> %{customdata[' + str(i) + ']:.2f}<br>'
                                                   for i, col in enumerate(data['customdata'].columns)]) + '<extra></extra>'
                    fig_['data'][ix] = trace
            else:
                for hemi in ('left', 'right'):
                    data = plot_data['left'] if hemi == 'left' else plot_data['right']
                    ix = 0 if hemi == 'left' else 1
                    fig_['data'][ix]['vertexcolor'] = data['vertexcolor']

            cbar_x = 1
            cbar_step = 0.1
            extra_traces = []
            for colorbar in plot_data['colorbars']:
                # Add any colorbar traces
                extra_traces.append(pl.make_colorbar(**colorbar, cbar_x=cbar_x))
                cbar_x += cbar_step
            for seed in plot_data['seeds']:
                # Add any seed traces
                extra_traces.append(pl.make_sphere(seed))
            fig_['data'].extend(extra_traces)

            fig_['layout'] = fig_prev['layout']
            fig_['layout']['meta']['project'] = project
            fig_['layout']['meta']['participant'] = participant
            fig = fig_

        if progress_fn is not None:
            progress_fn.value = 0
            progress_fn('', 0)

        return fig, {}

    @app.callback(Output('project-dropdown', 'options'),
                  Output('project-dropdown', 'value'),
                  Output('participant-dropdown', 'options'),
                  Output('participant-dropdown', 'value'),
                  Output('statmap-list', 'children'),
                  Output('participant-dropdown-wrapper', 'style'),
                  Output('preprocessing-label', 'style'),
                  Output('anatomical-directory-wrapper', 'style'),
                  Input('project-dropdown', 'value'),
                  Input('participant-dropdown', 'value'),
                  Input('add-statmap', 'n_clicks'),
                  Input('statmap-list', 'n_clicks'),
                  Input('main', 'clickData'),
                  Input('store', 'data'),
                  State('statmap-type', 'value'),
                  State('statmap-list', 'children'))
    def update_menu(
            project,
            participant,
            add_n_clicks,
            statmap_list_n_clicks,
            click_data,
            store,
            statmap_type,
            statmap_list
    ):
        if store is None:
            store = {}

        if os.path.exists(BIDS_PATH):
            projects = sorted([x for x in os.listdir(BIDS_PATH) if os.path.isdir(os.path.join(BIDS_PATH, x))])
        else:
            projects = []
        projects.append('local file')
        if project is None or project not in projects:
            if projects and 'climblab' in projects:
                project = 'climblab'
            else:
                project = 'local file'
        if project == 'local file':
            participants = []
        else:
            participants = sorted(
                [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
            )
        if participant is None or participant not in participants:
            participant = participants[0] if participants else None
        if project == 'local file':
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
                        placeholder=f'Model label (default: {MODEL_DEFAULT_KEY})',
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
                        placeholder=f'Parcellation label (default: {PARCELLATE_DEFAULT_KEY})',
                        style=dict(width='98%')
                    ),
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
                        id={'type': 'statmap-cleaning-label', 'index': statmap_ix},
                        placeholder=f'Cleaning label (default: {CLEAN_DEFAULT_KEY})',
                        style=dict(width='98%')
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

        return  [
            [{'label': p, 'value': p} for p in projects],
            project,
            [{'label': p, 'value': p} for p in participants],
            participant,
            statmap_list,
            {'display': 'none'} if project == 'local file' else {},
            {'display': 'none'} if project == 'local file' else {},
            {} if project == 'local file' else {'display': 'none'}
        ]

    @app.callback(
        Output('menu', 'opened'),
        Input('button1', 'n_clicks'),
        State('menu', 'opened'),
        prevent_initial_call=True
    )
    def menu_toggle(n_clicks, opened):
        if opened:
            return False
        return True

    @app.callback(Output('store', 'data'),
                  Input('main', 'clickData'),
                  State('store', 'data'))
    def update_store(click_data, store):
        if store is None:
            store = {}
        if 'statmap_ix' not in store:
            store['statmap_ix'] = 0
        if click_data is not None:
            store['seed'] = click_data

        return store


def get_value(statmap, key):
    key_child = None
    for child in statmap['props']['children']:
        if child['props']['id']['type'][8:].replace('-', '_') == key:
            key_child = child
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
        os.path.join(os.getcwd(), '.cache', 'viz'),
        size_limit=1024 * 1024 * 1024 * 16,  # 16GB
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
    args = argparser.parse_args()

    config_path = args.config_path
    debug = args.debug

    app, cache = initialize_app(config_path)
    assign_callbacks(app, cache)
    app.run(debug=debug, dev_tools_hot_reload=False)