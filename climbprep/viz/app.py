import os
import pprint
import json
import yaml
import time
import numpy as np
from nilearn import image
from nilearn.surface import PolyMesh, PolyData
from plotly import graph_objects as go
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import Dash, html, dcc, Input, Output, State, Patch, callback_context
import argparse

from climbprep.plot import plot_surface, get_functionals_and_masker, connectivity_from_seed, generate_sphere
from climbprep.constants import *

dirpath = '/mnt/c/Users/corys/Downloads/c001'


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

    loading = dcc.Loading(
        id='viewport-loader',
        children=[div],
        target_components={'main': '*'}
    )

    return loading


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
        ),
        dmc.Affix(
            dmc.Button(
                DashIconify(icon='mdi:brain', color='#fff',
                            style={'width': '3rem', 'height': '3rem'}),
                id='compile',
                style=dict(
                    height='4rem',
                    width='4rem',
                )
            ),
            position=dict(top='1rem', left='6rem'),
        ),
        dmc.Drawer(
            [
                html.Label('Projects'),
                dcc.Dropdown(
                    [],
                    None,
                    id='project-dropdown',
                    clearable=False
                ),
                html.Label('Participant'),
                dcc.Dropdown(
                    [],
                    None,
                    id='participant-dropdown',
                    clearable=False
                ),
                dmc.TextInput(
                    id='preprocessing-label',
                    placeholder=f'Preprocessing label (default: {PREPROCESS_DEFAULT_KEY})',
                    style={'width': '100%', 'padding-top': '0.25rem'}
                ),
                dmc.Switch(
                    id='additive-color',
                    label="Additive color",
                    checked=True,
                    style={'width': '49%', 'padding-top': '0.25rem'}
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
            opened=True,
        )
    ])

    return menu


def layout():
    return html.Div(
        children=[
            dcc.Store(id='store', storage_type='memory'),
            menu(),
            viewport()
        ]
    )


def assign_callbacks(app):
    @app.callback(Output('main', 'figure'),
                  Output('main', 'style'),
                  Input('compile', 'n_clicks'),
                  State('project-dropdown', 'value'),
                  State('participant-dropdown', 'value'),
                  State('preprocessing-label', 'value'),
                  State('additive-color', 'checked'),
                  State('statmap-list', 'children'),
                  prevent_initial_call=True)
    def update_graph(n_clicks, project, participant, preprocessing_label, additive_color, statmap_list):
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
                        x for x in os.path.join(
                            BIDS_PATH, project, 'derivatives', 'fmriprep', preprocessing_label, f'sub-{participant}'
                        ) if x.startswith('ses-')
                    ]
                    if session_subdirs:
                        session = session_subdirs[0][4:]
                        surf_path = os.path.join(
                            BIDS_PATH, project, 'derivatives', 'fmriprep', preprocessing_label, f'sub-{participant}',
                            f'ses-{session}', 'anat',
                            f'sub-{participant}ses-{session}_hemi-{hemi[0].upper()}_{surf_type}{suffix}'
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
                task = get_value(statmap, 'task')
                contrast = get_value(statmap, 'contrast')
                model_label = get_value(statmap, 'model_label') or MODEL_DEFAULT_KEY
                if task is None or contrast is None:
                    continue
                statmap_img = os.path.join(
                    BIDS_PATH, project, 'derivatives', 'firstlevels', model_label, task, f'node-{node}', subdir,
                    f'sub-{participant}{session_str}_contrast-{contrast}_stat-t_statmap.nii.gz'
                )
                if not os.path.exists(statmap_img):
                    continue
                statmap_label = statmap.get('label', f'{contrast} (t)')
                vmin_ = statmap.get('vmin', 0)
                vmax_ = statmap.get('vmax', 5)
            elif stat_type == 'network':
                network = get_value(statmap, 'network')
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
                statmap_label = statmap.get('label', f'p({network})')
                vmin_ = statmap.get('vmin', 0)
                vmax_ = statmap.get('vmin', 0.8)
            elif stat_type == 'connectivity':
                seed = (
                    get_value(statmap, 'seed_x'),
                    get_value(statmap, 'seed_y'),
                    get_value(statmap, 'seed_z'),
                )
                if seed[0] is None or seed[1] is None or seed[2] is None:
                    continue
                fwhm = get_value(statmap, 'fwhm')
                cleaning_label = get_value(statmap, 'cleaning_label') or CLEAN_DEFAULT_KEY
                space = PARCELLATE_DEFAULT_KEY

                t0 = time.time()
                functionals, masker = get_functionals_and_masker(
                    participant,
                    project=project,
                    session=session,
                    cleaning_label=cleaning_label,
                    space=space
                )
                t1 = time.time()
                print('Getting functionals and masker took %.2f seconds.' % (t1 - t0))

                # Not really an image, just a parameterization for seed-based connectivity,
                # but this is what the plotting function expects.
                statmap_img = dict(
                    functionals=functionals,
                    masker=masker,
                    seed=seed,
                    fwhm=fwhm
                )
                statmap_label = statmap.get('label', 'Connectivity (r)')
                vmin_ = get_value(statmap, 'vmin') or 0
                vmax_ = get_value(statmap, 'vmax') or 0.5
            else:
                raise ValueError(f'Unknown statmap type: {stat_type}. Must be one of contrast or network.')
            statmap_paths.append(statmap_img)
            statmap_labels.append(statmap_label)
            cmaps.append(get_value(statmap, 'color'))
            vmin.append(vmin_)
            vmax.append(vmax_)
            thresholds.append(get_value(statmap, 'threshold'))
            statmap_scales_alpha.append(get_value(statmap, 'statmap_scales_alpha'))

        plot_kwargs = dict(
            statmaps=statmap_paths,
            statmap_labels=statmap_labels,
            cmaps=cmaps,
            vmin=vmin,
            vmax=vmax,
            thresholds=thresholds,
            statmap_scales_alpha=statmap_scales_alpha,
            additive_color=additive_color
        )
        for surf_type in ('pial', 'white', 'midthickness', 'sulc'):
            for hemi in ('left', 'right'):
                if not surf_type in plot_kwargs:
                    plot_kwargs[surf_type] = dict()
                plot_kwargs[surf_type][hemi] = anat[surf_type][hemi]

        t0 = time.time()
        fig = plot_surface(**plot_kwargs)
        t1 = time.time()
        print('Plotting surface took %.2f seconds.' % (t1 - t0))

        return fig, {}

    @app.callback(Output('project-dropdown', 'options'),
                  Output('project-dropdown', 'value'),
                  Output('participant-dropdown', 'options'),
                  Output('participant-dropdown', 'value'),
                  Output('statmap-list', 'children'),
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

        projects = sorted([x for x in os.listdir(BIDS_PATH) if os.path.isdir(os.path.join(BIDS_PATH, x))])
        if project is None or project not in projects:
            project = 'climblab'
        participants = sorted(
            [x[4:] for x in os.listdir(os.path.join(BIDS_PATH, project)) if x.startswith('sub-')]
        )
        if participant is None or participant not in participants:
            participant = participants[0] if participants else None
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
                        x, y, z = seed['customdata'][0], seed['customdata'][1], seed['customdata'][2]
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
                    found_ = 'id' in child['props'] and child['props']['id']['type'] == 'statmap-session'
                else:
                    found_ = hasattr(child, 'id') and 'type' in child.id and \
                            child.id['type'] == 'statmap-session'
                found = found or found_
                if found_ and delete:
                    continue
                children.append(child)
            if not delete and not found:
                session_select = dmc.RadioGroup(
                    id={'type': 'statmap-session', 'index': statmap_ix},
                    children=dmc.Group([dmc.Radio(k, value=k) for k in ['all'] + sessions], my=10),
                    value='all',
                    label='Session',
                    mb=10,
                    style=dict(width='98%'),
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
            statmap_list
        ]

    @app.callback(
        Output('menu', 'opened'),
        Input('button1', 'n_clicks')
    )
    def menu_toggle(n_clicks):
        return True

    @app.callback(Output('store', 'data'),
                  Input('main', 'clickData'),
                  State('store', 'data'))
    def capture_seed(click_data, store):
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
    app = Dash(__name__)
    app.config_path = config_path
    app.layout = dmc.MantineProvider(layout())
    app.title = 'Cortical Surface Viewer'

    return app


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('CLiMBprep Cortical Surface Viewer')
    argparser.add_argument('-c', '--config_path', help=('Path to the configuration file (YAML) used to initialize '
                                                        'the app.'))
    argparser.add_argument('-d', '--debug', action='store_true', help='Run the app in debug mode.')
    args = argparser.parse_args()

    config_path = args.config_path
    debug = args.debug

    app = initialize_app(config_path)
    assign_callbacks(app)
    app.run(debug=debug, dev_tools_hot_reload=False)