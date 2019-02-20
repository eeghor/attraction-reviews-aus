import os
import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

import pandas as pd

import random

import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read the data first
df = pd.read_csv('data/brisb.csv', parse_dates=['date_of_experience'],
									infer_datetime_format=True)

genders = ['m', 'f']
age_groups = sorted([ag for ag in set(df['age']) if '-' in str(ag)], key=lambda x: int(x.split('-')[0]))
tourist_types = [c for c in df.columns if {'yes', 'no'} <= set(df[c])]

colors = {'seg1': 'orange',
			'seg2': '#2C72EC'}

def selector(col_names_and_values):

    """
    return a data frame obtained from the original one (df) by filtering out all rows that don't match
    the required values provided in the dictionary requirements which looks like, for example, 
    {'age': '13-17', 'gender': 'f',...}
    """

    actual_cols = set(df.columns)
    required_cols = set(col_names_and_values)

    if not (required_cols <= actual_cols):
        raise Exception('asking for wrong columns!')

    out = df

    for col in required_cols:
    	out = out[out[col] == col_names_and_values[col]]

    if not out.empty:
        return out
    else:
        raise Exception('no data matching your requirements!')

def make_number_reviews_scatter(df, name, color):

	"""
	return a scatter plot showing review counts
	"""

	if df.empty:
		return go.Scatter()

	d1 = df[['date_of_experience', 'id']].groupby(['date_of_experience']).count().reset_index()

	sc = go.Scatter(x=d1.date_of_experience, y=d1.id,
                    mode='markers',
                    marker=dict(size=12, line=dict(width=0), color=color),
                    name=name, 
                    text='top characteristic words',)

	return sc


navbar = dbc.NavbarSimple(
	brand="Demo TripAdvisor Dashboard",
	brand_href="#",
	sticky="top",)

body = dbc.Container([

	dbc.Row([
			dbc.Col(
				[	
				html.Br(),
				dcc.Graph(id='brisb-reviews',),
				]
				), 
			]),

			dbc.CardDeck([

        	dbc.Card(
            [
                dbc.CardHeader(dbc.Badge("Segment 1", color='info')),
                dbc.CardBody(
                    [
                    dbc.Row(
                    [dbc.Col(dbc.DropdownMenu(
                    						id='ddm-age-seg-1',
                        					label="Age",
                        					bs_size="sm",
                        					children=[dbc.DropdownMenuItem(ag, id='seg-1-' + ag, disabled=False) for ag in age_groups],
                    						)),
                    
                    dbc.Col(dbc.DropdownMenu(
                    				id='ddm-gen-seg-1',
                    				label="Gender",
                    				bs_size="sm",
                    				style={'color': 'info'},
                    				in_navbar=True,
                    				children=[dbc.DropdownMenuItem('m', id='gender1-m', disabled=False),
                    								dbc.DropdownMenuItem('f', id='gender1-f', disabled=False)],
                    			)),
                    dbc.Col(dbc.DropdownMenu(
                    					id='ddm_ttype1',
                        				label="Type",
                        				bs_size="sm",
                        				children=[dbc.DropdownMenuItem(tt, id='-'.join(['ttype1', str(tt)])) for tt in tourist_types],
                    					)),]
                    )

                    ]
                ),
            ]
        ),
		
		dbc.Card(
            [
                dbc.CardHeader(dbc.Badge("Segment 2", color='info')),
                dbc.CardBody(
                    [
                    dbc.Row(
                    [dbc.Col(dbc.DropdownMenu(
                    							id='ddm_age2',
                        						label="Age",
                        						bs_size="sm",
                        						children=[dbc.DropdownMenuItem(ag, id='-'.join(['ag2', str(ag)])) for ag in age_groups],
                    						 ), width='auto'),
                    
                             dbc.Col(dbc.DropdownMenu(
                    							id='ddm_gender2',
                        						label="Gender",
                        						bs_size="sm",
                        						children=[dbc.DropdownMenuItem(g, id='-'.join(['gender2', str(g)])) for g in genders],
                    							)),
                             dbc.Col(dbc.DropdownMenu(
                    							id='ddm_ttype2',
                        						label="Type",
                        						bs_size="sm",
                        						children=[dbc.DropdownMenuItem(tt, id='-'.join(['ttype2', str(tt)])) for tt in tourist_types],
                    									)), ]
                    )
                    ]
                ),
            ]
        ),

        ]

        ),
    ],
    className="main-container",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([navbar, body])

@app.callback(
    dash.dependencies.Output('brisb-reviews', 'figure'), # will be updating the figure part of the Graph
    [dash.dependencies.Input('seg-1-' + ag, 'n_clicks_timestamp') for ag in age_groups] +
    	[dash.dependencies.Input('ag2-' + ag, 'n_clicks_timestamp') for ag in age_groups] +
    		[dash.dependencies.Input('gender1-m', 'n_clicks_timestamp'),
    			dash.dependencies.Input('gender1-f', 'n_clicks_timestamp')]
    	)  # what inputs need to be monitored to update the output (figure in Graph)?

def update_graph(*menu_items_click_timestamps):

	print('menu_items_click_timestamps=', menu_items_click_timestamps)
	t = list(menu_items_click_timestamps)

	print('menu_items_click_timestamps=', t)
	tss_seg1 = [_ if _ else 0 for _ in t[:len(age_groups)]]
	print('tss_seg1=', tss_seg1)
	tss_seg2 = [_ if _ else 0 for _ in t[len(age_groups):2*len(age_groups)]]
	print('tss_seg2=', tss_seg2)
	tss_gend_seg1 = t[2*len(age_groups):]
	print('tss_gend_seg1=', tss_gend_seg1)

	if not any(tss_seg1):
		selected_ag_seg1 = age_groups[-1]
		print('default segment 1: ', selected_ag_seg1)
	else:
		max_ts_seg1 = max([ts if ts else 0 for ts in tss_seg1])
		print('max_ts_seg1=', max_ts_seg1)
		selected_ag_seg1 = age_groups[tss_seg1.index(max_ts_seg1)]
		print('selected_ag_seg1=', selected_ag_seg1)

	if not any(tss_seg2):
		selected_ag_seg2 = age_groups[-2]
		print('default segment 2: ', selected_ag_seg2)
	else:
		max_ts_seg2 = max([ts if ts else 0 for ts in tss_seg2])
		selected_ag_seg2 = age_groups[tss_seg2.index(max_ts_seg2)]
		print('selected_ag_seg2=', selected_ag_seg2)


	df_seg1 = selector({'age': selected_ag_seg1})
	df_seg2 = selector({'age': selected_ag_seg2})

	fig_data = [make_number_reviews_scatter(df_seg1, name='segment 1: ' + selected_ag_seg1, color=colors['seg1']), 
				make_number_reviews_scatter(df_seg2, name='segment 2: ' + selected_ag_seg2, color=colors['seg2'])]

	return {'data': fig_data,
			'layout': go.Layout(
							xaxis={'title': 'Date'},
							yaxis={'title': 'Number of Reviews'},
							margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
							legend={'x': 0, 'y': 1},
							height=400,
							hovermode='closest')}

if __name__ == '__main__':
	app.run_server(debug=True)