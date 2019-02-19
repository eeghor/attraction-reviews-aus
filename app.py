import os
import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

import pandas as pd
import arrow

import random

import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read the data first
df = pd.read_csv('data/brisb.csv', parse_dates=['date_of_experience'], infer_datetime_format=True)

def selector(requirements):

    """
    return a data frame obtained from the original one (df) by filtering out all rows that don't match
    the required values provided in the dictionary requirements which looks like, for example, 
    {'age': '13-17', 'gender': 'f',...}
    """
    print(set(requirements))
    if not (set(requirements) - {'all'}) <= set(df.columns):
        raise Exception('wrong user attributes!')
        
    out = df
    
    for attribute in requirements:
    	if requirements[attribute] != 'all':
        	out = out[out[attribute] == requirements[attribute]]
        
    if not out.empty:
        return out
    else:
        raise Exception('no data matching your requirements!')

genders = ['m', 'f']

age_groups = [ag for ag in set(df['age']) if ag] 

tourist_types = [c for c in df.columns if {'yes', 'no'} <= set(df[c])]

colors = ['orange', '#2C72EC', '#24C173']

def make_number_reviews_scatter(df):

	if df.empty:
		return go.Scatter()

	d1 = df[['date_of_experience', 'id']].groupby(['date_of_experience']).count().reset_index()

	return go.Scatter(x=d1.date_of_experience, 
                    y=d1.id, 
                    mode='markers', 
                    marker=dict(size=12, line=dict(width=0), color=random.choice(colors)),
                    name='n', text='some characteristic words',)


navbar = dbc.NavbarSimple(
    				brand="Demo TripAdvisor Dashboard",
    				brand_href="#",
    				sticky="top",
						)

body = dbc.Container(
    [
        dbc.Row(
            	[
            	    dbc.Col(
            	        [
            	            html.H2("Brisbane Attractions"),
            	            
            	            dcc.Graph(
    					    id='brisb-reviews',
    					    figure={
    					        'data': [make_number_reviews_scatter(selector({'age': age_groups[-1]}))],
    					        'layout': go.Layout(
    					            xaxis={'title': 'Date'},
    					            yaxis={'title': 'Number of Reviews'},
    					            margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
    					            legend={'x': 0, 'y': 1},
    					            hovermode='closest'
    					        )}
    					        ),
            	        ]
            	    ),
            	]
        		),

        dbc.CardColumns([dbc.Card(
            [
                dbc.CardHeader("Segment 1"),
                dbc.CardBody(
                    [
                    dbc.Row(
                    [dbc.Col(dbc.DropdownMenu(
                    										id='ddm_age',
                        									label="Age",
                        									children=[dbc.DropdownMenuItem(ag, id='-'.join(['ag', str(ag)])) for ag in age_groups],
                    									)),
                    
                                        dbc.Col(dbc.DropdownMenu(
                    										id='ddm_gender',
                        									label="Gender",
                        									children=[dbc.DropdownMenuItem(g, id='-'.join(['gender', str(g)])) for g in genders],
                    									)),
                                        dbc.Col(dbc.DropdownMenu(
                    										id='ddm_ttype',
                        									label="Tourist Type",
                        									children=[dbc.DropdownMenuItem(tt, id='-'.join(['ttype', str(tt)])) for tt in tourist_types],
                    									)),]
                    )

                    ]
                ),
            ]
        ),
		
		dbc.Card(
            [
                dbc.CardHeader("Segment 2"),
                dbc.CardBody(
                    [
                    dbc.Row(
                    [dbc.Col(dbc.DropdownMenu(
                    										id='ddm_age2',
                        									label="Age",
                        									children=[dbc.DropdownMenuItem(ag, id='-'.join(['ag2', str(ag)])) for ag in age_groups],
                    									)),
                    
                                        dbc.Col(dbc.DropdownMenu(
                    										id='ddm_gender2',
                        									label="Gender",
                        									children=[dbc.DropdownMenuItem(g, id='-'.join(['gender2', str(g)])) for g in genders],
                    									)),
                                        dbc.Col(dbc.DropdownMenu(
                    										id='ddm_ttype2',
                        									label="Tourist Type",
                        									children=[dbc.DropdownMenuItem(tt, id='-'.join(['ttype2', str(tt)])) for tt in tourist_types],
                    									)),]
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

# app.layout = html.Div(children=[

# 						html.Label(id="some_label"),

# 						dbc.DropdownMenu(
# 										id='ddm',
#     									label="Age",
#     									children=[dbc.DropdownMenuItem(ag, id='-'.join(['ag', str(ag)])) for ag in age_groups],
# 												),

# 						html.Div(children=
# 									[html.H3(children='Brisbane Attractions', style={'textAlign': 'center'}),
# 										html.Span(dcc.Markdown("""compare **opinions** by tourist segment"""), 
# 											style={'textAlign': 'center'})]),

#     					dcc.Graph(
#     					    id='brisb-reviews',
#     					    figure={
#     					        'data': [make_number_reviews_scatter(selector({'age': age_groups[-1]}))],
#     					        'layout': go.Layout(
#     					            xaxis={'title': 'Date'},
#     					            yaxis={'title': 'Number of Reviews'},
#     					            margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
#     					            legend={'x': 0, 'y': 1},
#     					            hovermode='closest'
#     					        )}
#     					        ),

# 						html.Div(children=[

#     							html.Label('age group'),

#     							dcc.Dropdown(
#     								id='ag-dropdown',
#     					    		options=[{'label': ag, 'value': ag} for ag in age_groups] + [{'label': 'all', 'value': 'all'}],
#        					 			value='all',
#        					 			multi=False)
#     									], style={'width': '20%', 'display': 'inline-block'}),

# 						html.Div(children=[

#     							html.Label('gender'),

#     							dcc.Dropdown(
#     								id='gender-dropdown',
#     					    		options=[{'label': g, 'value': g} for g in genders] + [{'label': 'all', 'value': 'all'}],
#        					 			value='all')
#     									], style={'width': '20%', 'display': 'inline-block'}),
						
# 						html.Div(children=[

#     							html.Label('tourist type'),

#     							dcc.Dropdown(
#     								id='ttype-dropdown',
#     					    		options=[{'label': g, 'value': g} for g in tourist_types] + [{'label': 'all', 'value': 'all'}],
#        					 			value='all')
#     									], style={'width': '20%', 'display': 'inline-block'})
# 						])


# @app.callback(
#     dash.dependencies.Output('brisb-reviews', 'figure'), # will be updating the figure part of the Graph
#     [dash.dependencies.Input('ag-dropdown', 'value'),
#     	dash.dependencies.Input('gender-dropdown', 'value'),
#     		dash.dependencies.Input('ttype-dropdown', 'value')])  # what inputs need to be monitored to update the output (figure in Graph)?

# def update_graph(required_age_group, required_gender, required_tourist_type):

# 	# select data from the required_age_group
# 	# d = selector({'age': required_age_group})

# 	# now return the updated figure description, i.e. a dictionary with the 
# 	# keys like data and layout

# 	return {

# 			'data': [make_number_reviews_scatter(selector({'age': required_age_group, 
# 															'gender': required_gender,
# 																required_tourist_type: 'yes'}))],
# 			'layout': go.Layout(
#     					            xaxis={'title': 'Date'},
#     					            yaxis={'title': 'Number of Reviews'},
#     					            margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
#     					            legend={'x': 0, 'y': 1},
#     					            height=400,
#     					            hovermode='closest'
#     					        )
# 			}

# @app.callback(dash.dependencies.Output('some_label', 'children'),
# 				[dash.dependencies.Input('-'.join(['ag', str(ag)]), 'n_clicks_timestamp') for ag in age_groups])
# def f(*tss):

# 	_ = [ts if ts else 0 for ts in tss]

# 	if _:
# 		mx_ts = max(_)
# 		ag = age_groups[_.index(mx_ts)]
# 		return ag
# 	else:
# 		return ''

if __name__ == '__main__':

    app.run_server(debug=True)