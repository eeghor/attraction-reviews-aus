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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# read the data first
df = pd.read_csv('data/brisb.csv', parse_dates=['date_of_experience'], infer_datetime_format=True)

def selector(requirements):

    """
    return a data frame obtained from the original one (df) by filtering out all rows that don't match
    the required values provided in the dictionary requirements which looks like, for example, 
    {'age': '13-17', 'gender': 'f',...}
    """
    
    if not (set(requirements) - {'all'} <= set(df.columns)):
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

# trav_types = ['foodie', 'luxury traveller', 'beach goer']

colors = ['orange', '#2C72EC', '#24C173']

def make_number_reviews_scatter(df):

	d1 = df[['date_of_experience', 'id']].groupby(['date_of_experience']).count().reset_index()

	return go.Scatter(x=d1.date_of_experience, 
                    y=d1.id, 
                    mode='markers', 
                    marker=dict(size=12, line=dict(width=0), color=random.choice(colors)),
                    name='n', text='some characteristic words',)

app.layout = html.Div(children=[

						html.Div(children=
									[html.H2(children='Brisbane Attractions', style={'textAlign': 'center'}),
										html.Span(dcc.Markdown("""compare **opinions** by tourist segment"""), style={'textAlign': 'center'})]),

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

						html.Div(children=[

    							html.Label('age group'),

    							dcc.Dropdown(
    								id='ag-dropdown',
    					    		options=[{'label': ag, 'value': ag} for ag in age_groups] + [{'label': 'all', 'value': 'all'}],
       					 			value='all',
       					 			multi=False)
    									], style={'width': '20%', 'display': 'inline-block'}),

						html.Div(children=[

    							html.Label('gender'),

    							dcc.Dropdown(
    								id='gender-dropdown',
    					    		options=[{'label': g, 'value': g} for g in genders] + [{'label': 'all', 'value': 'all'}],
       					 			value='all')
    									], style={'width': '20%', 'display': 'inline-block'})
						])


@app.callback(
    dash.dependencies.Output('brisb-reviews', 'figure'), # will be updating the figure part of the Graph
    [dash.dependencies.Input('ag-dropdown', 'value'),
    dash.dependencies.Input('gender-dropdown', 'value')])  # what inputs need to be monitored to update the output (figure in Graph)?

def update_graph(required_age_group, required_gender):

	# select data from the required_age_group
	# d = selector({'age': required_age_group})

	# now return the updated figure description, i.e. a dictionary with the 
	# keys like data and layout

	return {

			'data': [make_number_reviews_scatter(selector({'age': required_age_group, 
															'gender': required_gender}))],
			'layout': go.Layout(
    					            xaxis={'title': 'Date'},
    					            yaxis={'title': 'Number of Reviews'},
    					            margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
    					            legend={'x': 0, 'y': 1},
    					            hovermode='closest'
    					        )
			}


if __name__ == '__main__':

    app.run_server(debug=True)