import os
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import arrow

import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = pd.read_csv('data/brisb.csv', parse_dates=['date_of_experience'], infer_datetime_format=True)

def selector(dk):

    """
    dk is {'column': value}
    """
    
    if not (set(dk) <= set(df.columns)):
        print('wrong segments!')
        raise Exception()
        
    out = df
    
    for k in dk:
        out = out[out[k] == dk[k]]
        
    if not out.empty:
        return out
    else:
        print('empty result!')
        raise Exception()

genders = ['m', 'f']

age_groups = [ag for ag in set(df['age']) if ag] 

trav_types = ['foodie', 'luxury traveller', 'beach goer']

colors = ['orange', '#2C72EC', '#24C173', '#E332E3', '#E39132', '#948778']

data = []

for g in genders:

	for t in trav_types:

		d = selector({'gender': g, t: 'yes'})

		d1 = d[['date_of_experience', 'id']].groupby(['date_of_experience']).count().reset_index()

		d1_scatter = go.Scatter(x=d1.date_of_experience, 
                            	y=d1.id, mode='markers', 
                               	marker=dict(size=12, line=dict(width=0), color=colors.pop()),
                                name=t, text='pigeons diamonds ticket cruise champaign',)
		data.append(d1_scatter)


app.layout = html.Div(children=[

						html.Div(children=
									[html.H2(children='Brisbane Attractions', style={'textAlign': 'center'}),
										dcc.Markdown("""compare **opinions** by tourist segment""")]),

    					dcc.Graph(
    					    id='brisb-reviews',
    					    figure={
    					        'data': data,
    					        'layout': go.Layout(
    					            xaxis={'title': 'Date'},
    					            yaxis={'title': 'Number of Reviews'},
    					            margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
    					            legend={'x': 0, 'y': 1},
    					            hovermode='closest'
    					        )}),

    					html.Div(children=[
						html.Span(children=[

    							html.Label('age group'),

    							dcc.Dropdown(
    					    		options=[{'label': ag, 'value': ag} for ag in age_groups],
       					 			value=age_groups[0],
       					 			multi=False)
    									]),

						html.Span(children=[

    							html.Label('gender'),

    							dcc.Dropdown(
    					    		options=[{'label': g, 'value': g} for g in genders],
       					 			value=genders[0])
    									])
						])
						])


if __name__ == '__main__':

    app.run_server(debug=True)