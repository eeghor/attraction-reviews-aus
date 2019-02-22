import os
import dash
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

import pandas as pd

import plotly.graph_objs as go

# read the data first
df = pd.read_csv('data/brisb.csv.gz', parse_dates=['date_of_experience'],
											infer_datetime_format=True)

genders = ['all', 'm', 'f']
age_groups = ['all'] + sorted([ag for ag in set(df['age']) if '-' in str(ag)], key=lambda x: int(x.split('-')[0]))
tourist_types = ['all'] + [c for c in df.columns if {'yes', 'no'} <= set(df[c])]
countries = ['all'] + sorted(list({c.title() for c in set(df['country']) if str(c).lower().strip() and 
												(str(c).lower().strip() not in ['none', 'nan'])}), 
													key=lambda x: x.split()[0])
colors = {'seg1': 'orange',
			'seg2': '#2C72EC'}

def selector(col_names_and_values):

    """
    return a data frame obtained from the original one (df) by filtering out all rows that don't match
    the required values provided in the dictionary requirements which looks like, for example, 
    {'age': '13-17', 'gender': 'f',...}
    """

    actual_cols = set(df.columns) | {'tourist_type'}
    required_cols = set(col_names_and_values)

    if not (required_cols <= actual_cols):
        raise Exception('asking for wrong columns!')

    out = df

    for col in required_cols:
    	if col_names_and_values[col] != 'all':
    		if col != 'tourist_type':
    			out = out[out[col] == col_names_and_values[col]]
    		else:
    			out = out[out[col_names_and_values[col]] == 'yes']

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
                    marker=dict(size=12, line=dict(width=0), color=color, opacity=0.95),
                    name=name, 
                    # text='top characteristic words',
                    )

	return sc

def make_dropdown(attr_name, seg_num, attr_options):

	return dbc.DropdownMenu(
                    id=f'ddm-{attr_name}-seg-{seg_num}',
                    label=attr_name.title(),
                    bs_size="sm",
                    nav=True,
                    in_navbar=True,
                    children=[dbc.DropdownMenuItem(_, id=f'seg-{seg_num}-{attr_name}-' + _.replace(' ', '_'), disabled=False) for _ in attr_options],
                    )


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
						dbc.Nav(
                    	[
                    	make_dropdown('age', '1', age_groups), 
                    	make_dropdown('gender', '1', genders),
                 		make_dropdown('type', '1', tourist_types),
                 		make_dropdown('country', '1', countries),
                    	])

                    ]
                ),
            ]
        ),
		
		dbc.Card(
            [
                dbc.CardHeader(dbc.Badge("Segment 2", color='info')),
                dbc.CardBody(
                    [
                    dbc.Nav(
                    	[
                    	make_dropdown('age', '2', age_groups), 
                    	make_dropdown('gender', '2', genders),
                 		make_dropdown('type', '2', tourist_types),
                 		make_dropdown('country', '2', countries),
                    	])
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
    [dash.dependencies.Input('seg-1-age-' + ag, 'n_clicks_timestamp') for ag in age_groups] +
    	[dash.dependencies.Input('seg-2-age-' + ag, 'n_clicks_timestamp') for ag in age_groups] +
    		[dash.dependencies.Input('seg-1-gender-' + g, 'n_clicks_timestamp') for g in genders] +
    			[dash.dependencies.Input('seg-2-gender-' + g, 'n_clicks_timestamp') for g in genders] +
    				[dash.dependencies.Input('seg-1-type-' + tp.replace(' ','_'), 'n_clicks_timestamp') for tp in tourist_types] +
    					[dash.dependencies.Input('seg-2-type-' + tp.replace(' ','_'), 'n_clicks_timestamp') for tp in tourist_types] +
    						[dash.dependencies.Input('seg-1-country-' + c.replace(' ','_'), 'n_clicks_timestamp') for c in countries] +
    							[dash.dependencies.Input('seg-2-country-' + c.replace(' ','_'), 'n_clicks_timestamp') for c in countries]
    	)  # what inputs need to be monitored to update the output (figure in Graph)?

def update_graph(*menu_items_click_timestamps):

	t = list(menu_items_click_timestamps)

	tts_seg1_age = [_ if _ else 0 for _ in t[:len(age_groups)]]
	tts_seg2_age = [_ if _ else 0 for _ in t[len(age_groups):2*len(age_groups)]]

	tts_seg1_gend = t[2*len(age_groups):2*len(age_groups)+len(genders)]
	tts_seg2_gend = t[2*len(age_groups)+len(genders):2*len(age_groups)+2*len(genders)]

	tts_seg1_types = t[2*len(age_groups)+2*len(genders):2*len(age_groups)+2*len(genders)+len(tourist_types)]
	tts_seg2_types = t[2*len(age_groups)+2*len(genders)+len(tourist_types):2*len(age_groups)+2*len(genders)+2*len(tourist_types)]

	tts_seg1_countries = t[2*len(age_groups)+2*len(genders)+2*len(tourist_types):2*len(age_groups)+2*len(genders)+2*len(tourist_types) + len(countries)]
	tts_seg2_countries = t[2*len(age_groups)+2*len(genders)+2*len(tourist_types) + len(countries):2*len(age_groups)+2*len(genders)+2*len(tourist_types) + 2*len(countries)]

	if not any(tts_seg1_age):
		selected_ag_seg1 = age_groups[-1]
	else:
		max_ts_seg1 = max([ts if ts else 0 for ts in tts_seg1_age])
		selected_ag_seg1 = age_groups[tts_seg1_age.index(max_ts_seg1)]

	if not any(tts_seg2_age):
		selected_ag_seg2 = 'all'
	else:
		max_ts_seg2 = max([ts if ts else 0 for ts in tts_seg2_age])
		selected_ag_seg2 = age_groups[tts_seg2_age.index(max_ts_seg2)]

	if not any(tts_seg1_gend):
		selected_gen_seg1 = 'all'
	else:
		max_gen_seg1 = max([ts if ts else 0 for ts in tts_seg1_gend])
		selected_gen_seg1 = genders[tts_seg1_gend.index(max_gen_seg1)]

	if not any(tts_seg2_gend):
		selected_gen_seg2 = 'all'
	else:
		max_gen_seg2 = max([ts if ts else 0 for ts in tts_seg2_gend])
		selected_gen_seg2 = genders[tts_seg2_gend.index(max_gen_seg2)]

	if not any(tts_seg1_types):
		selected_types_seg1 = 'all'
	else:
		max_type_seg1 = max([ts if ts else 0 for ts in tts_seg1_types])
		selected_types_seg1 = tourist_types[tts_seg1_types.index(max_type_seg1)]

	if not any(tts_seg2_types):
		selected_types_seg2 = 'all'
	else:
		max_type_seg2 = max([ts if ts else 0 for ts in tts_seg2_types])
		selected_types_seg2 = tourist_types[tts_seg2_types.index(max_type_seg2)]

	if not any(tts_seg1_countries):
		selected_country_seg1 = 'all'
	else:
		max_country_seg1 = max([ts if ts else 0 for ts in tts_seg1_countries])
		selected_country_seg1 = countries[tts_seg1_countries.index(max_country_seg1)]

	if not any(tts_seg2_countries):
		selected_country_seg2 = 'all'
	else:
		max_country_seg2 = max([ts if ts else 0 for ts in tts_seg2_countries])
		selected_country_seg2 = countries[tts_seg2_countries.index(max_country_seg2)]


	df_seg1 = selector({'age': selected_ag_seg1, 'gender': selected_gen_seg1, 'tourist_type': selected_types_seg1, 'country': selected_country_seg1})
	df_seg2 = selector({'age': selected_ag_seg2, 'gender': selected_gen_seg2, 'tourist_type': selected_types_seg2, 'country': selected_country_seg2})

	fig_data = [make_number_reviews_scatter(df_seg1, name='segment 1: ' + '/'.join([selected_gen_seg1, selected_ag_seg1, selected_types_seg1, selected_country_seg1]), 
						color=colors['seg1']), 
				make_number_reviews_scatter(df_seg2, name='segment 2: ' + '/'.join([selected_gen_seg2, selected_ag_seg2, selected_types_seg2, selected_country_seg2]), 
						color=colors['seg2'])]

	return {'data': fig_data,
			'layout': go.Layout(
							xaxis={'title': 'Date'},
							yaxis={'title': 'Number of Reviews'},
							margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
							legend={'x': 0, 'y': 1},
							# height=500,
							hovermode='closest')
			}

if __name__ == '__main__':

	app.run_server(debug=True)