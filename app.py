import os
import dash
import dash_core_components as dcc
from dash_html_components import Img, Col, Div, Br, Span

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from helpers import selector, normcdf

from scipy.stats import hmean
from itertools import chain
from collections import Counter

class TripAdvisorDashboard:

	def __init__(self):

		self.data = pd.read_csv('data/data.csv', 
							parse_dates=['date_of_experience'],
							infer_datetime_format=True)

	def _make_navbar(self, brand, sticky='top'):

		return dbc.NavbarSimple(brand=brand, sticky=sticky)

	def _make_wc_card(self, pic, title):

		return dbc.Card([
					dbc.CardBody([dbc.CardImg(src=f'assets/{pic}')]),
					dbc.CardFooter(Span(title))
						])

	def _make_seg_card(self, badge_text, menu_item_list):

		return dbc.Card([
					dbc.CardHeader([
						dbc.Badge(badge_text, color='info')], style={'display': 'inline-grid'}
									),
					dbc.Collapse([
						dbc.CardBody([
							dbc.Nav([
								dbc.DropdownMenu(label=it, bs_size="sm", nav=True) for it in menu_item_list
									]),
									])
								]),
					dbc.CardFooter([
						dbc.Nav(dbc.NavItem(dbc.NavLink("females/foodie/australia/20-24", disabled=True, href="#")))
									])
						])

	def create_body(self):

		df = self.get_scfscores({'gender': 'm'}, {'gender': 'f'})

		return dbc.Container([
					dbc.Row([
						dbc.Col([self._make_wc_card('wc_seg_1.png', 'Segment 1 word cloud'), 
								 self._make_wc_card('wc_seg_2.png', 'Segment 2 word cloud')], md=4),
						dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=self.create_fsc(df))])])], md=8)
							]),
					dbc.Row([
							self._make_seg_card('Segment 1', 'Age Gender Country Type'.split()),
							self._make_seg_card('Segment 2', 'Age Gender Country Type'.split())
						])
							 ])

	def create_layout(self):

		return Div([self._make_navbar(brand='TripAdvisor Reviews for Melbourne'), 
					self.create_body()])

	def create_fsc(self, data, min_freq=4):

		scale_marker = lambda fscore: 4 if abs(fscore) < 0.2 else 6 if 0.2 <= abs(fscore) <=0.4 else 8

		df = data[(data['#seg1'] > min_freq) & (data['#seg2'] > min_freq)]

		layout = go.Layout(
					hovermode= 'closest',
					autosize=False,
					width=700,
					height=540,
					margin=go.layout.Margin(
											l=0,
											r=0,
											b=30,
											t=0,
											pad=0),											
					
					xaxis= dict(
        					title='Frequency in Reviews by Seg 1',
        					ticklen= 5,
        					tickmode='array',
        					tickvals=np.linspace(df['fseg1n'].min(), df['fseg1n'].max(), num=5),
        					ticktext=['low', '', '', '', 'high'],
        					zeroline= False,
        					gridwidth= 2,
        					showticklabels=True,
        					showgrid=True,),
    				
    				yaxis=dict(
        					ticklen= 5,
        					tickmode='array',
        					tickvals=np.linspace(df['fseg2n'].min(), df['fseg2n'].max(), num=5),
        					ticktext=['low', '', '', '', 'high'],
        					gridwidth= 2,
        					zeroline=False,
        					showticklabels=True,
        					showgrid=True,
        					tickangle=-90,
        					title='Frequency in Reviews by Seg 2',)
        			)

		trace = go.Scatter(
    				x = df['fseg1n'],
    				y = df['fseg2n'],
    				mode = 'markers',
    				hoverinfo='text', 
    				marker=dict(
                		cmin=-1,
                		cmax=1,
                		size=df['fscore'].apply(scale_marker), 
                		opacity=0.85,
                		color=df['fscore'],
                		colorbar = dict(
                        		title = 'Term Affinity',
                        		titleside = 'right',
                        		tickmode = 'array',
                        		tickvals = np.linspace(-1,1, num=9),
                        		ticktext = ['Seg 2','','','','','','', '', 'Seg 1'],
                        		ticks = '',
                        		tickangle=-90,
                        		outlinewidth=0),
                		colorscale='Portland',),
    				text=df.index)

		return go.Figure(data=[trace], layout=layout)


	def get_scfscores(self, seg1_dict, seg2_dict):
	 
		
		d = pd.DataFrame()

		rev_seg1 = selector(self.data, seg1_dict)
		rev_seg2 = selector(self.data, seg2_dict)

		if rev_seg1.empty or rev_seg2.empty:
			print('no scaled f-scores can be calculated due to empty segment dataframes!')
			return d
	

		d = pd.DataFrame.from_dict(Counter(chain.from_iterable(rev_seg1['lemmatised'].str.split())), orient='index').rename(columns={0: '#seg1'}) \
				.join(pd.DataFrame.from_dict(Counter(chain.from_iterable(rev_seg2['lemmatised'].str.split())), orient='index').rename(columns={0: '#seg2'}),
					 how='outer').fillna(0)

		d['pseg1'] = d['#seg1']/(d['#seg1'] + d['#seg2'])
		d['pseg2'] = 1. - d['pseg1']

		d['fseg1'] = d['#seg1']/d['#seg1'].sum()
		d['fseg2'] = d['#seg2']/d['#seg2'].sum()

		d['fscseg1'] = d.apply(lambda x: hmean([x['pseg1'], x['fseg1']]) if x['pseg1'] > 0 and x['fseg1'] > 0 else 0, axis=1)
		d['fscseg2'] = d.apply(lambda x: hmean([x['pseg2'], x['fseg2']]) if x['pseg2'] > 0 and x['fseg2'] > 0 else 0, axis=1)

		# normalize pseg1
		d['pseg1n'] = normcdf(d['pseg1'])
		# and fseg1
		d['fseg1n'] = normcdf(d['fseg1'])

		d['pseg2n'] = normcdf(d['pseg2'])
		d['fseg2n'] = normcdf(d['fseg2'])

		d['fscseg1n'] = d.apply(lambda x: hmean([x['pseg1n'], x['fseg1n']]), axis=1)
		d['fscseg2n'] = d.apply(lambda x: hmean([x['pseg2n'], x['fseg2n']]), axis=1)

		# corrected f-score
		d['fscore'] = 0

		# where the seg1 score is larger make it f-score
		d['fscore'] = d['fscore'].where(d['fscseg1n'] <= d['fscseg2n'], d['fscseg1n'])

		d['fscore'] = d['fscore'].where(d['fscseg1n'] >= d['fscseg2n'], 1. - d['fscseg2n'])

		d['fscore'] = 2*(d['fscore'] - 0.5)
			  
		return d


if __name__ == '__main__':

	app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
	server = app.server

	tad = TripAdvisorDashboard()

	app.layout = tad.create_layout()

	app.run_server(debug=True)







# wc1_card = dbc.Card([
# 				dbc.CardBody(
# 							[dbc.CardImg(src='assets/wc_seg_Q.png')] 
# 							),
# 				dbc.CardFooter(Span('Words frequently used by Seg 1'))

# 					])

# wc2_card = dbc.Card([
# 				dbc.CardBody(
# 							[dbc.CardImg(src='assets/wc_seg_Q.png')] 
# 							),
# 				dbc.CardFooter(Span('Words frequently used by Seg 1'))

# 					])

# dummy_card = dbc.Card([
# 				dbc.CardBody(
# 							[Img(src='assets/dummy.png', style={'max-width': '100%'})]
# 							)

# 					])


# body_data = dbc.Container(
# 				[
# 				dbc.Row([
# 					dbc.Col([wc1_card, wc2_card], md=4),
# 					dbc.Col([dummy_card], md=8)
# 						]),
# 				dbc.Row([
# 					dbc.Col([drop_card], md=4),
# 					dbc.Col([drop_card], md=4),
# 					dbc.Col([drop_card], md=4)
# 					])
# 				])

# # ,

# # dbc.CardColumns([
		
# # 		dbc.Card(
# # 			[
# # 				dbc.CardHeader([
					
# # 					dbc.Badge("Segment 1", color='info'),
# # 					# dbc.Fade([dbc.Badge("unavailable", color='danger')],
# # 					# 					id='seg-1-alert', is_in=False, appear=False),
# # 					], style={'display': 'inline-grid'}
# # 					),
# # 				dbc.Collapse([
# # 					dbc.CardBody([
# # 						dbc.Nav([
# # 								dbc.DropdownMenu(label='Age', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Gender', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Type', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Country', bs_size="sm", nav=True)
# # 								]),
# # 								])
# # 							]),
# # 				dbc.CardFooter([
# # 						dbc.Nav(dbc.NavItem(dbc.NavLink("females/foodie/australia/20-24", disabled=True, href="#")))
# # 					])
# # 			]),

# # 		dbc.Card(
# # 			[
# # 				dbc.CardHeader([
					
# # 					dbc.Badge("Segment 2", color='info'),
# # 					# dbc.Fade([dbc.Badge("unavailable", color='danger')],
# # 					# 					id='seg-1-alert', is_in=False, appear=False),
# # 					], style={'display': 'inline-grid'}
# # 					),
# # 				dbc.Collapse([
# # 					dbc.CardBody([
# # 						dbc.Nav([
# # 								dbc.DropdownMenu(label='Age', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Gender', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Type', bs_size="sm", nav=True),
# # 								dbc.DropdownMenu(label='Country', bs_size="sm", nav=True)
# # 								]),
# # 								])
# # 							]),
# # 				dbc.CardFooter([
# # 						dbc.Nav(dbc.NavItem(dbc.NavLink("females/like a local/hong kong/any age", disabled=True, href="#")))
# # 					])
# # 			]),

# # 		dbc.Card(
# # 			[
# # 				dbc.CardHeader([
					
# # 					dbc.Badge("Attraction Type", color='info'),
# # 					# dbc.Fade([dbc.Badge("unavailable", color='danger')],
# # 					# 					id='seg-1-alert', is_in=False, appear=False),
# # 					], style={'display': 'inline-grid'}
# # 					),
# # 				dbc.Collapse([
# # 					dbc.CardBody([
# # 						dcc.Slider(
# #         							id='year-slider',
# #         							min=2013,
# #         							max=2019,
# #         							step=1,
# #         							value=1,
# #     								),
# # 								])
# # 							]),
# # 				dbc.CardFooter([
# # 						dbc.Nav(dbc.NavItem(dbc.NavLink("museums", disabled=True, href="#")))
# # 					])
# # 			], style={'width': '100%'})


# # 		]

# # 		),
# # 	],
# # 	className="main-container",
# # )

# app.layout = Div([navbar, body_data])


# if __name__ == '__main__':

# 	app.run_server(debug=True)