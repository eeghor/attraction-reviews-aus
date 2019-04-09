import os
import dash
import dash_core_components as dcc
from dash_html_components import Img, Col, Div, Br, Span

from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from helpers import selector, normcdf

from scipy.stats import hmean
from itertools import chain
from collections import Counter
import json

from wordcloud import WordCloud
import matplotlib.pyplot as plt


class TripAdvisorDashboard:

	def __init__(self):

		self.data = pd.read_csv('data/data.csv', 
							parse_dates=['date_of_experience'],
							infer_datetime_format=True)

		self.attrs = json.load(open('data/attributes.json'))

		self.seg_options = {what: list(self.attrs[what].values()) for what in self.attrs}

		self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
		self.server = self.app.server

	def prefilter_seg_options(self):

		for what in self.seg_options:

			opts_upd = [self.seg_options[what][0]]

			for opt in self.seg_options[what][1:]:

				df_ = self.data[self.data[what].apply(lambda x: str(opt) in str(x))]

				c = set(df_['review_id'])

				if c and len(c) > 99:

					opts_upd += [opt]

			self.seg_options[what] = opts_upd

		return self


	def _make_navbar(self, brand, sticky='top'):

		return dbc.NavbarSimple(brand=brand, sticky=sticky)

	def _make_wc_card(self, pic, title):

		return dbc.Card([
					dbc.CardBody([dbc.CardImg(src=f'assets/{pic}')]),
					dbc.CardFooter(Span(title))
						])

	def make_id(self, pref, text):

		print(pref + '_' + text.lower().replace(' ',''))

		return pref + '_' + text.lower().replace(' ','')

	def _make_seg_card(self, badge_text):

		return dbc.Card([
					dbc.CardHeader([
						dbc.Row([
						dbc.Button(id=self.make_id('badge', badge_text), 
							children=badge_text, color='info', size='sm', outline=True),
						dbc.Nav(dbc.NavItem(dbc.NavLink(id=self.make_id('nl', badge_text), 
														children='segment description', 
														disabled=True, href="#")))
						]), 
									]),
					dbc.Collapse(id=self.make_id('collapse', badge_text), children=[
						dbc.Card([
						dbc.CardBody([
							dbc.Nav([
								dbc.DropdownMenu(label=it, 
												children=[dbc.DropdownMenuItem(id=self.make_id(self.make_id('mi', badge_text), c), children=c) 
															for c in self.seg_options.get(it, None)], 
												bs_size="sm", 
												nav=True, 
												style={'font-size': 14}) for it in self.seg_options 
												
									]),
									])
								]),]),
						])

	def create_body(self):

		us1, re1, us2, re2, df = self.get_scfscores({'gender': 'm', 'country': 'australia'}, {'age': '35-49'})
		self.make_wordcloud(df)

		return dbc.Container([
					dbc.Row([
						dbc.CardGroup([
						dbc.Col([self._make_wc_card('wc_seg_1.png', 'Segment 1 word cloud'), 
								 self._make_wc_card('wc_seg_2.png', 'Segment 2 word cloud'),
								 dbc.Card([dbc.CardBody([dbc.CardText(f'Users: {us1:,}(1)/{us2:,}(2)', 
									style={'font-size': 18, 'background-color': '#FAF4A0'}),
									dbc.CardText(f'Reviews: {re1:,}(1)/{re2:,}(2)', 
										style={'font-size': 18, 'background-color': '#1BF022'})
								 ])])
								 ], md=4),
						dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=self.create_fsc(df))])])], md=8)
						])
							]),
					Br(),
					dbc.Row([
						Div([
							dbc.Row([
									dbc.Col([self._make_seg_card('Segment 1')]),
									dbc.Col([self._make_seg_card('Segment 2')]),
									])
							], style={'width': '92%'}),
						Div([
									dbc.Col([dbc.Button('OK', outline=True, color='success')])
							], style={'width': '8%'})
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
			  
		return (len(set(rev_seg1['by_user'])), len(set(rev_seg1['review_id'])), 
					len(set(rev_seg2['by_user'])), len(set(rev_seg2['review_id'])), d)

	def make_wordcloud(self, df):

		wc1 = WordCloud(background_color='white', 
						width=600, height=300, max_words=300).generate_from_frequencies(df[['#seg1']].to_dict()['#seg1'])

		wc2 = WordCloud(background_color='white', 
						   width=600, height=300, max_words=300).generate_from_frequencies(df[['#seg2']].to_dict()['#seg2'])

		for i, wc in enumerate([wc1, wc2], 1):

			plt.figure(figsize=(5,6))
			fig = plt.imshow(wc, interpolation='bilinear')
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			plt.axis("off")
			plt.savefig(f'assets/wc_seg_{i}.png', dpi=300, bbox_inches = 'tight', pad_inches = 0.0)

		return self



if __name__ == '__main__':

	tad = TripAdvisorDashboard().prefilter_seg_options()

	tad.app.layout = tad.create_layout()

	@tad.app.callback(
		Output('collapse_segment1', "is_open"),
			[Input('badge_segment1', "n_clicks")],
				[State('collapse_segment1', "is_open")],
					)

	def toggle_collapse1(n, is_open):
		
		if n:
			return not is_open
		return is_open

	@tad.app.callback(
		Output('collapse_segment2', "is_open"),
			[Input('badge_segment2', "n_clicks")],
				[State('collapse_segment2', "is_open")],
					)

	def toggle_collapse2(n, is_open):
		
		if n:
			return not is_open
		return is_open

	@tad.app.callback(
		Output('nl_segment1', 'children'),
		[Input('mi_segment1_18-24', 'n_clicks_timestamp')]
		)
	def update_description(when_clicked):

		if when_clicked:
			return '18-24'

	tad.app.run_server(debug=True)

	
	
	

