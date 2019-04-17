import os
import dash
import dash_core_components as dcc
from dash_html_components import Img, Col, Div, Br, Span

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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

import io
import base64


make_id = lambda pref, text: pref + '_' + text.lower().replace(' ','')
scale_marker = lambda fscore: 4 if abs(fscore) < 0.2 else 6 if 0.2 <= abs(fscore) <=0.4 else 8
filter_df = lambda df, seg1_dict, seg2_dict: (selector(df, seg1_dict), selector(df, seg2_dict))

attr_list = ['age', 'gender', 'type', 'country', 'attraction type']

seg_descr_from_dict = lambda dict_: '/'.join([dict_[what] for what in attr_list])

def _make_seg_card(txt):
	
	return dbc.Card([
					dbc.CardHeader([
						dbc.Row([
						dbc.Button(id=make_id('badge', txt), 
							children=txt, color='info', size='sm', outline=True),
						dbc.Collapse(id=make_id('collapse', txt), 
								 is_open=False,
								 children=[
									dbc.Card([
										dbc.CardBody([
											dbc.Nav([
												dbc.DropdownMenu(label=it, 
														direction='right',
														children=[dbc.DropdownMenuItem(
																			id=make_id(make_id('mi', txt), c), 
																			children=c) 
															for c in seg_options.get(it, None)], 
														bs_size="sm", 
														nav=True, 
														style={'font-size': 14}) for it in seg_options 
												
													]),
													])
											]),
											]),
						]), 
									]),

					
					])

def calculate_scaled_fscores(df, min_freq, seg1_dict, seg2_dict):

	"""
	returns (users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d)
	"""

	d = pd.DataFrame()

	rev_seg1, rev_seg2 = filter_df(df, seg1_dict, seg2_dict)

	# collect some stats
	users_in_seg1, reviews_in_seg1 = len(set(rev_seg1['by_user'])), len(set(rev_seg1['review_id']))
	users_in_seg2, reviews_in_seg2 = len(set(rev_seg2['by_user'])), len(set(rev_seg2['review_id']))	

	if rev_seg1.empty or rev_seg2.empty:
		print('no scaled f-scores can be calculated due to empty segment data frames!')
		return (users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d)

	d = pd.DataFrame.from_dict(Counter(chain.from_iterable(rev_seg1['lemmatised'].str.split())), 
											orient='index').rename(columns={0: '#seg1'}) \
					.join(pd.DataFrame.from_dict(Counter(chain.from_iterable(rev_seg2['lemmatised'].str.split())), 
						orient='index').rename(columns={0: '#seg2'}),
					 how='outer').fillna(0)

	d = d[(d['#seg1'] > min_freq) & (d['#seg2'] > min_freq)]

	"""

	at this stage d is like 

			 #seg1  #seg2
	go         862    862
	time      1292   1292
	week        90     90
	concert    108    108
	decor       12     12

	"""

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
		  
	return (users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d)

def generate_main_figure(df):

	layout = go.Layout(
				hovermode= 'closest',
				autosize=False,
				width=700,
				height=550,
				margin=go.layout.Margin(
										l=0,
										r=0,
										b=10,
										t=10,
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


def make_wordcloud(df):

	"""
	returns a list of two word clouds (for segments 1 and 2)
	"""

	wc1 = WordCloud(background_color='white', 
					width=600, height=340, max_words=300).generate_from_frequencies(df[['#seg1']].to_dict()['#seg1'])

	wc2 = WordCloud(background_color='white', 
					   width=600, height=340, max_words=300).generate_from_frequencies(df[['#seg2']].to_dict()['#seg2'])

	pngs = []

	for i, wc in enumerate([wc1, wc2], 1):

		pil_img = wc.to_image()
		img = io.BytesIO()
		pil_img.save(img, "PNG")
		img.seek(0)
		img_b64 = base64.b64encode(img.getvalue()).decode()

		pngs.append(img_b64)
	
	return pngs

def create_app_layout(df, wc1, wc2, users_in_seg1, users_in_seg2, reviews_in_seg1, reviews_in_seg2, seg1_descr, seg2_descr):

	# figure
	fig = generate_main_figure(df)

	# navigation bar
	navbar = dbc.NavbarSimple(brand='Tourist Review Comparison', 
								sticky='top', 
								brand_style={'font-size': 22, 'color': '#1773C3'},
								children=[Img(src='assets/tripavisor_logo.png', height='36px'), 
								Img(src='assets/melbourne.png', height='30px')],
								style={'z-index': 1})

	body = dbc.Container([
					dbc.Row([
						dbc.CardGroup([
						dbc.Col([dbc.Card([
											dbc.CardBody([dbc.CardImg(id='wc1', src=f'data:image/png;base64,{wc1}')]),
											dbc.CardFooter(
												Span(id='wc1_text', children=seg1_descr)
												)
											], style={'height': '280px'}), 
								 dbc.Card([
											dbc.CardBody([dbc.CardImg(id='wc2', src=f'data:image/png;base64,{wc2}')]),
											dbc.CardFooter(
												Span(id='wc2_text', children=seg2_descr)
												)
											], style={'height': '280px'}),
								 dbc.Card([
											dbc.CardBody([dbc.CardText(id='seg_info_text_line1', children=f'Users: {users_in_seg1:,}/{users_in_seg2:,} in Seg1/Seg2', 
											style={'font-size': 18, 'background-color': '#FAF4A0'}),
											dbc.CardText(id='seg_info_text_line2', children=f'Reviews: {reviews_in_seg1:,}/{reviews_in_seg2:,} in Seg1/Seg2', 
											style={'font-size': 18, 'background-color': '#76F238'})
											])], style={'height': '112px'})
								 ], md=4),

						dbc.Col([dbc.Card([
											dbc.CardBody([dcc.Graph(id='main_graph', figure=fig)])
											]), dbc.Card([dbc.CardBody([
												dbc.Row([
												Div([
												dbc.Button('Seg 1', 
															outline=True, 
															color='success', 
															id='test_b1')], style={'width': '10%'}),
												Div([
												dbc.Button('Seg 2', 
															outline=True, 
															color='success', 
															id='test_b2')], style={'width': '10%'}),

												Div(id='seg1nav', children=[
												dbc.Nav(children=[
												dbc.DropdownMenu(label=it, 
														children=[dbc.DropdownMenuItem(
																			id=make_id(make_id('mix', 'da'), c), 
																			children=c, style={'z-index': 2, 'position': 'relative'}) 
															for c in seg_options.get(it, None)], 
														bs_size="sm", 
														nav=True, 
														style={'font-size': 14}) for it in seg_options 
												
													], justified=True)], style={'display': 'none', 'width': '70%'}),

												Div(id='seg2nav', children=[
												dbc.Nav(children=[
												dbc.DropdownMenu(label=it, 
														children=[dbc.DropdownMenuItem(
																			id=make_id(make_id('mix', 'du'), c), 
																			children=c) 
															for c in seg_options.get(it, None)], 
														bs_size="sm", 
														nav=True, 
														style={'font-size': 14}) for it in seg_options 
												
													], justified=True)], style={'display': 'none', 'width': '70%'}),
												
												Div(id='test_ok', children=[
													dbc.Button('OK', 
																id='update_everything',
																outline=True, 
																color='success')], 
													style={'display': 'none', 'width': '10%'})])

												])])], md=8)
						], style={'display': 'flex'})
							]),
						Div(id='selector description', 
							style={'display': 'none'})
						])

	return Div([navbar, body])



if __name__ == '__main__':

	data = pd.read_csv('data/data.csv.gz', 
					parse_dates=['date_of_experience'],
					infer_datetime_format=True)

	attrs = json.load(open('data/attributes.json'))
	
	seg_options = {what: list(attrs[what].values()) for what in attrs}

	default_segs = {1: {'age': 'all ages', 'gender': 'm', 'type': 'all types', 
					'country': 'all countries', 'attraction type': 'all attraction types'},
					2: {'age': 'all ages', 'gender': 'f', 'type': 'all types', 
					'country': 'all countries', 'attraction type': 'all attraction types'}}

	seg1_descr = 'Seg 1: ' + seg_descr_from_dict(default_segs[1])
	seg2_descr = 'Seg 2: ' + seg_descr_from_dict(default_segs[2])

	users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d = calculate_scaled_fscores(data, 4, default_segs[1], default_segs[2])
	# create word clouds for both segments
	wc1, wc2 = make_wordcloud(d)

	external_stylesheets = [dbc.themes.BOOTSTRAP]
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	# server = app.server
	app.layout = create_app_layout(d, wc1, wc2, users_in_seg1, users_in_seg2, 
									reviews_in_seg1, reviews_in_seg2, seg1_descr, seg2_descr)
	
	@app.callback(
		[Output('seg1nav', 'style'), 
			Output('seg2nav', 'style'),
				Output('test_b1', 'active'), 
					Output('test_b2', 'active'),
						Output('test_ok', 'style')],
		[Input('test_b1', 'n_clicks'), Input('test_b2', 'n_clicks')])

	def tc_test(nclicks_seg1, nclicks_seg2):

		styles = [{'display': 'none', 'width': '70%'}, 
					{'display': 'none', 'width': '70%'}]
		actives = [False, False]
		ok_style = {'width': '10%', 'display': 'none'}

		changed = dash.callback_context.triggered

		if changed:

			if (changed[0]['prop_id'] == 'test_b1.n_clicks') and changed[0]['value']:

				styles = [{'display': 'inline-block', 'width': '70%'}, {'display': 'none', 'width': '70%'}]
				actives = [True, False]
				ok_style['display'] = 'inline'

			elif (changed[0]['prop_id'] == 'test_b2.n_clicks') and changed[0]['value']:

				styles = [{'display': 'none', 'width': '70%'}, {'display': 'inline-block', 'width': '70%'}]
				actives = [False, True]
				ok_style['display'] = 'inline'

			elif changed[0]['value'] is None:
				raise PreventUpdate

		return styles + actives + [ok_style]

	@app.callback([Output('selector description', 'children'),
					Output('wc1_text', 'children'),
					Output('wc2_text', 'children')
					],
					[Input(make_id(make_id('mix', 'da'), _), 'n_clicks_timestamp') 
							for what in seg_options 
							for _ in seg_options[what]] + \
					[Input(make_id(make_id('mix', 'du'), _), 'n_clicks_timestamp') 
							for what in seg_options 
							for _ in seg_options[what]]
		)

	def specify_segments(*lst):

		if all([_ is None for _ in lst]):
			raise PreventUpdate

		lengths = {what: len(seg_options[what]) for what in seg_options}

		span_ends1 = np.cumsum([lengths[what] for what in seg_options])
		span_ends2 = span_ends1 + span_ends1[-1]

		spans1 = {what: (span_ends1[i-1] if i > 0 else 0, span_ends1[i]) for i, what in enumerate(seg_options)}
		spans2 = {what: (span_ends2[i-1] if i > 0 else span_ends1[-1], span_ends2[i]) for i, what in enumerate(seg_options)}

		pad_zeroes = lambda lst: [0 if not _ else _ for _ in lst]

		when_clicked1 = {what: pad_zeroes(lst[spans1[what][0]: spans1[what][1]]) for what in seg_options}
		when_clicked2 = {what: pad_zeroes(lst[spans2[what][0]: spans2[what][1]]) for what in seg_options}

		max_idx1 = {what: when_clicked1[what].index(max(when_clicked1[what])) for what in seg_options}
		max_idx2 = {what: when_clicked2[what].index(max(when_clicked2[what])) for what in seg_options}

		dict_seg1 = {'seg1': {what: seg_options[what][max_idx1[what]] for what in seg_options}}
		dict_seg2 = {'seg2': {what: seg_options[what][max_idx2[what]] for what in seg_options}}

		sel = json.dumps({**dict_seg1, **dict_seg2})

		wrd1 = 'Seg 1: ' + '/'.join([seg_options[what][max_idx1[what]] for what in seg_options])
		wrd2 = 'Seg 2: ' +'/'.join([seg_options[what][max_idx2[what]] for what in seg_options])

		return (sel, wrd1, wrd2)

	@app.callback(
		[
			Output('wc1', 'src'), 
			Output('seg_info_text_line1', 'children'),
			Output('wc2', 'src'),
			Output('seg_info_text_line2', 'children'),
			Output('main_graph', 'figure')],
		[Input('update_everything', 'n_clicks')],
		[State('selector description', 'children')]
		)

	def update(n, dict_str):

		if (not n) or (not dict_str):
			raise PreventUpdate
		
		ctx = dash.callback_context

		if not ctx.triggered:
			pass
		else:
			changed = ctx.triggered[0]['prop_id']


		if (changed == 'update_everything.n_clicks') and dict_str:
			
			d1 = json.loads(dict_str)
		
			seg1_dict = d1['seg1']
			seg2_dict = d1['seg2']

			users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d = calculate_scaled_fscores(data, 4, seg1_dict, seg2_dict)
			
			if d.empty:
				raise Exception('empty data frames!')

			wc1, wc2 = make_wordcloud(d)

			new_main_figure = generate_main_figure(d)

			return [
					f'data:image/png;base64,{wc1}',
					f'Users: {users_in_seg1:,}/{users_in_seg2:,} in Seg1/Seg2',
					f'data:image/png;base64,{wc2}',
					f'Reviews: {reviews_in_seg1:,}/{reviews_in_seg2:,} in Seg1/Seg2',
					new_main_figure
					]
		else:
			return [None, '', None, '', go.Figure()]

	app.run_server(debug=True)