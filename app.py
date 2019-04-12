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

import io
import base64


make_id = lambda pref, text: pref + '_' + text.lower().replace(' ','')
scale_marker = lambda fscore: 4 if abs(fscore) < 0.2 else 6 if 0.2 <= abs(fscore) <=0.4 else 8
filter_df = lambda df, seg1_dict, seg2_dict: (selector(df, seg1_dict), selector(df, seg2_dict))

def _make_seg_card(txt):
	
	return dbc.Card([
					dbc.CardHeader([
						dbc.Row([
						dbc.Button(id=make_id('badge', txt), 
							children=txt, color='info', size='sm', outline=True),
						dbc.Nav(dbc.NavItem(dbc.NavLink(id=make_id('nl', txt), 
														children='segment description', 
														disabled=True, href="#")))
						]), 
									]),
					dbc.Collapse(id=make_id('collapse', txt), 
						children=[
							dbc.Card([
								dbc.CardBody([
									dbc.Nav([
										dbc.DropdownMenu(label=it, 
												children=[dbc.DropdownMenuItem(id=make_id(make_id('mi', txt), c), 
															children=c) 
															for c in seg_options.get(it, None)], 
												bs_size="sm", 
												nav=True, 
												style={'font-size': 14}) for it in seg_options 
												
											]),
											])
									]),
							]),
					])

def calculate_scaled_fscores(df, min_freq, seg1_dict, seg2_dict):

	"""
	returns (users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d)
	"""

	print('calculating scaled f-scores...')

	d = pd.DataFrame()

	rev_seg1, rev_seg2 = filter_df(df, seg1_dict, seg2_dict)

	# collect some stats
	users_in_seg1, reviews_in_seg1 = len(set(rev_seg1['by_user'])), len(set(rev_seg1['review_id']))
	print(f'segment1: users: {users_in_seg1}, reviews: {reviews_in_seg1}')
	users_in_seg2, reviews_in_seg2 = len(set(rev_seg2['by_user'])), len(set(rev_seg2['review_id']))	
	print(f'segment2: users: {users_in_seg2}, reviews: {reviews_in_seg2}')

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

def generate_main_graph(df):

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


def make_wordcloud(df):

	wc1 = WordCloud(background_color='white', 
					width=600, height=300, max_words=300).generate_from_frequencies(df[['#seg1']].to_dict()['#seg1'])

	wc2 = WordCloud(background_color='white', 
					   width=600, height=300, max_words=300).generate_from_frequencies(df[['#seg2']].to_dict()['#seg2'])

	pngs = []

	# wc1.to_image()

	for i, wc in enumerate([wc1, wc2], 1):

		# plt.figure(figsize=(5,6))
		# fig = plt.imshow(wc, interpolation='bilinear')
		# fig.axes.get_xaxis().set_visible(False)
		# fig.axes.get_yaxis().set_visible(False)
		# plt.axis("off")

		# figfile = io.BytesIO()
		# plt.savefig(figfile, format='png')
		# figfile.seek(0) 
		# # getvalue() Return bytes containing the entire contents of the buffer
		# # figdata_png = base64.b64encode(figfile.getvalue()).decode()

		# pngs.append(figfile)

		# plt.savefig(f'assets/wc_seg_{i}.png', dpi=300, bbox_inches = 'tight', pad_inches = 0.0)

		pil_img = wc.to_image()
		img = io.BytesIO()
		pil_img.save(img, "PNG")
		img.seek(0)
		img_b64 = base64.b64encode(img.getvalue()).decode()

		pngs.append(img_b64)
	
	return pngs

if __name__ == '__main__':

	app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
	server = app.server

	data = pd.read_csv('data/data.csv', 
					parse_dates=['date_of_experience'],
					infer_datetime_format=True)

	attrs = json.load(open('data/attributes.json'))
	
	seg_options = {what: list(attrs[what].values()) for what in attrs}
	
	# pre-filter segment options
	for what in seg_options:
		opts_upd = [seg_options[what][0]]
		for opt in seg_options[what][1:]:
			df_ = data[data[what].apply(lambda x: str(opt) in str(x))]
			c = set(df_['review_id'])
			if c and len(c) > 99:
				opts_upd += [opt]
				seg_options[what] = opts_upd

	seg1_dict = {'age': '35-49'}
	seg2_dict = {'gender': 'f'}

	users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d = calculate_scaled_fscores(data, 4, seg1_dict, seg2_dict)
	
	if d.empty:
		raise Exception('empty data frames!')

	fig = generate_main_graph(d)

	navbar = dbc.NavbarSimple(brand='TripAdvisor Reviews for Melbourne', sticky='top')

	body = dbc.Container([
					dbc.Row([
						dbc.CardGroup([
						dbc.Col([dbc.Card([
											dbc.CardBody([dbc.CardImg(id='wc1', src=None)]),
											dbc.CardFooter(Span(id='span_s1', children='Segment 1 word cloud'))
											]), 
								 dbc.Card([
											dbc.CardBody([dbc.CardImg(id='wc2', src=None)]),
											dbc.CardFooter(Span(id='span_s2', children='Segment 2 word cloud'))
											]),
								 dbc.Card([
											dbc.CardBody([dbc.CardText(id='cardtext_s1', children='', 
											style={'font-size': 18, 'background-color': '#FAF4A0'}),
											dbc.CardText(id='cardtext_s2', children='', 
											style={'font-size': 18, 'background-color': '#1BF022'})
											])])
								 ], md=4),

						dbc.Col([dbc.Card([
											dbc.CardBody([dcc.Graph(id='main_graph', figure=fig)])
											])], md=8)
						])
							]),
					Br(),
					dbc.Row([
						Div([
							dbc.Row([
									dbc.Col([_make_seg_card('Segment 1')]),
									dbc.Col([_make_seg_card('Segment 2')]),
									])
							], style={'width': '92%'}),
						Div([
									dbc.Col([dbc.Button('OK', outline=True, color='success', id='update_everything')])
							], style={'width': '8%'})
							]),
					# hidden Div to keep selector description as a string
						Div(id='selector description', style={'display': 'none'})
						])

	app.layout = Div([navbar, body])
		
	# collapse callback
	@app.callback(
		[Output('collapse_segment1', 'is_open'), 
			Output('collapse_segment2', 'is_open')],
				[Input('badge_segment1', 'n_clicks'), 
					Input('badge_segment2', 'n_clicks')],
					[State('collapse_segment1', 'is_open'), 
						State('collapse_segment2', 'is_open')],
					)
	def toggle_collapse(*arg_lst):

		"""
		n_clicks is how many times clicked so far; callback triggers whenever n_clicks changes
		"""

		n_seg1, n_seg2, is_open_seg1, is_open_seg2 = arg_lst

		ctx = dash.callback_context
		
		changed = ctx.triggered[0]['prop_id']

		upd_open_states = [is_open_seg1, is_open_seg2]

		if changed and (changed == 'badge_segment1.n_clicks'):  
			upd_open_states[0] = (not is_open_seg1)
			return upd_open_states

		if changed and (changed == 'badge_segment2.n_clicks'):  
			upd_open_states[1] = (not is_open_seg2)
			return upd_open_states

		return upd_open_states

	@app.callback([Output('selector description', 'children'), 
					Output('nl_segment1', 'children'),
					Output('nl_segment2', 'children')],
					[Input(make_id(make_id('mi', 'Segment 1'), _), 'n_clicks_timestamp') 
							for what in seg_options 
							for _ in seg_options[what]] + \
					[Input(make_id(make_id('mi', 'Segment 2'), _), 'n_clicks_timestamp') 
							for what in seg_options 
							for _ in seg_options[what]]
		)
	def specify_segments(*lst):

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

		wrd1 = '/'.join([seg_options[what][max_idx1[what]] for what in seg_options])
		wrd2 = '/'.join([seg_options[what][max_idx2[what]] for what in seg_options])

		return (sel, wrd1, wrd2)

	# when clicked the OK button, update everything
	@app.callback(
		[Output('span_s1', 'children'), 
			Output('span_s2', 'children'),
			Output('wc1', 'src'), 
			Output('wc2', 'src')],
		[Input('update_everything', 'n_clicks')],
		[State('selector description', 'children')]
		)
	def update(n, dict_str):

		if (not n) or (not dict_str):
			return ['nothing', 'nothing', None, None]
		
		ctx = dash.callback_context

		if not ctx.triggered:
			pass
		else:
			changed = ctx.triggered[0]['prop_id']


		if (changed == 'update_everything.n_clicks') and dict_str:
			
			d1 = json.loads(dict_str)
		
			seg1_dict = d1['seg1']
			seg2_dict = d1['seg2']

			print('seg1_dict=', seg1_dict)
			print('seg2_dict=', seg2_dict)

			users_in_seg1, reviews_in_seg1, users_in_seg2, reviews_in_seg2, d = calculate_scaled_fscores(data, 4, seg1_dict, seg2_dict)
			
			if d.empty:
				raise Exception('empty data frames!')

			print('making word clouds')
			wc1, wc2 = make_wordcloud(d)
			print('done')

			return ['/'.join([str(v) for v in seg1_dict.values()]),
					'/'.join([str(v) for v in seg2_dict.values()]),
					f'data:image/png;base64,{wc1}',
					f'data:image/png;base64,{wc2}'
					]
		else:
			return ['nothing', 'nothing', None, None]


	app.run_server(debug=True)