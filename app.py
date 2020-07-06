# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:24:15 2020

@author: ss21930
"""
#%% Import Libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import dash_table

import pandas as pd
import datetime as dt
import plotly.graph_objs as go

#%% Read Data (Need to run the forecasting pipeline before this)

date = str(dt.date.today() + dt.timedelta(days=-1))
file_name = './Forecast_Data/forecast_' + date + '.csv'
full_data = pd.read_csv(file_name)

prediction_dates = list(full_data['date'])[-7:]
training_dates = list(full_data['date'])[:-7]

states = [col.split('_')[0] for col in full_data.columns if len(col.split('_'))==2]

dropdown_states = [{'label':state, 'value':state} for state in states]

#%% Functions 

# Function to Plot Figure
def plot_figure(state, df=full_data):
    fig = go.Figure()

    actual = df[df['date'].isin(training_dates)][state].values
    fit = df[df['date'].isin(training_dates)][state+'_fit'].values
    forecasted = df[df['date'].isin(prediction_dates)][state].values
    
    # Add actual data
    fig.add_trace(go.Scatter(x=training_dates, y=actual, mode='markers', name='Actual'))
    # Add fit data
    fig.add_trace(go.Scatter(x=training_dates, y=fit, mode='lines', name='Fit'))
    # Add forecasted data
    fig.add_trace(go.Scatter(x=prediction_dates, y=forecasted, mode='markers', name='Forecast'))
    
    fig.update_layout(template='plotly_dark',
                      title=dict(text =state+' Cumulative Deaths',
                                 font =dict(family='sans-serif', size=30, color='#B3B3B3')),
                      title_x=0.1,
                      xaxis_title='Date',
                      yaxis_title='Cumulative Deaths',
                      height=550)

    return fig


def get_table(state, df=full_data, prediction_dates=prediction_dates):
    state_df = df[df['date'].isin(prediction_dates)][['date', state]].reset_index(drop=True)
    fig = go.Figure(data=[go.Table(
                            columnwidth=[85,150],
                            header=dict(values=['Date', 'Forecasted Values'],
                                        line_color='#808080',
                                        fill_color='#111111',
                                        align='center',
                                        font=dict(color='#B3B3B3', size=20, family='sans-serif'),
                                        height=40),
                            cells=dict(values=[state_df[col] for col in state_df.columns],
                                       font=dict(color='#B3B3B3', size=18, family='sans-serif'),
                                       fill_color='#111111',
                                       line_color='#808080',
                                       align='center',
                                       height=30))
                         ])
    
    fig.update_layout(template='plotly_dark')
    
    return fig

#%% Page Layout
    
app = dash.Dash(__name__,\
                meta_tags=[{"name": "viewport","content": "width=device-width,initial-scale=1" \
                                                                                }])


colors = {'background': '#E3E2E2', 'text': '#FFFFFF'}

# markdown_text = "## Statewise Cumulative Death Forecast"

app.layout = html.Div(
                [html.Div([html.H1('Statewise Cumulative Death Forecast', style={'textAlign':'center', 'padding-top':'10px', 'color':'#B3B3B3'})], 
                          style={'backgroundColor':'#0D0D0D', 'height':'90px'}),
                 html.Div(
                    [html.Div(
                        [html.H6("Select State", style={'color':'#B3B3B3'}),
                         dcc.Dropdown(id='state-dropdown',options = dropdown_states,value='California',clearable=False, 
                                      style={'width':'100%', 'backgroundColor':'#B3B3B3'}) 
                        ], style={'display':'inline-block', 'height':'1000px', 'width':'12%', 'verticalAlign':'top', 'padding-left':'20px',\
                                  'padding-top':'5%','padding-right':'15px', 'backgroundColor':'#0F0F0F'}),
                     html.Div(
                        dcc.Graph(id='plot', style={'width':'100%', 'align':'center', 'color':'black'}),\
                            style={'display':'inline-block', 'height':'1000px', 'width':'50%', 'backgroundColor':'#111111', 'padding-left':'50px'}),                                    
                     html.Div(
                        dcc.Graph(id='table', style={'width':'100%', 'align':'right'}), \
                            style={'display':'inline-block', 'height':'1000px', 'width':'30%', 'backgroundColor':'#111111', \
                                   'padding-top':'40px'})
                     ])
                ], style={'backgroundColor':'#111111'})

#%% Add interactivity 
                                                                                              
@app.callback(
    Output('plot', 'figure'),
    [Input('state-dropdown', 'value')])
def plot_fig(selected_state):
    fig = plot_figure(selected_state)
    return fig


@app.callback(
    [Output('table', 'figure')],
    [Input('state-dropdown', 'value')])
def get_datatable(selected_state):
    fig = get_table(selected_state)
    return [fig]
    

#%% Run App
    
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True, dev_tools_hot_reload=True)