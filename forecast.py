# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:54:53 2020

@author: ss21930
"""

import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append('.')

import pandas as pd
import numpy as np
import datetime as dt

import curvefit
from curvefit.core.functions import ln_gaussian_cdf, gaussian_cdf

np.random.seed(1234)

#%%
# Link function for beta
def identity_fun(x) :
    return x

# link function used for alpha, p
def exp_fun(x) :
    return np.exp(x)

#%%
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
cases_us = pd.read_csv(url)

cases_us = cases_us[cases_us['iso3']=='USA'].reset_index(drop=True)
drop_cols = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
cases_us = cases_us.drop(drop_cols, axis=1)

aggfunc = {}
for col in cases_us.columns[1:]:
    aggfunc[col] = 'sum'

cases_us = cases_us.groupby('Province_State', as_index=False).agg(aggfunc)
cases_us = pd.melt(cases_us, id_vars =['Province_State'], value_vars =cases_us.columns[2:])
cases_us['date'] = pd.to_datetime(cases_us['variable'])
cases_us = cases_us.rename(columns = {'value':'death', 'Province_State':'state'}).drop('variable', axis=1)
cases_us = cases_us.sort_values(['state', 'date'])

#%%
url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
a = pd.read_csv(url)

a['date'] = pd.to_datetime(a['date'])
ny_data = a[a['state']=='New York'].reset_index(drop=True)
ny_data = ny_data[['state', 'deaths', 'date']].rename(columns={'deaths':'death'})

cases_us = pd.concat([cases_us[cases_us['state']!='New York'], ny_data]).sort_values(['state', 'date']).reset_index(drop=True)

#%%
consider_states = {'New York','California', 'New Jersey', 'Massachusetts', 'Connecticut', 'Rhode Island', \
                   'Maryland', 'Illinois', 'Michigan'}
remove_states = {'Alaska', 'Hawaii', 'Idaho', 'Montana', 'Nebraska', 'South Dakota', 'Vermont','West Virginia', 'Wyoming', 'Diamond Princess', 'Grand Princess'}
    
# cases_us = cases_us[cases_us['state'].isin(consider_states)].reset_index(drop=True)
cases_us = cases_us[~cases_us['state'].isin(remove_states)].reset_index(drop=True)
cases_us = cases_us[cases_us['death'].notnull()]

min_date = cases_us.groupby('state', as_index=False)['date'].min()['date'].max()

state_pop = pd.read_csv("Data/nst-est2019-01.csv")
state_pop['State'] = state_pop['State'].apply(lambda state: state[1:])
state_pop['Population'] = state_pop['Population'].astype('int')

cases_us = pd.merge(cases_us, state_pop, left_on='state', right_on='State', how='left')

cases_us['death_rate'] = cases_us['death']/cases_us['Population']
cases_us['log_death_rate'] = np.log(cases_us['death']/cases_us['Population'])

cases_us = cases_us.replace(to_replace=-np.inf, value=100)

cases_us['threshold'] = cases_us['log_death_rate'].apply(lambda x: int(x>=-15 and x<100))
day_1_df = cases_us[cases_us['threshold']==1].groupby('state', as_index=False)['date'].min().rename(columns = {'date':'day_1'})
cases_us = pd.merge(cases_us, day_1_df, on='state', how='left')

cases_us = cases_us[cases_us['date']>=cases_us['day_1']]

cases_us = cases_us.sort_values(['state', 'date']).reset_index(drop=True)
cases_us['day'] = (cases_us['date'] - cases_us['day_1']).apply(lambda x: x.days)+1
cases_us['constant_one'] = 1

#%%
cases_us_backup = cases_us.copy(deep=True)

most_recent_date = cases_us_backup['date'].max()
fit_start_date = most_recent_date - dt.timedelta(days=80)

final_df = cases_us_backup[cases_us_backup['date']>=fit_start_date].reset_index(drop=True)

# number of parameters, fixed effects, random effects
num_params   = 3
num_fe       = 3
n_group = final_df['state'].nunique()
num_re       = num_fe * n_group

target = 'log_death_rate' #'log_death_rate', 'death_rate'
model_fun = {'death_rate':gaussian_cdf, 'log_death_rate':ln_gaussian_cdf}


# ------------------------------------------------------------------------
# curve_model
col_t        = 'day'
col_obs      = target
col_covs     = [['constant_one'], ['constant_one'], ['constant_one']]
col_group    = 'state'
param_names  = [ 'alpha', 'beta', 'p']
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = num_fe * [ identity_fun ]
fun          = model_fun[target]
#
curve_model = curvefit.core.model.CurveModel(
    df= final_df,
    col_t = col_t,
    col_obs = col_obs,
    col_covs = col_covs,
    col_group = col_group,
    param_names = param_names,
    link_fun = link_fun,
    var_link_fun = var_link_fun,
    fun = fun,
)

# -------------------------------------------------------------------------
# fit_params
#
fe_init   = np.array( [ -2, 10, -1 ] )
re_init   = np.zeros( num_re )
fe_bounds = [ [-np.inf, 0], [1,100], [-100,0] ]
re_bounds = [ [-np.inf, np.inf] ] * num_fe
options={
    'disp'    : 0,
    'maxiter' : 1000,
    'ftol'    : 1e-8,
    'gtol'    : 1e-8,
}
#
curve_model.fit_params(
    fe_init,
    re_init,
    fe_bounds,
    re_bounds,
    options=options
)

fe_estimate = curve_model.result.x[0 : num_fe]
re_estimate = curve_model.result.x[num_fe :].reshape(n_group, num_fe)

#%%
start_date = final_df.date.min()
max_date = (final_df.groupby('state')['date'].max()).min()
final_df = final_df[final_df['date'] <= max_date].reset_index(drop=True)

prediction_dates = [max_date + dt.timedelta(days = i) for i in range(1,8)]

train_period = [start_date + dt.timedelta(days = i) for i in range((max_date - start_date).days + 1)]
total_period = train_period + prediction_dates

y_pred, y_train, y_fit  = {}, {}, {}

for i, state in enumerate(final_df['state'].unique()):
    
    max_day = final_df[final_df['state']==state]['day'].max()
    prediction_window = [max_day + i for i in range(1,8)]
    
    actual_death_rates = (final_df[final_df['state']==state][target]).values
    fit_death_rates = curve_model.predict(t=final_df[final_df['state']==state]['day'], group_name=state)
    forecasted_death_rates = curve_model.predict(t=np.array(prediction_window), group_name=state)
    
    y_train[state] = final_df[final_df['state']==state][target].values
    y_pred[state] = forecasted_death_rates
    y_fit[state] = fit_death_rates

#%%
death_pred = {}
for i, state in enumerate(final_df['state'].unique()):
    death_pred[state] = np.exp(y_pred[state])*cases_us_backup[cases_us_backup['state']==state]['Population'].values[0]
    
death_training = {}
for i, state in enumerate(final_df['state'].unique()):
    death_training[state] = np.exp(y_train[state])*cases_us_backup[cases_us_backup['state']==state]['Population'].values[0]
    
death_fitting = {}
for i, state in enumerate(final_df['state'].unique()):
    death_fitting[state] = np.exp(y_fit[state])*cases_us_backup[cases_us_backup['state']==state]['Population'].values[0]
    
death_predictions_df = pd.DataFrame.from_dict(death_pred)
death_training_df = pd.DataFrame.from_dict(death_training)
death_fitting_df = pd.DataFrame.from_dict(death_fitting)

death_training_df['date'] = train_period
death_fitting_df['date'] = train_period
death_predictions_df['date'] = prediction_dates

for col in set(death_predictions_df.columns).difference({'date'}):
    death_predictions_df[col] = death_predictions_df[col].apply(lambda x: round(x))
    death_fitting_df[col] = death_fitting_df[col].apply(lambda x: round(x))
    
full_df = pd.concat([death_training_df, death_predictions_df]).reset_index(drop=True)

death_fitting_df.columns = [col+'_fit' if col!='date' else col for col in death_fitting_df.columns]

full_df = pd.merge(full_df, death_fitting_df, on='date', how='left')

#%%
date = str(min(prediction_dates).date())
file_name = './Forecast_Data/forecast_'+date+'.csv'
print(file_name)

full_df.to_csv(file_name, index=False)