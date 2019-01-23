#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'CACLA code'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Explore Training Curves
# 
# A basic notebook to generate plots from training logs. See **`Logger`** class in **`utils.py`**.
# 
# Python Notebook by Patrick Coady: [Learning Artificial Intelligence](https://learningai.io/)
#%% [markdown]
# ## Summary
# 
# Hypothesis, conditions, and results explored in this notebook ...

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"] = (10, 5)
plt.style.use('ggplot')


#%%
def df_plot(dfs, x, ys, ylim=None, legend_loc='best'):
    """ Plot y vs. x curves from pandas dataframe(s)
    
    Args:
        dfs: list of pandas dataframes
        x: str column label for x variable
        y: list of str column labels for y variable(s)
        ylim: tuple to override automatic y-axis limits
        legend_loc: str to override automatic legend placement:
            'upper left', 'lower left', 'lower right' , 'right' ,
            'center left', 'center right', 'lower center',
            'upper center', and 'center'
    """
    if ylim is not None:
        plt.ylim(ylim)
    for df, name in dfs:
        # print(name)
        # name = name.split('_')[1]
        for y in ys:
            plt.plot(df[x], df[y], linewidth=2, label=name + ' ' + y.replace('_', ''))
    plt.xlabel(x.replace('_', ''))
    leg = plt.legend(loc=legend_loc)
    for t in leg.get_texts():
        plt.setp(t, color="black")
    plt.show()


#%%
# ENTER LIST OF LOG FILENAMES HERE:
filepaths = ['C:/Users/Jonathan/Documents/School/Project_Farkas/CACLA code/log-files/V-rep_AL5D/Jan-15_13.48.07_best_V-rep/log.csv',
             'C:/Users/Jonathan/Documents/School/Project_Farkas/CACLA code/log-files/V-rep_AL5D_no_sim/Jan-22_15.45.30_best_sim/log.csv']
dataframes = []
names = []
for filepath in filepaths:
    names.append(filepath.split('/')[-2])
    dataframes.append(pd.read_csv(filepath))
data = list(zip(dataframes, names))

#%% [markdown]
# # Plots

#%%

df_plot(data, '_episode', ['_mean_reward'])
df_plot(data, '_episode', ['mean_policy_loss'])
df_plot(data, '_episode', ['mean_value_loss'])
df_plot(data, '_episode', ['Steps'])


#%%
df_plot(data, "_episode", ["mean_last_reward"])
df_plot(data, "_episode", ["_min_last_reward", "_max_last_reward"])
df_plot(data, "_episode", ["_std_last_reward"])


#%%
df_plot(data, '_episode', ['policy_lr', "value_lr"])
df_plot(data, '_episode', ["exploration_factor"])


#%%
df_plot(data, '_episode', ['_mean_observation'])
df_plot(data, "_episode", ['_min_observation', '_max_observation'])


#%%
df_plot(data, '_episode', ['_mean_action', '_min_action', '_max_action'])
df_plot(data, '_episode', ['_std_action'])


#%%



