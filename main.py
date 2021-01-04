# horses.csv

# rid - Race id;
# horseName - Horse name;
# age - Horse age;
# saddle - Saddle # where horse starts;
# decimalPrice - 1/Decimal price;
# isFav - Was horse favorite before start? Can be more then one fav in a race;
# trainerName - Trainer name;
# jockeyName - Jockey name;
# position - Finishing position, 40 if horse didn't finish;
# dist - how far a horse has finished from a winner;
# overWeight - Overweight code;
# outHandicap - Handicap;
# headGear - Head gear code;
# RPR - RP Rating;
# TR - Topspeed;
# OR - Official Rating
# father - Horse's Father name;
# mother - Horse's Mother name;
# gfather - Horse's Grandfather name;
# runners - Runners total;
# margin - Sum of decimalPrices for the race;
# course - Course of the race, country code in brackets, AW means All Weather, no brackets means UK;
# time - Time of the race in hh:mm format, London TZ;
# date - Date of the race;
# title - Title of the race;
# rclass - Race class;
# band - Band;
# ages - Ages allowed
# distance - Distance;
# condition - Surface condition;
# hurdles - Hurdles, their type and amount;
# prizes - Places prizes;
# winningTime - Best time shown;
# prize - Prizes total (sum of prizes column);
# metric - Distance in meters;
# countryCode - Country of the race;
# ncond - condition type (created from condition feature);
# class - class type (created from rclass feature).

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy import stats

#%%
## Grab the datasets and merge them.

df = pd.read_csv('data/horses_2020.csv')
dfr = pd.read_csv('data/races_2020.csv')
df_copy = df.copy()
dfn = df.merge(dfr, on='rid')
#%%
## Get info about horses
print(f'Shape\n\n{dfn.shape}')
print(f'Columns\n\n{dfn.columns}')
print(f'dtypes\n\n{dfn.dtypes}')
pd.set_option('display.max_columns', None)
print(f'Description\n\n{dfn.describe()}')
print(dfn.sample(n=1))
pd.set_option('display.max_columns', 5)

#%% Get info about races

# TODO: Join with races


# TODO: Missing data
# Total: 149513
# saddle: 6
# Overweight: 144613
# outHandicap: 148159
# RPR: 72446
# TR: 101769
# OR: 69976

#%%
# TODO: Remove irrelevant data
# positionL (is it there) - Doesn't exist
# weightSt - We have kilos
# weightLb - We have kilos
# res_win - This shows result
# res_place - This shows result
# distance - Like 7f - Given in metres in 'metric'
# band - No idea what it is.
# prize - Given in different currencies
# currency - See above
# prizes - List of prizes - Combined (added together) in prize
# winningTime - Not relevant
# rclass - Contains word 'Class' at start.  Repeated as number in 'class'
# condition - Repeated as number in 'cond'

dfn = dfn.drop(['weightSt',
                'weightLb',
                'res_win',
                'res_place',
                'distance',
                'band',
                'prize',
                'currency',
                'prizes',
                'winningTime',
                'rclass',
                'condition'], axis=1)

#%%
# TODO: Encode categorical data
# horseName
# trainerName
# jockeyName
# headGear
# father
# mother
# gfather
# ages - Like 3yo+
# hurdles - Like '8 hurdles'

# label_encoder = LabelEncoder()
print("headGear Values")
print(dfn.headGear.unique())
print(f'No of values = {len(dfn.headGear.unique())}')

print("horseName Values")
print(f'No of values = {len(dfn.horseName.unique())}')

print("trainerName Values")
print(f'No of values = {len(dfn.trainerName.unique())}')

print("jockeyName Values")
print(f'No of values = {len(dfn.jockeyName.unique())}')

print("mother Values")
print(f'No of values = {len(dfn.mother.unique())}')

print("father Values")
print(f'No of values = {len(dfn.father.unique())}')

print("ages Values")
print(dfn.ages.unique())
print(f'No of values = {len(dfn.ages.unique())}')

print("hurdles Values")
print(dfn.hurdles.unique())
print(f'No of values = {len(dfn.hurdles.unique())}')

#%%
## Split hurdles column by fences and hurdles

## Create boolean series showing whether hurdles column contains the word
## 'hurdle'.
hurdles = dfn['hurdles'].str.contains('hurdle')

# Create new series containing 0 if no hurdle or grab the number if available.
# hurdle_count = pd.Series(dtype='int64')
hurdle_count = np.array([], dtype=int)

for count, hurdle in enumerate(hurdles):
    if hurdle and isinstance(dfn['hurdles'][count], str):
        ## Grab the number - i.e. first 'word' in string
       # and convert to int
        number = int(dfn['hurdles'][count].split(' ')[0])
    else:
        number = 0
    hurdle_count = np.append(hurdle_count, number)

    # hurdle_count = hurdle_count.set_value(count, number)
    if count % 1000 == 0:
        print(f'hurdles processed = {count}')

## Add Series to our dataset
dfn['hurdles_cnt'] = hurdle_count.tolist()

#%%
## Create boolean series showing whether hurdles column contains the word
## 'fence'.
fences = dfn['hurdles'].str.contains('fences')

# Create new series containing 0 if no hurdle or grab the number if available.
# hurdle_count = pd.Series(dtype='int64')
fence_count = np.array([], dtype=int)

for count, fence in enumerate(fences):
    if fence and isinstance(dfn['hurdles'][count], str):
        ## Grab the number - i.e. first 'word' in string
        # and convert to int
        number = int(dfn['hurdles'][count].split(' ')[0])
    else:
        number = 0
    fence_count = np.append(fence_count, number)

    # fence_count = fence_count.set_value(count, number)
    if count % 1000 == 0:
        print(f'fences processed = {count}')

## Add Series to our dataset
dfn['fences_cnt'] = fence_count.tolist()

#%%
dfn = dfn.drop(['hurdles'], axis=1)

#%%
# TODO: Remove all handicap rows. Will try with first and then without
# These are horses that run in handicap
# races and have weight added.  The skill here is in guessing if the
# handicap is too much or too little.

#%%
# TrainerName contains some numbers.  Let's find out what they are...

numberdf = dfn[not isinstance(dfn['trainerName'], str)]

