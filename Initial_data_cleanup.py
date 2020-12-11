import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#plots the price against every other feature in the dataframe and
def plotFeatureAgainstY(df,yvalues,ylabel):
    
    count = 1
   
    for (columnName, columnData) in df.iteritems():
        plt.figure(count)
        plt.xlabel(columnName)
        plt.ylabel(ylabel)
        plt.scatter(columnData, yvalues, s = 1)
        count += 1
        
    return count
        
    
    
# Change this to the path of the folder where the csv is
path = r"C:\Users\Laurentiu\OneDrive - City, University of London\IN3062 - Introduction to Artificial Intelligence\Coursework"
file_path = os.path.join(path,"diamonds.csv")
diamonds_df = pd.read_csv(file_path)


# Filters out all of the columns that contain a 0
diamonds_df = diamonds_df.loc[(diamonds_df != 0).all(axis=1), :]

# There are two diamonds that have very odd z values compared to the rest so this negates this and makes the graph more readable
diamonds_df = diamonds_df[diamonds_df["z"] < 8]
diamonds_df = diamonds_df[diamonds_df["z"] > 2]

# Same issue as above for y except there is only one value
diamonds_df = diamonds_df[diamonds_df["y"] < 10]

# Same for table
diamonds_df = diamonds_df[diamonds_df["table"] < 85]

print(diamonds_df.groupby('clarity')['price'].mean().sort_values())
print(diamonds_df.groupby('cut')['price'].mean().sort_values())
print(diamonds_df.groupby('color')['price'].mean().sort_values())

#Mapping for replacing the non-numerical values
cut_mapping = {'Ideal':1,
                'Good':2,
                'Very Good':3,
                'Fair':4,
                'Premium':5}

color_mapping = {'E':1, 
                  'D':2, 
                  'F':3, 
                  'G':4, 
                  'H':5, 
                  'I':6, 
                  'J':7,}

clarity_mapping = {'VVS1':1, 
                    'IF':2, 
                    'VVS2':3, 
                    'VS1':4, 
                    'I1':5,
                    'VS2':6, 
                    'SI1':7, 
                    'SI2':8}

diamonds_df = diamonds_df.replace({'cut':cut_mapping})
diamonds_df = diamonds_df.replace({'color':color_mapping})
diamonds_df = diamonds_df.replace({'clarity':clarity_mapping})

y = diamonds_df['price']

df_no_price = diamonds_df.drop(['price'],1)

print('\n')

figurecount = plotFeatureAgainstY(df_no_price, y, "price ($)")



plt.figure(figurecount + 3)
sns.heatmap(diamonds_df.corr(method="pearson"))




