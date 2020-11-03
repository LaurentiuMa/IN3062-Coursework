# Replace the values for cut, clarity and colour from letters to numbers as per the document

import os
import pandas as pd
import matplotlib.pyplot as plt

def plotFeatureAgainstY(df,yvalues,ylabel):
    
    count = 1
    
    for (columnName, columnData) in df.iteritems():
        plt.figure(count)
        plt.xlabel(columnName)
        plt.ylabel(ylabel)
        plt.scatter(columnData, yvalues, s = 1)
        count += 1
    

path = r"C:\Users\Laurentiu\OneDrive - City, University of London\IN3062 - Introduction to Artificial Intelligence\Coursework"
file_path = os.path.join(path,"diamonds.csv")
diamonds_df = pd.read_csv(file_path)

# uncomment this if you do not want the columns with non-numerical values
# diamonds_df = diamonds_df.select_dtypes(include=['int', 'float'])

# Filters out all of the columns that contain a 0
diamonds_df = diamonds_df.loc[(diamonds_df != 0).all(axis=1), :]


cut_mapping = {'Ideal':5,
            'Premium':4,
            'Very Good':3,
            'Good':2,
            'Fair':1}

color_mapping = {'D':'7', 
                 'E':'6', 
                 'F':'5', 
                 'G':'4', 
                 'H':'3', 
                 'I':'2', 
                 'J':'1',}

clarity_mapping = {'IF':'8', 
                   'VVS1':'7', 
                   'VVS2':'6', 
                   'VS1':'5', 
                   'VS2':'4', 
                   'SI1':'3', 
                   'SI2':'2', 
                   'I1':'1'}

diamonds_df = diamonds_df.replace({'cut':cut_mapping})
diamonds_df = diamonds_df.replace({'color':color_mapping})
diamonds_df = diamonds_df.replace({'clarity':clarity_mapping})

y = diamonds_df['price']
df_no_price = diamonds_df.drop('price',1)

print('\n')

plotFeatureAgainstY(df_no_price, y, "price ($)")