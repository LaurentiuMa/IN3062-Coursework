import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Select your own path to the dataset
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

# Show the mean for each categorical feature
print(diamonds_df.groupby('color')['price'].mean().sort_values())
print(diamonds_df.groupby('clarity')['price'].mean().sort_values())
print(diamonds_df.groupby('cut')['price'].mean().sort_values())

#Mapping for replacing the non-numerical values
cut_mapping = {'Ideal':0,
                'Good':1,
                'Very Good':2,
                'Fair':3,
                'Premium':4}

color_mapping = {'E':0, 
                  'D':1, 
                  'F':2, 
                  'G':3, 
                  'H':4, 
                  'I':5, 
                  'J':6,}

clarity_mapping = {'VVS1':0, 
                    'IF':1, 
                    'VVS2':2, 
                    'VS1':3, 
                    'I1':4,
                    'VS2':5, 
                    'SI1':6, 
                    'SI2':7}
diamonds_df = diamonds_df.replace({'cut':cut_mapping})
diamonds_df = diamonds_df.replace({'color':color_mapping})
diamonds_df = diamonds_df.replace({'clarity':clarity_mapping})

# Plot the correlation matrix
plt.figure(100)
print(sns.heatmap(diamonds_df.corr(method="pearson")))

#print the corelation matrix
print(diamonds_df.corr(method="pearson"))

# Series of plots to understand the data
plt.figure(0)
plt.xlabel("Price $")
plt.ylabel("Count")
plt.hist(diamonds_df['price'].values.tolist(), bins = 500)

plt.figure(1)
plt.xticks(range(7))
plt.xlabel("Color")
plt.ylabel("Count")
plt.hist(diamonds_df['color'].values.tolist(), bins=range(8), align='left', rwidth=0.9)

plt.figure(2)
plt.xticks(range(5))
plt.xlabel("Cut")
plt.ylabel("Count")
plt.hist(diamonds_df['cut'].values.tolist(), bins=range(6), align='left', rwidth=0.9)

plt.figure(3)
plt.xticks(range(8))
plt.xlabel("Clarity")
plt.ylabel("Count")
plt.hist(diamonds_df['clarity'].values.tolist(),bins=range(9), align='left', rwidth=0.9)

plt.figure(4)
plt.xlabel("log Price $")
plt.ylabel("Count")
plt.hist(np.log(diamonds_df['price'].values.tolist()), bins = 100)

# Plot the two distributions (log and non-log of price) on the same subplot
fig, axs = plt.subplots(1,2)
fig.suptitle('Normalising the diamond prices')
axs[0].spines["right"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[0].set_ylabel("Count")
axs[0].set_xlabel("Price $")
axs[0].hist(diamonds_df['price'].values.tolist(), bins = 200)
axs[1].set_xlabel("log(Price)")
axs[1].hist(np.log(diamonds_df['price'].values.tolist()), bins = 200)
