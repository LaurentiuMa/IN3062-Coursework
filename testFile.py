import os
import pandas as pd
from cleanedData import Cleaner

cleaner = Cleaner(r"C:\Users\Laurentiu\OneDrive - City, University of London\IN3062 - Introduction to Artificial Intelligence\Coursework")

print(cleaner.getDataFrame())