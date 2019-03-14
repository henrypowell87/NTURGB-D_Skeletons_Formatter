# Python script to format the NTURGB+D skeletons dataset for Recurrent Neural Network applications.
# The script iterates through a directory of .skeleton files (determined by the user), drops unnecessary rows, makes
# each 25*12 frame of data into a 1-D input array for an RNN. Each action is thus stored in the data set as a time
# series of 1-D arrays of length 25x12=300. Each action is therefore a 300*len(action) array where len(action) = the
# number of frames for each action.

from __future__ import print_function

import csv
import numpy as np
import pandas as pd

from pathlib import Path

# Insert path to .skeleton files here
path = ''
directory = Path(path)

# Store files as list to be iterated through
files = [p for p in directory.iterdir() if p.is_file()]

# You may have a .CD file hidden in this folder. This drops this from [files] so that the code doesn't run over it.
files.pop(0)

# Empty list where we will store the whole formatted data set.
data_set = []

# This variable tracks how many files have been formatted and added to the new data set
progress = 0

# Iteration loop which formats the .skeleton files.
for file in files:
        data_list = []
        data = pd.read_csv(file, header=None)

        # This block filters out the irrelevant rows. I.e. everything that not the 25x12 frame data.
        data['length'] = data[0].apply(lambda x: len(x))
        cond = data['length'] > 10
        data = data[cond]
        data = data.reset_index(drop=True)
        data = data[data.index % 26 != 0]
        data = data.drop(columns=['length'])
        data = data.reset_index(drop=True)

        # Split the data into a matrix so each value is in it's own cell so we
        # now have a 25x12 pd.Dataframe for each frame of the skeleton data.
        # We then iterate through the data row by row and add it to the data list.
        data = data[0].str.split(" ", expand=True)
        for i in range(len(data)):
            data_list.append(data.iloc[i])

        # We now convert the data_list to a numpy array, flatten it into 1 dimension and divide by the number of
        # joints per frame (25) to give an array of arrays containing each frame of skeletal data.
        data_list = np.array(data_list, dtype=np.float32)
        division = len(data_list)/25
        data_list = data_list.flatten()
        data_list = np.array(np.split(data_list, division))
        data_list = data_list.tolist()
        data_set.append([data_list])

        # This block tracks the progress of the formatting and prints the progress to the terminal.
        # The "end='\r' argument has only been tested on mac and may not work on other platforms.
        # If you don't see any progress on your terminal delete this.
        progress += 1
        perc = progress/(56881/100)
        print('Samples Processed: {0}/56,881 - - - Percentage Complete = {1:.2f}%'.format(progress, perc), end='\r')

# Convert the whole data_set to an numpy array.
data_set = np.array(data_set)

# Save the array to a new CSV named "skeleton_data_formatted.csv"
with open("skeleton_data_formatted.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data_set)
