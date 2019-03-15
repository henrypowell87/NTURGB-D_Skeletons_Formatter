# NTURGB-D_Skeletons_Formatter
## Python code for formatting the NTU RGB+D Dataset for use in a RNN.

Python script to format the NTURGB+D skeletons dataset for Recurrent Neural Network applications.\
The script iterates through a directory of .skeleton files (determined by the user), drops unnecessary rows, makes \
each 25 * 12 frame of data into a 1-D input array for an RNN. Each action is thus stored in the data set as a timeseries \
of 1-D arrays of length 25 * 12 = 300. Each action is therefore a 300*len(action) array where len(action) = the number of 
frames for each action.

Find information about the dataset here: https://github.com/shahroudy/NTURGB-D \
For help using this code: h.powell.2@research.gla.ac.uk
