# NTURGB-D_Skeletons_Formatter
## Python code for formatting the NTU RGB+D Dataset for use in a RNN.

Python script for formatting the NTU RGB+D Skeletons data set into a format suitable for most LSTM RNNs. The aim is to
take each .skeletons files and compress it into a 3D numpy array with [samples, time steps, features] as its dimensions.
The final data set will thus be a [56,881, max(len(all_files)) =  600, 3000] numpy array. The data has been left
normal (i.e. not normalized) for the sake of flexibility.

Find information about the dataset here: https://github.com/shahroudy/NTURGB-D \
For help using this code: h.powell.2@research.gla.ac.uk
