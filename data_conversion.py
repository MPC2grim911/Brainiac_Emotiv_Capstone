import pandas as pd
import scipy.io
import numpy as np
import csv

for x in range(1, 35):
    # loading the data mat file
    file = scipy.io.loadmat('EEG DATA/eeg_record' + str(x) + '.mat')  # change filename accordingly
    data = pd.DataFrame.from_dict(file["o"]["data"][0, 0])

    data = data.to_numpy()  # transform to numpy array
    new_data = data[:, 3:17]  # trancating the data of the sensors only from the array
    print(new_data)

    # saving the result in csv file
    np.savetxt('EEG Data CSV/eeg_record' + str(x) + '.csv', new_data, delimiter=",", fmt='%1g')
