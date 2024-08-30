# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:11:45 2024

@author: SOSA
"""
import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/SOSA/OneDrive/clgggg/Desktop/Parkinsons Disease Detection/trained_model.joblib', 'rb'))

input_data = (122.4,148.65,113.819,0.00968,0.00008,0.00465,0.00696,0.01394,0.06134,0.626,0.03134,0.04518,0.04368,0.09403,0.01929,19.085,1,0.458359,0.819521,-4.075192,0.33559,2.486855) # Removed one element to match 22 features

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Person does not have Parkinsons Disease')
else:
  print('The person have Parkinson Disease ')