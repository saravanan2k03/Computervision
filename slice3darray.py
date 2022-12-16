import numpy as np
#Creating a 2D array consisting car names, horsepower and acceleration
# car_names = ['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino']
# horsepower = [130, 165, 150, 150, 140]
# acceleration = [18, 15, 18, 16, 17]
# car_hp_acc_arr = np.array([car_names, horsepower, acceleration])
# #Accessing name and horsepower
# print(car_hp_acc_arr[0:2])


#creating a list of 5 horsepower values
horsepower = [130, 165, 150, 150, 140, 170]
#creating a numpy array from horsepower list
horsepower_arr = np.array(horsepower)
print("Index of Minimum horsepower: ", np.argmin(horsepower_arr))
print("Index of Maximum horsepower: ", np.argmax(horsepower_arr))



