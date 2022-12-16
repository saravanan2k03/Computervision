import numpy as np

#Creating a 2D array consisting car names, horsepower and acceleration
# car_names = ['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino']
# horsepower = [130, 165, 150, 150, 140]
# acceleration = [18, 15, 18, 16, 17]
# car_hp_acc_arr = np.array([car_names, horsepower, acceleration])
# #Accessing name and horsepower
# print(car_hp_acc_arr[0:2])


#Creating a 2D array consisting car names, horsepower and acceleration
car_names = ['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino']
horsepower = [130, 165, 150, 150, 140]
acceleration = [18, 15, 18, 16, 17]
car_hp_acc_arr = np.array([car_names, horsepower, acceleration])
#Accessing name and horsepower of last two cars
print(car_hp_acc_arr[0:2, 3:5])

