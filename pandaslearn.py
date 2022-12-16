import pandas as pd

#Creating a car price series with a dictionary
car_price_dict = {'Swift':  700000,
                       'Jazz' :  800000,
                       'Civic' : 1600000,
                       'Altis' : 1800000,
                       'Gallardo': 30000000
                      }
car_price = pd.Series(car_price_dict)
# Creating the car manufacturer series with a dictionary
car_man_dict = {'Swift' : 'Maruti',
                  'Jazz'   : 'Honda',
                  'Civic'  : 'Honda',
                  'Altis'  : 'Toyota',
                   'Gallardo' : 'Lamborghini'}
car_man = pd.Series(car_man_dict)
print(car_price)
print(car_man)
