import pandas as pd

#Using dictionary to create a series
car_price_dict = {'Swift':  700000,
                       'Jazz' :  800000,
                       'Civic' : 1600000,
                       'Altis' : 1800000,
                       'Gallardo': 30000000
                      }
car_price = pd.Series(car_price_dict)
print(car_price)
#Creating a DataFrame from car_price Series
print(pd.DataFrame(car_price, columns=['Car Price']))
