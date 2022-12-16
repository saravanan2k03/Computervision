import pandas as pd
car_price_dict = {'Swift':  700000,
                       'Jazz' :  800000,
                       'Civic' : 1600000,
                       'Altis' : 1800000,
                       'Gallardo': 30000000
                      }
car_price = pd.Series(car_price_dict)
car_man_dict = {'Swift' : 'Maruti',
                  'Jazz'   : 'Honda',
                  'Civic'  : 'Honda',
                  'Altis'  : 'Toyota',
                   'Gallardo' : 'Lamborghini'}
car_man = pd.Series(car_man_dict)
cars = pd.DataFrame({'price':car_price,'Manufacturer':car_man})
print(cars)