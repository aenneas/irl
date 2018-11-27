import numpy as np

def current_price(x):
    return ((x - 3.66) * 5500) + ((x - 3.74) * 25000) + ((x - 3.77) * 19500) + ((x - 4.65) * 5400) + ((x-6.12) * 6500) + ((x-6.55) * 10000)

print(current_price(5.30))