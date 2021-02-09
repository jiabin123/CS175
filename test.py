import os
import json

path = './data/'
x = []

for f_name in os.listdir('./data'):
    if f_name.endswith('.txt'):
        file = open((path + f_name), 'r')
        x.append({
            "rating" : int(f_name[-5]),
            "review" : file.read()
        })
        file.close()

with open("reviews.json", 'w') as f:
    json.dump(x, f, indent=4)

f.close()
