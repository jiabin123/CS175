import os
import json

path = './data/'
x = []

i = 0
for f_name in os.listdir('./data'):
    if f_name.endswith('.txt'):
        file = open((path + f_name), 'r')
        x.append({
            "rating" : int(f_name[-5]),
            "review" : file.read()
        })
        i += 1
        file.close()

with open("reviews.json", 'w') as f:
    json.dump(x, f, indent=4)

f.close()
