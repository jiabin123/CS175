import os
import json

path = './data/'
x = {}

i = 0
for f_name in os.listdir('./data'):
    if f_name.endswith('.txt'):
        file = open((path + f_name), 'r')
        x[i] = [f_name[-5], file.read()]
        i += 1
        file.close()

y = json.dumps(x, indent=4)
with open("reviews.json", 'w', encoding='utf-8') as f:
    f.write(y)

f.close()
