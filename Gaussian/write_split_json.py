import json

# SAVE_PATH = "/sdf1/yx/dataset/scannet++/data/f6659a3107/dslr/split.json" #1ada7a0617 5748ce6f01
SAVE_PATH = "/sde1/yx/dataset/replica/habitat/apartment_2/split.json"

setting = {}
# setting['train'] = list(range(0,343,1))
# setting['test'] = list(range(343,360,1))
# setting['train'] = list(range(0,1000,1))
# setting['test'] = list(range(1000,1011,1))
setting['train'] = list(range(0,750,1))
setting['test'] = list(range(0,1,1))

with open(SAVE_PATH, 'w') as f:
    json.dump(setting, f)