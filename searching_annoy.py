from annoy import AnnoyIndex
from os import listdir
from os.path import isfile, join
import json
import torch

all_json = [f for f in listdir("results/") if isfile(join("results/", f))]
print(all_json)


def get_vector(filename):
    with open("results/" + filename) as f:
        data = json.load(f)
        return torch.FloatTensor(data["vector"])


def get_name(filename):
    with open("results/" + filename) as f:
        data = json.load(f)
        return data["name"]


print(get_vector(all_json[0]).shape)
dim = 2048
tree = AnnoyIndex(dim, 'angular')
print(len(all_json))
for i in range(len(all_json)):
    print(get_vector(all_json[i])[0])
    tree.add_item(i, get_vector(all_json[i]))

tree.build(10)
print(tree.get_nns_by_item(4, 20))
