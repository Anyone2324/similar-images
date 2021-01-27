import torch.nn as nn
import json
from os import listdir
from os.path import isfile, join
import torch
from scipy.spatial import distance


model = nn.CosineSimilarity(dim=1, eps=1e-08)

all_json = [f for f in listdir("results/") if isfile(join("results/", f))]

target = all_json[0]


def get_vector(filename):
    with open("results/" + filename) as f:
        data = json.load(f)
        return torch.FloatTensor(data["vector"])


def get_name(filename):
    with open("results/" + filename) as f:
        data = json.load(f)
        return data["name"]


print(get_vector(target))
distances = []
for i in range(len(all_json)):
    compared = all_json[i]
    distances.append({"name": get_name(compared), "tensor": distance.cosine(get_vector(target), get_vector(compared))})

print(distances)
top = []
count_of_top = 0
while count_of_top < 25:
    minimum = 1
    minimumIndex = -1
    for i in range(len(distances)):
        if distances[i]["tensor"] < minimum:
            minimumIndex = i
            minimum = distances[i]["tensor"]
    top.append(distances[minimumIndex]["name"])
    del distances[minimumIndex]
    count_of_top += 1

print(top)
