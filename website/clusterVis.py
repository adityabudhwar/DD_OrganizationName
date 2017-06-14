import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

newNodeList = []
newEdgeList = []

for group, orgs in data.items():
    namesList = []
    for org in orgs:
        namesList.append((len(org['name'].split()), org['oid']))
    namesList = sorted(namesList, key=lambda x: x[0])
    centerId = namesList[0][1]
    for org in orgs:
        newNodeList.append({'id': org['oid'], 'label': org['name'], 'group': group})
        if centerId != org['oid']:
            newEdgeList.append({'from': org['oid'], 'to': centerId})

with open('clusterNodes.js', 'w') as f:
    f.write("var nodes = ")
    json.dump(newNodeList, f)
    f.write(";")
with open('clusterEdges.js', 'w') as f:
    f.write("var edges = ")
    json.dump(newEdgeList, f)
    f.write(";")
