import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

newNodeList = []
newEdgeList = []

orgNums = [(int(group), len(orgs)) for group, orgs in data.items() if len(orgs) < 80]
orgNums = sorted(orgNums, key=lambda x: x[1], reverse=True)[:120]

for group, numOrgs in orgNums:
    namesList = []
    orgs = data[str(group)]
    for org in orgs:
        namesList.append((len(org['name'].split()), int(org['oid'])))
    namesList = sorted(namesList, key=lambda x: x[0])
    centerId = namesList[0][1]
    for org in orgs:
        """
        if centerId != int(org['oid']):
            newNodeList.append({'id': int(org['oid']), 'value': 15, 'x': float(org['x']), 'y': float(org['y']), 'title': str(org['originalMention']), 'label': str(org['name']), 'group': group})
            newEdgeList.append({'from': int(org['oid']), 'to': centerId})
        else:
            newNodeList.append({'id': int(org['oid']), 'value': 30, 'x': float(org['x']), 'y': float(org['y']), 'title': str(org['originalMention']), 'label': str(org['name']), 'group': group})
        """
        if centerId != int(org['oid']):
            newNodeList.append({'id': int(org['oid']), 'value': 15, 'title': str(org['originalMention']), 'label': str(org['name']), 'group': group})
            newEdgeList.append({'from': int(org['oid']), 'to': centerId})
        else:
            newNodeList.append({'id': int(org['oid']), 'value': 30, 'title': str(org['originalMention']), 'label': str(org['name']), 'group': group})

print ('Nodes : {}'.format(len(newNodeList)))
print ('Edges : {}'.format(len(newEdgeList)))
print ('Groups : {}'.format(len(orgNums)))

with open('clusterNodes.js', 'w') as f:
    f.write("var nodes = ")
    json.dump(newNodeList, f)
    f.write(";")
with open('clusterEdges.js', 'w') as f:
    f.write("var edges = ")
    json.dump(newEdgeList, f)
    f.write(";")
