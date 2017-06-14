#!/usr/bin/env bash
python3 clustering.py $1
python3 website/clusterVis.py clusterOut.json
mv clusterNodes.js website/
mv clusterEdges.js website/
