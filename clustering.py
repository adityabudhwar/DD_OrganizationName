import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation, KMeans
from sklearn import manifold
import editdistance
import re
import sys
import spacy
import json
import requests
import untangle
import time
import pathlib
from joblib import Parallel, delayed

nlp = spacy.load('en')


class DBPedia:
    def __init__(self, endpoint):
        self.endpoint = endpoint + '/api/search.asmx/KeywordSearch?QueryClass=Organisation&QueryString='

    def getURI(self, keyword):
        url = self.endpoint + keyword

        try:
            result = requests.get(url, timeout=1)
            d = untangle.parse(result.content.decode("utf-8"))
            if hasattr(d.ArrayOfResult, 'Result'):
                all_r = "#####".join(Result.URI.cdata for Result in d.ArrayOfResult.Result)
            else:
                all_r = ""
            all_r = all_r.split("#####")
            if len(all_r) == 0:
                return None
            else:
                return all_r[0]
        except:
            print('Failed to retrieve URI for', keyword)
            return None

    def getURIDict(self, dataframe, output_file):
        fd = open(output_file, "w")
        for index, row in dataframe.iterrows():
            uri = self.getURI(row['name'])
            str1 = str(row['oid']) + "\t" + row['name']
            if uri:
                str1 = str1 + "\t" + uri
            else:
                str1 = str1 + "\t" + ""
            fd.write(str1 + "\n")
        fd.close()


def _addLocation(oid, label, remainingText, fullText, loc, statesDf):
    qualifierRE = re.compile(r'(?:west|western|north|northern|east|eastern|south|southern|central)?\s*({})'.format(
        loc
    ))
    matches = qualifierRE.finditer(remainingText)
    locations = []

    if matches:
        for match in matches:
            remainingText = remainingText[:match.start(0)] + remainingText[match.end(0):]
            remainingText = remainingText.strip()
            locations.append({
                'oid': oid,
                'type': label,
                'text': match.group(0).strip(),
                'startIndex': match.start(0),
                'endIndex': match.end(0)
            })
    return remainingText, locations


def expandLocationAbbrev(loc, statesDf):
    loc = loc.upper()
    abbrevMatch = statesDf.State.loc[statesDf.Abbreviation == loc]

    if not abbrevMatch.empty:
        loc = str(abbrevMatch)

    return loc


def getLocationDf(dataframe, forceReload=False):
    locationDfFile = 'dd_org_ents.csv'

    if forceReload:
        states = pd.read_csv('states.csv')

        locationRows = []
        for index, oid, name in dataframe.itertuples():
            doc = nlp(str(name))
            newLocations = []
            remainingText = str(doc.text)
            locationNames = set()

            for ent in doc.ents:
                if ent.label_ == 'LOC' or ent.label_ == 'GPE':
                    remainingText = remainingText[:ent.start_char] + remainingText[ent.end_char:]
                    remainingText = remainingText.strip()
                    entText = str(ent.text).strip()
                    locationNames.add(entText)
                    newLocations.append({
                        'oid': oid,
                        'type': ent.label_,
                        'text': entText,
                        'startIndex': ent.start_char,
                        'endIndex': ent.end_char
                    })

            lowerText = remainingText.lower()
            for index, state, abbrev in states.itertuples():
                remainingText, locations = _addLocation(oid, 'LOC', lowerText, remainingText.lower(), state, states)
                newLocations.extend(locations)

                remainingText, locations = _addLocation(oid, 'LOC', remainingText.lower(), lowerText, abbrev, states)
                newLocations.extend(locations)

            locationRows.extend(newLocations)

        locationDf = pd.DataFrame(locationRows)
        locationDf.to_csv(locationDfFile, sep=',', index=False)
        print('Finished creating location DF')
    else:
        locationDf = pd.read_csv(locationDfFile)

    return locationDf


def cleanOrgName(org):
    if '&' in org:
        org = org.replace('&', 'and')
    if '.' in org:
        org = org.replace('.', '')
    if ',' in org:
        org = org.replace(',', '')
    if '/' in org:
        org = org.replace('/', '')
    return org


def splitOrgSections(orgMention, oid, locationDf):
    """
    Split the org mention into the location, organization qualifier (llc, co., inc., etc.),
        and the primary organization name with location and org qualifier removed

    :return {locations: list, qualifiers: list, org: str, orgURI: str, originalText: str, oid: int}
    """

    thisLocDf = locationDf[locationDf.oid == oid]
    remainingText = str(orgMention)
    locations = []

    if not thisLocDf.empty:
        for idx, endIndex, oid, startIndex, text, type in thisLocDf.itertuples():
            locations.append(text.lower())
            remainingText = remainingText[:startIndex] + remainingText[endIndex:]

    orgQualifiers = []
    for m in re.finditer(r'\s+(co\.|ltd|lp|llc|inc|pac|assn|political action committee)\.?',
                         remainingText.lower()):  # print('Qualifier match:', m.group(1))
        remainingText = remainingText[:m.start()] + remainingText[m.end():]
        orgQualifiers.append(m.group(1).strip())

    processedMention = nlp(remainingText)
    mentionWords = [str(word.lemma_) for word in processedMention
                    if not word.is_stop and not word.is_punct
                    and not word.is_space]
    org = cleanOrgName(' '.join(mentionWords))
    dbpedia = DBPedia('http://localhost:1111')
    uri = dbpedia.getURI(org)

    if not uri:
        uri = None
    else:
        splitURI = uri.split('/')[-1].split('_')
        uri = ' '.join(splitURI)

    return {'locations': locations,
            'qualifiers': orgQualifiers,
            'org': org,
            'orgURI': uri,
            'originalText': orgMention,
            'oid': str(oid)}


def splitAllOrgSections(splitOrgsFile=None, orgNamesDf=None, locationDf=None):
    if splitOrgsFile is not None:
        with open(splitOrgsFile, 'r') as file:
            splitOrgs = json.load(file)
    elif orgNamesDf is not None and locationDf is not None:
        outfile = 'splitOrgs.json'
        splitOrgs = []
        numMentions = len(orgNamesDf)
        step = int(numMentions / 100)
        if step == 0:
            step = 1

        processed = 0

        for idx, oid, name in orgNamesDf.itertuples():
            split = splitOrgSections(name, oid, locationDf)
            splitOrgs.append(split)
            processed += 1

            if numMentions % step == 0:
                print('Processed {}% of orgs'.format(processed * 100 / numMentions))
                # with open(outfile, 'w') as file:
                #     json.dump(splitOrgs, file)
    else:
        raise ValueError('Neither splitOrgsFile nor orgNamesDf was defined')

    print('SplitAllOrgSections finished')
    return splitOrgs


def computeWordListDistance(listOne, listTwo):
    numDifferentWords = 0

    for wordOne in listOne:
        for wordTwo in listTwo:
            if wordOne != wordTwo:
                numDifferentWords += 1

    return numDifferentWords


def computeOrgDistance(splitOrgOne, splitOrgTwo):
    """
    :param splitOrgOne: {locations: list, qualifier: str, org}
    :param splitOrgTwo: {locations: list, qualifier: str, org}
    :return: distance between splitOrgOne and splitOrgTwo
    """

    dist = 0.0
    orgWeight = 100.0
    qualifierWeight = 1.0
    locationWeight = 2.0

    orgNameOne, orgNameTwo = splitOrgOne['org'], splitOrgTwo['org']
    qualifiersOne, qualifiersTwo = splitOrgOne['qualifiers'], splitOrgTwo['qualifiers']
    locationsOne, locationsTwo = splitOrgOne['locations'], splitOrgTwo['locations']
    orgOneURI, orgTwoURI = splitOrgOne['orgURI'], splitOrgTwo['orgURI']

    # If DBPedia found the URI, compare those, otherwise compare retrieved org names
    if orgOneURI is not None and orgTwoURI is not None:
        spacyOrgOne, spacyOrgTwo = nlp.make_doc(orgOneURI), nlp.make_doc(orgTwoURI)
    else:
        spacyOrgOne, spacyOrgTwo = nlp.make_doc(orgNameOne), nlp.make_doc(orgNameTwo)

    # Spacy doc distance = cosine similarity
    orgSimilarity = spacyOrgOne.similarity(spacyOrgTwo)

    if orgSimilarity < 1:
        dist += orgWeight * (1 - orgSimilarity)

    dist += qualifierWeight * computeWordListDistance(qualifiersOne, qualifiersTwo)
    dist += locationWeight * computeWordListDistance(locationsOne, locationsTwo)

    return dist


def groupOrgClusters(fitCluster, splitOrgs, coordinates):
    uniqueClusters = set(fitCluster.labels_)
    groups = {str(clusterId): [] for clusterId in uniqueClusters if clusterId != -1}

    for i, id in enumerate(fitCluster.labels_):
        # cluster ID == -1 means the point is seen as noise
        if id != -1:
            groups[str(id)].append({'oid': str(splitOrgs[i]['oid']),
                                    'name': splitOrgs[i]['org'],
                                    'originalMention': splitOrgs[i]['originalText'],
                                    'x': coordinates[i][0],
                                    'y': coordinates[i][1]})

    return groups


def computeCoordsFromDist(distances):
    coordFile = 'coords.json'

    if pathlib.Path(coordFile).is_file():
        print('Loading coordinates file')
        with open(coordFile, 'r') as file:
            coords = np.array(json.load(file))
    else:
        print('Recomputing coordinates')
        adist = np.array(distances)
        maxDist = np.amax(adist)
        adist /= maxDist

        mds = manifold.MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)
        results = mds.fit(adist)
        coords = np.array(results.embedding_)

        with open(coordFile, 'w') as outfile:
            json.dump(coords.tolist(), outfile)
    # coords = np.ndarray(shape=distances.shape, dtype=np.float)
    #
    # for i in range(len(distances)):
    #     for j in range(len(distances)):
    #         coords[i][j] = (np.square(distances[0][j]) + np.square(distances[i][0]) - np.square(distances[i][j])) / 2
    return coords


def runDBScan(splitOrgs, eps=40.0, distanceFile=None):
    if distanceFile is None:
        distances = np.ndarray(shape=(len(splitOrgs), len(splitOrgs)), dtype=np.float)
        start = int(time.time()) * 1000
        orgCombinations = [(orgOne, orgTwo) for j, orgTwo in enumerate(splitOrgs)
                           for i, orgOne in enumerate(splitOrgs)]
        distances_1D = Parallel(n_jobs=-1)(delayed(computeOrgDistance)(*orgCombo) for orgCombo in orgCombinations)
        numSplitOrgs = len(splitOrgs)

        for i in range(numSplitOrgs):
            for j in range(numSplitOrgs):
                distances[i][j] = distances_1D[i * numSplitOrgs + j]

        """
        Old single-threaded distance computation:

        for i, orgOne in enumerate(splitOrgs):
            for j, orgTwo in enumerate(splitOrgs):
                if i == j:
                    distances[i][j] = 0.0
                else:
                    distances[i][j] = computeOrgDistance(orgOne, orgTwo)
        """

        print('Finished computing distances in {}ms'.format(
            int(time.time()) * 1000 - start
        ))

        with open('distances.json', 'w') as distFile:
            json.dump(distances.tolist(), distFile)
    else:
        print('Loading distance file')
        with open(distanceFile, 'r') as file:
            distances = np.array(json.load(file))

    coords = computeCoordsFromDist(distances) * 500
    print(coords)
    cluster = DBSCAN(eps=eps, min_samples=1)
    cluster.fit(distances)
    groups = groupOrgClusters(cluster, splitOrgs, coords)
    return groups


class DDOrgTools:
    def preprocess(self, content):
        data = list()
        for line in content:
            if re.match('[a-zA-Z0-9_]+', line):
                data.append(line)
        return data

    stoplist = [',', 'of', '.', 'california', 'inc', 'association', '&', 'pac', 'and', 'for',
                'the', 'llc', 'school', 'group', 'city', 'ca', 'committee', 'community',
                'corporation', 'union', '-', 'associates', 'llp', 'foundation', 'inc.',
                'assn', 'a', 'assoc', 'co', 'corp', '#', 'at', 'cal', 'by', 'on', ':', ';', "enterprise", "enterprises",
                "systems"]

    def filterdata(self, line):
        line = line.lower()
        if '&' in line:
            line = line.replace('&', 'and')
        if '.' in line:
            line = line.replace('.', "")
        if ',' in line:
            line = line.replace(',', "")
        for word in self.stoplist:
            line = line.replace(word, " ")
        for m in re.finditer(r'(llc|inc|pac|assn|political action committee)( |$)', line):
            line = line.replace(m.group(0).strip(), ' ')
        return line

    def bracesRemove(self, line):
        for m in re.finditer(r'(\(|\[).*?(\]|\))', line):
            line = line.replace(m.group(), '')
        return line

    def getDistance(self, org1, org2):
        org1 = self.filterdata(org1)
        org1 = self.bracesRemove(org1)
        org2 = self.filterdata(org2)
        org2 = self.bracesRemove(org2)

        list1 = re.split('\s+', org1)
        list2 = re.split("\s+", org2)
        word_distance = editdistance.eval(re.split('\s+', org1), re.split("\s+", org2))
        char_distance = editdistance.eval(org1.replace(" ", ""), org2.replace(" ", ""))
        if word_distance != 0:
            word_distance = float(word_distance) / max(len(list1), len(list2))
        return word_distance
        # import math
        # return math.pow(100, word_distance)

    def _findAcronyms(self, org, targetOrg, matcher):
        org1Matches = list(matcher.finditer(targetOrg))

        if org1Matches:
            org2FirstChars = ''.join([w[0] for w in org.lower().split(' ')
                                      if len(w) > 0])
            for match in org1Matches:
                acronymChars = re.sub(r'\.', '', match.group().lower())
                if len(acronymChars) > 2 and acronymChars in org2FirstChars:
                    print('Matched acronmym {} in {} to {}'.format(match.group(), targetOrg, org))

    def tryExpandAcronyms(self, org1, org2):
        acronymRE = re.compile(r'((?:[A-Z]\.)+)')

        org1Matches = self._findAcronyms(org1, org2, acronymRE)
        org2Matches = self._findAcronyms(org2, org1, acronymRE)

        if org1Matches:
            pass
        if org2Matches:
            pass

    def getDistanceCombined(self, org1, org2, locationDf):
        # TODO: if one org is a subset of other org, should be very close match
        oid_1, org1 = org1
        oid_2, org2 = org2

        org1 = str(org1)
        org2 = str(org2)

        org_one_locs = locationDf[locationDf.oid == oid_1]
        org_two_locs = locationDf[locationDf.oid == oid_2]

        if not org_one_locs.empty:
            print('Org one locs:', org_one_locs)
        if not org_two_locs.empty:
            print('Org two locs:', org_two_locs)
        # self.tryExpandAcronyms(org1, org2)
        # print(org1)
        # print(org2)
        org1 = self.filterdata(org1)
        org1 = self.bracesRemove(org1)
        org2 = self.filterdata(org2)
        org2 = self.bracesRemove(org2)
        list1 = re.split('\s+', org1)
        list2 = re.split("\s+", org2)
        word_distance = editdistance.eval(list1, list2)
        if word_distance != 0:
            word_distance = float(word_distance) / max(len(list1), len(list2))
        org1 = org1.replace(" ", "")
        org2 = org2.replace(" ", "")
        char_distance = editdistance.eval(org1, org2)
        if char_distance != 0:
            char_distance = float(char_distance) / max(len(org1), len(org2))
        distance = word_distance * char_distance
        return distance

    def clusterData(self, dataframe, locationDf):
        # Preprocess the data to remove the junk org names
        # filterd_data = np.asarray(self.preprocess(dataframe.name.values))
        # print(filterd_data)
        filteredData = dataframe.values
        # create the distance matrix
        distances = -1 * np.array([[(orgOne, orgTwo, locationDf)
                                    for orgOne in filteredData]
                                   for orgTwo in filteredData],
                                  dtype=float)

        # affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
        # affprop = AgglomerativeClustering(n_clusters=500, affinity='precomputed', linkage='complete')
        cluster = DBSCAN(eps=10, min_samples=2)

        cluster.fit(distances)
        for cluster_id in np.unique(cluster.labels_):
            # exemplar = filterd_data[affprop.cluster_centers_indices_[cluster_id]]
            cluster = filteredData[np.nonzero(cluster.labels_ == cluster_id)]
            # cluster_str = ", ".join(cluster)
            # print(" - *%s:* %s" % (exemplar, str(cluster)))
            print("Cluster: " + str(cluster))

    def kMeansCluster(self, dataframe, k):
        distances = np.array([[editdistance.eval(orgOne, orgTwo)
                               for orgOne in dataframe.name.values]
                              for orgTwo in dataframe.name.values],
                             dtype=float)
        # with open('editDistances.json', 'w') as distanceOutfile:
        #     json.dump(distances, distanceOutfile)
        cluster = KMeans(k, n_jobs=-1)
        cluster.fit(distances)
        print('Labels:', cluster.labels_)
        groups = self.groupEntities(cluster, dataframe)
        return groups

    def levenshteinCluster(self, dataframe, maxDist=2):

        # Preprocess the data to remove the junk org names
        # filterd_data = self.preprocess(dataframe.name.values)
        filtered_data = dataframe.name.values
        # create the distance matrix
        distances = np.array([[self.getDistanceCombined(orgOne, orgTwo)
                               for orgOne in filtered_data]
                              for orgTwo in filtered_data],
                             dtype=float)

        # print(distances)
        # cluster = AgglomerativeClustering(n_clusters=500, affinity='precomputed', linkage='complete')
        cluster = DBSCAN(eps=maxDist, metric='precomputed')
        cluster.fit(distances)
        print('Labels:', list(cluster.labels_))
        # print('Core sample indices:', cluster.core_sample_indices_)
        # print('Number of labels:', len(cluster.labels_))

        return cluster

    def groupEntities(self, fitCluster, dataframe):
        uniqueClusters = set(fitCluster.labels_)
        groups = {str(clusterId): [] for clusterId in uniqueClusters if clusterId != -1}

        for i, id in enumerate(fitCluster.labels_):
            # cluster ID == -1 means the point is seen as noise
            if id != -1:
                groups[str(id)].append({'oid': str(dataframe.oid.values[i]),
                                        'name': dataframe.name.values[i]})

        print('Cluster groups:', groups)
        return groups

    def findTopFrequencyWords(self, dataframe, output_file):
        allWords = list()
        for value in dataframe.name.values:
            import nltk
            if pd.isnull(value):
                continue
            allWords.extend(nltk.tokenize.word_tokenize(value))
            # stopwords = nltk.corpus.stopwords.words('english')
        allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords)
        # frequent_words = allWordExceptStopDist.most_common(200)
        list_all = list(allWordExceptStopDist.keys())
        fd = open(output_file, "w")
        for value in dataframe.name.values:
            vector = np.zeros(len(list_all))
            import nltk
            if not pd.isnull(value):

                allWords = nltk.tokenize.word_tokenize(value)
                for word in allWords:
                    if word in list_all:
                        vector[list_all.index(word)] = 1
            fd.write(",".join(map(str, vector)))
            fd.write("\n")
        fd.close()
        # stopwords = nltk.corpus.stopwords.words('english')


def main(argv):
    orgNamesDf = pd.read_csv(argv[1],
                             dtype={'name': str},
                             sep='\t')
    print('Clustering {} orgs'.format(len(orgNamesDf)))
    locationDf = getLocationDf(orgNamesDf, True)
    print(orgNamesDf.head())
    print(orgNamesDf.columns)
    print('Location unique types:', locationDf.type.unique())
    print(locationDf.head())
    #
    splitOrgs = splitAllOrgSections(orgNamesDf=orgNamesDf, locationDf=locationDf)
    # splitOrgs = splitAllOrgSections(splitOrgsFile='splitOrgs.json')
    # splitOrgs = splitAllOrgSections(splitOrgsFile='splitOrgs_gold.json')

    # groups = runDBScan(splitOrgs, 'distances.json')
    # groups = runDBScan(splitOrgs, distanceFile='distances_gold.json')
    groups = runDBScan(splitOrgs, 40.0)

    with open('clusterOut.json', 'w') as clusterOutfile:
        json.dump(groups, clusterOutfile)


if __name__ == '__main__':
    main(sys.argv)
