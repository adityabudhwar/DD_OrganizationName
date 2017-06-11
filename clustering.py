import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation, KMeans
import editdistance
import re
import sys
import spacy

nlp = spacy.load('en')


def _addLocation(oid, label, remainingText, fullText, loc):
    index = remainingText.find(loc)
    locations = []

    if index > 0:
        remainingText = remainingText[:index] + remainingText[index + len(loc):]
        locations.append({
            'oid': oid,
            'type': label,
            'text': fullText,
            'startIndex': index,
            'endIndex': index + len(loc)
        })
    return remainingText, locations


def createLocationDf(dataframe):
    """
    TODO: Also look for qualifiers like 'northern', 'eastern', 'western', 'southern', etc.
    """
    states = pd.read_csv('states.csv')

    locationRows = []
    for index, oid, name in dataframe.itertuples():
        # print(row)
        doc = nlp(str(name))
        newLocations = []
        remainingText = str(doc.text)

        for ent in doc.ents:
            remainingText = remainingText[:ent.start] + remainingText[ent.end:]
            newLocations.append({
                'oid': oid,
                'type': ent.label_,
                'text': ent.text,
                'startIndex': ent.start_char,
                'endIndex': ent.end_char
            })

        lowerText = remainingText.lower()
        for index, state, abbrev in states.itertuples():
            remainingText, locations = _addLocation(oid, 'LOC', lowerText, ent.text, state)
            newLocations.extend(locations)
            remainingText, locations = _addLocation(oid, 'LOC', remainingText.lower(), ent.text, abbrev)
            newLocations.extend(locations)

        locationRows.extend(newLocations)

    locationDf = pd.DataFrame(locationRows)
    locationDf.to_csv('dd_org_ents.csv', sep=',', index=False)
    print('Finished creating location DF')

    return locationDf


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

    def getDistanceCombined(self, org1, org2):
        org1 = str(org1)
        org2 = str(org2)
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

    def clusterData(self, dataframe):
        # Preprocess the data to remove the junk org names
        # filterd_data = np.asarray(self.preprocess(dataframe.name.values))
        # print(filterd_data)
        filterd_data = dataframe.name.values
        # create the distance matrix
        distances = -1 * np.array([[self.getDistanceCombined(orgOne, orgTwo)
                                    for orgOne in filterd_data]
                                   for orgTwo in filterd_data],
                                  dtype=float)

        affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
        # affprop = AgglomerativeClustering(n_clusters=500, affinity='precomputed', linkage='complete')

        affprop.fit(distances)
        for cluster_id in np.unique(affprop.labels_):
            # exemplar = filterd_data[affprop.cluster_centers_indices_[cluster_id]]
            cluster = filterd_data[np.nonzero(affprop.labels_ == cluster_id)]
            # cluster_str = ", ".join(cluster)
            # print(" - *%s:* %s" % (exemplar, str(cluster)))
            print("Cluster: " + str(cluster))

    def kMeansCluster(self, dataframe, k):
        distances = np.array([[editdistance.eval(orgOne, orgTwo)
                               for orgOne in dataframe.name.values]
                              for orgTwo in dataframe.name.values],
                             dtype=float)
        cluster = KMeans(k, n_jobs=-1)
        cluster.fit(distances)
        print('Labels:', cluster.labels_)
        groups = self.groupEntities(cluster, dataframe)
        return cluster

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
        groups = {clusterId: [] for clusterId in uniqueClusters if clusterId != -1}

        for i, id in enumerate(fitCluster.labels_):
            # cluster ID == -1 means the point is seen as noise
            if id != -1:
                groups[id].append({'oid': dataframe.oid.values[i],
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
    # orgNamesDf['Locations'] = orgNamesDf.apply(createLocationDf)
    locationDf = createLocationDf(orgNamesDf)
    # orgNamesDf.to_csv('dd_org_loc_samples.csv', sep=',')
    # print(orgNamesDf.Locations)
    ddorgtools = DDOrgTools()
    # ddorgtools.clusterData(orgNamesDf)
    # ddorgtools.findTopFrequencyWords(orgNamesDf, "matrix.csv")
    # cluster = ddorgtools.levenshteinCluster(orgNamesDf)
    # cluster = ddorgtools.kMeansCluster(orgNamesDf, int(len(orgNamesDf) * .15))
    # groups = ddorgtools.groupEntities(cluster, orgNamesDf)
    # print(ddorgtools.getDistance("Hello There World. llc xyz", "Hello World llc"))


if __name__ == '__main__':
    main(sys.argv)
