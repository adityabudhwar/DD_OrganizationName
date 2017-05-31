import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import editdistance
from joblib import Parallel, delayed
from multiprocessing import Pool

from lavenstienDistance import levenshtein


def lavensteinCluster(dataframe):
    # distances = np.ndarray(shape=len(dataframe) * len(dataframe),
    #                        dtype=float)
    # distances = Parallel(n_jobs=-1)(delayed(editdistance.eval)(orgOne, orgTwo)
    #                                    for orgOne in dataframe.name.values
    #                                    for orgTwo in dataframe.name.values)
    distances = np.array([[editdistance.eval(orgOne, orgTwo)
                           for orgOne in dataframe.name.values]
                          for orgTwo in dataframe.name.values],
                         dtype=int)

    # for i, orgOne in enumerate(dataframe.name.values):
    #     for j, orgTwo in enumerate(dataframe.name.values):
    #         if orgOne != orgTwo:
    #             distances[i][j] = levenshtein(orgOne, orgTwo)
    print(distances)
    cluster = DBSCAN(eps=8, metric='precomputed')
    cluster.fit(distances)
    print('Labels:', list(cluster.labels_))
    print('Core sample indices:', cluster.core_sample_indices_)
    print('Number of labels:', len(cluster.labels_))

    return cluster


def groupEntities(fitCluster, dataframe):
    uniqueClusters = set(fitCluster.labels_)
    groups = {clusterId: [] for clusterId in uniqueClusters if clusterId != -1}

    for i, id in enumerate(fitCluster.labels_):
        # cluster ID == -1 means the point is seen as noise
        if id != -1:
            groups[id].append({'oid': dataframe.oid.values[i],
                               'name': dataframe.name.values[i]})

    print('Cluster groups:', groups)
    return groups


def main():
    orgNamesDf = pd.read_csv('corpus/dd_organizations.tsv',
                             dtype={'name': str},
                             sep='\t').sample(n=2000)
    cluster = lavensteinCluster(orgNamesDf)
    groups = groupEntities(cluster, orgNamesDf)


if __name__ == '__main__':
    main()
