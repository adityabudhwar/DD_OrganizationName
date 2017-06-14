import pandas as pd

testEnts = ['American Civil Liberties Union,',
            'u American Civil Liberties Union',
            'American Civil Liberties Union (4',
            'American Civil Liberties Union, California',
            'American Civil Liberties Union of California',
            'American Civil Liberties Union of Northern California',
            'American Civil Liberties Union of CA',
            'American Civil Liberties Union of California',
            'American Civil Liberties Union in California',
            'u American Civil Liberties Union of California',
            'The American Civil Liberties Union of California',
            'American Civil Liberties Union California Public Defenders Association',
            'American Civil Liberties Union Foundation Of Southern California',
            'Silicon Valley De - Bug American Civil Liberties Union',
            'American Civil Liberties Union of California Center for Advocacy & Policy',
            'American Civil Liberties Union / Northern California / Southern California / San Diego & Imperial Counties',
            'American Civil Liberties Union / Northern California / Southern California / San Diego And Imperial Counties',
            'Adobe', 'Adobe Inc.', 'Adobe Systems', 'Adobe Systems Inc.',
            'Americans for Civil Liberties Union of California',
            'American Civil Liberties of California',
            'American Civil Liberties Association of California',
            'California Civil Liberties Advocacy',
            'California Civil Liberties Council',
            'ACLU California',
            'ACLU of California',
            'ACLU of Northern California',
            'u ACLU California',
            'Google Inc.',
            'Google Capital',
            'Google Inc. Major Donor FPPC ID#1278416']
testOrgs = []
for i, org in enumerate(testEnts):
    testOrgs.append({'oid': i, 'name': org})

pd.DataFrame(testOrgs).to_csv('testOrgs.tsv', index=False, sep='\t', columns=['oid', 'name'])
