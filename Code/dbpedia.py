import requests
import untangle

class dbpedia:

    def getURI(self, keyword):
        url = "http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryClass=Organisation&QueryString=" + keyword
        result = requests.get(url)
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

    def getURIDict(self, dataframe, output_file):
        fd =  open(output_file, "w")
        for index, row in dataframe.iterrows():
            uri = self.getURI(row['name'])
            str1 = str(row['oid']) + "\t" + row['name']
            if uri:
                str1 = str1 + "\t" + uri
            else:
                str1 = str1 + "\t" + ""
            fd.write(str1 + "\n")
        fd.close()

import pandas as pd
import sys
if __name__ == '__main__':

    dbpedia = dbpedia()
    print(dbpedia.getURI("Adobe"))
    print(dbpedia.getURI("YMCA"))
    print(dbpedia.getURI("Adobe Systems"))
    print(dbpedia.getURI("Apple"))
    print(dbpedia.getURI("Apple Inc"))
    orgNamesDf = pd.read_csv(sys.argv[1],
                             dtype={'name': str},
                             sep='\t')
    dbpedia.getURIDict(orgNamesDf, sys.argv[2])


