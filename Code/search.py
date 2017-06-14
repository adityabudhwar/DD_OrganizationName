import sys
import re
from os import listdir
from os.path import isfile, join
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import cgi
import json
import nltk
import requests
import html
import string
import demjson
from acronym import Acronym

url = "http://localhost:9200/orgs/_search"

def search(url, query):
    r = requests.post(url, data = query)
    return r.content

from pickle import dump, load
def save_classifier(classifier, filename):
    f = open(filename, 'wb')
    dump(classifier, f, -1)
    f.close()

def load_classifier(filename):
    f = open(filename, 'rb')
    classifier = load(f)
    f.close()
    return classifier

def create_query_exact(search_str):
    query = {}
    query['size'] = 100
    query['query']= {}
    query['query']['bool'] = {}
    should_list = []

    match_1 = {}
    match_1['term'] = {}
    match_1['term']['ORG_E'] = search_str
    should_list.append(match_1)
    len_str = len(search_str.split())
    # Possibly an acronym
    if(len_str == 1):
        match_1 = {}
        match_1['term'] = {}
        match_1['term']['ABR'] = search_str
        should_list.append(match_1)
        match_1 = {}
        match_1['term'] = {}
        match_1['term']['ORG_E'] = search_str
        should_list.append(match_1)
    elif(len_str > 2):

        ac = Acronym(search_str)
        list_p = ac.get_possible_acronyms()
        if(len(list_p) != 0):
            #match_1 = {}
            #match_1['terms'] = {}
            #match_1['terms']['ABR'] = list_p
            #should_list.append(match_1)
            match_1 = {}
            match_1['terms'] = {}
            match_1['terms']['ORG_E'] = list_p
            should_list.append(match_1)
    query['query']['bool']['should'] = should_list
    return query
    
def create_query_all_match(search_str):
    # Create match queries for all the individual words in the search_str and then create a bool must query

    query = {}
    query['size'] = 100
    query['query'] = {}
    query['query']['bool'] = {}
    should_list = []

    for word in search_str.split():
        match_1 = {}
        match_1['match'] = {}
        match_1['match']['ORG'] = {}
        match_1['match']['ORG']['query'] = word
        should_list.append(match_1)

    query['query']['bool']['must'] = should_list
    
    return query

def create_abbr_query(search_str):
    len_str = len(search_str.split())
    # Possibly an acronym
    if(len_str == 1):
        query = {}
        
        query['size'] = 100
        query['query'] = {}
        query['query']['bool'] = {}
        #query['bool'] = {}
        should_list = []

        match_1 = {}
        match_1['term'] = {}
        match_1['term']['ABR'] = search_str
        should_list.append(match_1)

        query['query']['bool']['must'] = should_list
        return query
    elif(len_str == 2):
        return None
    else:
        ac = Acronym(search_str)
        list_p = ac.get_possible_acronyms()
        if(len(list_p) == 0):
             return None
        else:
             query = {}
             
             query['size'] = 100
             query['query'] = {}
             query['query']['bool'] = {}
             #query['bool'] = {}
             should_list = []

             match_1 = {}
             match_1['terms'] = {}
             match_1['terms']['ABR'] = list_p
             should_list.append(match_1)
             
             match_1 = {}
             match_1['query_string'] = {}
             match_1['query_string']['default_field'] = 'ORG'
             match_1['query_string']['query'] = ' OR '.join(list_p)
             should_list.append(match_1)

             query['query']['bool']['should'] = should_list
             return query
def create_query_match(search_str, phrase=0, size=100):

    search_str = re.escape(search_str)
    query = {}
    query['size'] = size
    query['query'] = {}
    query['query']['bool'] = {}
    should_list = []

    match_1 = {}
    match_1['match'] = {}
    match_1['match']['ORG'] = {}
    match_1['match']['ORG']['query'] = search_str
    if phrase==1:
        match_1['match']['ORG']['type'] = 'phrase'
    should_list.append(match_1)

    query['query']['bool']['must'] = should_list

    return query

def query_ES(query, prev_list=[]):
    import json
    query = json.dumps(query)
    content = search(url, query)
    result = {}
    result = demjson.decode(content)
    idd = None
    org = None
    org_list = list()
    for i in result.keys():
        if i == "hits" and result["hits"]:
            for j in result["hits"].keys():
                if j == "hits" and result["hits"]["hits"]:
                    for val in result["hits"]["hits"]:
                        #for k in val.keys():
                        org = val["_source"]["ORG"]
                        idd  = val["_source"]["ID"]
                        if org not in prev_list:
                       # print(idd.encode('utf-8'))
                            org_list.append(org)
     #                       print(org + ": " + str(val["_score"]))
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return org_list

def main(argv):


    search_str = argv[1].lower()
    final_out = {}
    orgs = []
    # Find the org names which match the exact query
    query = create_query_exact(search_str)
    #print(query)
    org_list = query_ES(query) 
    final_out["exact_org"] = org_list
    orgs.extend(org_list)
    query = create_query_match(search_str, phrase=1)
    org_list = query_ES(query, orgs) 
    final_out["org_chapters"] = org_list
    orgs.extend(org_list)
    query = create_query_all_match(search_str)
    org_list_similar = query_ES(query, orgs)
    final_out["similar_orgs"] = org_list_similar
    orgs.extend(org_list_similar)
    # Find the org names which are related i.e contain some terms similar to the query
    query = create_query_match(search_str, phrase=0, size=len(orgs)+3)
    org_list_related = query_ES(query, orgs) 
    #final_out["related_orgs"] = org_list_related
    final_out["similar_orgs"].extend(org_list_related)
    #print(json.dumps(final_out))

    query = create_abbr_query(search_str)
    if query is not None:
      #  print(query)
        org_list_abbr = query_ES(query, orgs) 
        final_out["abbr_match"] = org_list_abbr
    else:
        final_out["abbr_match"] = []
    print(json.dumps(final_out))

if __name__ == "__main__":
    main(sys.argv)

