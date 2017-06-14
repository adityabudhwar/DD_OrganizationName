#!/usr/bin/env bash

wget http://downloads.dbpedia-spotlight.org/dbpedia_lookup/dbpedia-lookup-3.1-jar-with-dependencies.jar
wget http://downloads.dbpedia-spotlight.org/dbpedia_lookup/models/2015-10.tar.gz
tar -xvf 2015-10.tar.gz
rm 2015-10.tar.gz
java -jar dbpedia-lookup-3.1-jar-with-dependencies.jar 2015-10