# Digital Democracy Organization Disambiguation

## Contributors

## Running Search
- Run Elasticsearch:
    - Copy `search/elasticsearch2.2.0` folder to the server
    - `cd search/elasticsearch2.2.0` (on server)
    - `./elasticsearch -d`
- Indexing:
    - `cd search/MD_Search`
    - `python3 index.py <path to json_data>`
- Search: UI is hosted at [frank.ored.calpoly.edu/MDSearch/index_organization.html)

## Clustering

### DBPedia Local Installation: requires Java
DBPedia is required to resolve abbreviations and help normalize organization names.
- From one terminal session, run the run_dbpedia.sh script to download DBPedia dependencies and start the local DBPedia server on port 1111 (DBPedia’s index file will require about 7GB of storage)

### Clustering Script
#### NOTE: Re-running the clustering script will consume a large amount of memory, and will likely take a full day to run depending on the machine
After starting the DBPedia server from one terminal window, open another terminal and
- Run `pip install -r requirements.txt` from within the project directory
- Run `python3 -m spacy.en.download all`
- Run the run.sh script with the path to the Digital Democracy organization TSV file as a command line argument
	- E.g.: `./run.sh dd_organizations.tsv`
