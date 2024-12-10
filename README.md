# DSDE Project with Scopus datadset

## Project Structure
```
dsde-proj
|
|-- clean.ipynb
|-- scopus.ipynb
|-- selenium_scraping.ipynb
|-- viz.py
|-- .streamlit/
|    |-- config.toml
|-- data/
|    |-- ASJC_cat.csv
|    |-- coordinate_country.csv
|    |-- ref_cite_count.csv
|    |-- ref_cite_count_href.csv
|    |-- raw_data.csv
|    |-- scopus_data.csv
|    |-- viz_data.parquet.gzip
```


## How to get "raw_data.csv"
1. In "clean.ipynb"
2. run first cell to import library (assuming you already have scopus dataset in your machine)
3. run cell below one of these header __"Load json (multithread)"__ , __"Load json (serial)"__
4. run cell below __"Load json (multithread)"__ to export csv
