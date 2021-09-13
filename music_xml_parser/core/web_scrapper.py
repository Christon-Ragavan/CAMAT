"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""
import re
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
warnings.filterwarnings("ignore", 'This pattern has match groups')
pd.options.display.max_colwidth = 2000
pd.set_option('display.max_rows', 9999)
pd.set_option('display.max_columns', 9999)
pd.set_option('display.width', 99999)


def get_from_server():
    pass

def auth(site):
    r = requests.get(site, auth=('analysekurs', 'noten'))
    return r

def extract_links(url):
    # site = "https://analyse.hfm-weimar.de/doku.php?id=scfr"
    root_url = "https://analyse.hfm-weimar.de"
    r = auth(url)
    # soup = bs(r.content, features="html.parser")
    soup_2 = bs(r.content, 'lxml')

    data = []
    for link in soup_2.find_all('a'):

        try:
            if 'wikilink1' in link.get('class'):
                data.append([str(link.text), str(root_url + link.get('href'))])
        except:
            continue
    return data


def scrape_database(database_csv, do_print=False):
    composer_url = "https://analyse.hfm-weimar.de/doku.php?id=komponisten"
    composer_data = extract_links(url=composer_url)


    entrie_database_list = []
    # 51 62
    for id, i in enumerate(composer_data):
        if do_print:
            print(f"{id}/{len(composer_data)} - {i}")
        if i[0]=='hier':
            continue

        pd_data = extract_tables(i[0], i[1])
        entrie_database_list.append(pd_data)
    df_entrie_database = pd.concat(entrie_database_list,
                                   ignore_index=True,
                                   verify_integrity=True,
                                   copy=False)
    df_entrie_database = df_entrie_database.drop_duplicates()
    df_entrie_database.to_csv(database_csv, index=False, sep=';')

    return df_entrie_database