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

import os

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
                d = [str(link.text), str(root_url + link.get('href'))]
                if ' (1' in d[0]:
                    data.append(d)
        except:
            continue
    return data

def _convert_col(thead):
    column_names = []
    for th in thead.find_all('th'):
        t = th.text.strip()
        if t == 'Werktitel':
            t = 'url'
        if t == 'Satz-Nr.':
            t = 'Movement Number'
        if t == 'Tonart':
            t = 'Key'
        if t == 'Jahr':
            t = 'Year'

        column_names.append(t)
    column_names.insert(-1, 'Title')
    return column_names


def _clean_year_col(yr_r):
    yr_n = []
    for id, yr in enumerate(yr_r):
        yr = str(yr)
        if len(yr) == 0:
            yr_a = np.nan
        else:
            if 'ca.' in yr:
                yr = yr.replace('ca.', '')

            if '/' in yr:
                yr_s, yr_e = re.split('-', yr)
                if '/' in yr_s:
                    yr_s = re.split('/', yr_s, 2)[0]
                elif '/' in yr_e:
                    yr_e = re.split('/', yr_e, 2)[1]
                yr_s = str(yr_s)
                yr_e = str(yr_e)
                yr_a = yr_s+'-'+yr_e
            else:
                yr_a = yr
        yr_n.append(yr_a)
    assert len(yr_n) == len(yr_r)
    return yr_n


def extract_tables(artist_meta, url):
    artist_name = artist_meta.split('(')[0]

    try:
        at = artist_name.split(',')
        artist_name = at[1]+at[0]
        artist_name = artist_name.lstrip()
    except:

        artist_name = artist_name
        artist_name = artist_name.lstrip()

    try:
        artist_lega = artist_meta.split('(')[1].replace(')', '')
        artist_lega = artist_lega.lstrip()
    except:
        artist_lega = np.nan

    root_url = "https://analyse.hfm-weimar.de"
    r = auth(url)
    soup = bs(r.content, 'lxml')
    table = soup.find('table')
    thead = soup.find('thead')
    column_names = _convert_col(thead)


    data = []
    for row in table.find_all('tr'):
        row_data = []
        for i, td in enumerate(row.find_all('td')):
            td_check = td.find('a')
            if td_check is not None:
                link = td.a['href']
                row_data.append(td_check.text)
                row_data.append(link)
            else:
                not_link = ''.join(td.stripped_strings)
                if not_link == '':
                    not_link = None
                row_data.append(not_link)
        data.append(row_data)
    data = data[1:]
    pd_data = pd.DataFrame(data, columns=column_names)
    artist_lega = [artist_lega]*len(pd_data)
    artist_name = [artist_name]*len(pd_data)

    artist_lega_c = _clean_year_col(artist_lega)
    pd_data.insert(0, 'Composer', artist_name)
    pd_data.insert(1, 'Life Time', artist_lega_c)

    return pd_data

def search(head, keyword, df):
    search = '|'.join(keyword)
    searched = df[df[head].str.contains(search, na=False)]
    return searched


def _search_database(col, keywords):
    database_csv = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/composer_web_database.csv'

    if os.path.isfile(database_csv):
        df = pd.read_csv(database_csv)
        df = df.replace(np.nan, '', regex=True)
    else:
        composer_url = "https://analyse.hfm-weimar.de/doku.php?id=komponisten"
        composer_data = extract_links(url=composer_url)
        entrie_database_list = []

        for id, i in enumerate(composer_data):

            print(f"{id}/{len(composer_data)} - {i}")
            pd_data = extract_tables(i[1])
            entrie_database_list.append(pd_data)


        df = pd.concat(entrie_database_list,
                                       ignore_index=True,
                                       verify_integrity=True,
                                       copy=False)

        df.to_csv('/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/composer_web_database.csv',
                                  index=False)

    search_key = '|'.join(keywords)
    s_df = df[df[col].str.contains(search_key)]
    return s_df

def search_lifetime_range(df, yr_keys):
    yr_r = df['Life Time'].to_list()
    l, r = re.split('-', yr_keys, 2)
    l = int(l)
    r = int(r)

    yr_idx = []
    for id, yr in enumerate(yr_r):
        yr = str(yr)
        if len(yr) == 0:
            continue
        else:

            if 'ca.' in yr:
                yr = yr.replace('ca.', '')
            yr_s, yr_e = re.split('-', yr)
            r1 = int(yr_s)
            r2 = int(yr_e)
            if l in range(r1, r2 + 1) or r in range(r1, r2 + 1):
                yr_idx.append(id)

    df_y = df.iloc[yr_idx, :]

    return df_y



    #sdf = df[df['duration'].between(left=l, right=r)]
    pass

def search_year_range(df, yr_keys):
    yr_r = df['Year'].to_list()
    l, r = re.split('-', yr_keys, 2)
    l = int(l)
    r = int(r)

    yr_idx = []
    for id, yr in enumerate(yr_r):
        yr = str(yr)
        if len(yr) == 0:
            continue
        elif yr == np.nan:
            continue
        else:
            try:
                ref = int(yr)

                if ref in range(l, r + 1):
                    yr_idx.append(id)

            except:
                continue
    df_y = df.iloc[yr_idx, :]

    return df_y


def search_year(df, yr_keys):
    yr_r = df['Life Time'].to_list()
    sel_idx = []
    for id, yr in enumerate (yr_r):
        yr= str(yr)
        if len(yr) ==0:
            continue
        try:
            yr_s, yr_e = re.split('-', yr)
        except:
            continue

        if '/' in yr_s:
            yr_s = re.split('/', yr_s, 2)[0]
        elif '/' in yr_e:
            yr_e = re.split('/', yr_e, 2)[1]
        yr_s = yr_s.replace('ca.', '')
        yr_e = yr_e.replace('ca.', '')
        yr_s = int(yr_s)
        yr_e = int(yr_e)
        for ykc in yr_keys:
            ykc = int(ykc)
            if ykc in range(yr_s, yr_e):
                sel_idx.append(id)
    df_y = df.iloc[sel_idx, :]
    return df_y


def get_year_range(df, yr_keys):
    yr_r = df['Life Time'].to_list()

    sel_idx = []
    for id, yr in enumerate (yr_r):
        yr= str(yr)
        yr_s, yr_e = re.split('-', yr)
        yr_s = int(yr_s)
        yr_e = int(yr_e)
        for ykc in yr_keys:
            ykc = int(ykc)
            if ykc in range(yr_s, yr_e):
                sel_idx.append(id)
    df_y = df.iloc[sel_idx, :]

    return df_y



def search_database_intern2(df, col, keywords):
    search_key = '|'.join(keywords) # find all keywords
    s_df = df[df[col].str.contains(search_key)]
    return s_df

def scrape_database(database_csv, do_print=False):
    composer_url = "https://analyse.hfm-weimar.de/doku.php?id=komponisten"
    composer_data = extract_links(url=composer_url)
    entrie_database_list = []
    for id, i in enumerate(composer_data):
        if do_print:
            print(f"{id}/{len(composer_data)} - {i}")
        if i[0]=='hier':
            continue
        print(i[0], i[1])
        pd_data = extract_tables(i[0], i[1])
        entrie_database_list.append(pd_data)
    df_entrie_database = pd.concat(entrie_database_list,
                                   ignore_index=True,
                                   verify_integrity=True,
                                   copy=False)
    df_entrie_database = df_entrie_database.drop_duplicates()
    df_entrie_database['xml_file_name'] = [os.path.basename(i) for i in  df_entrie_database['url'].tolist()]
    try:
        df_entrie_database.to_csv(database_csv, index=False, sep=';')
    except:
        web_database_csv_path = os.getcwd().replace('ipynb', os.path.join('data', 'xmls_to_parse', 'hfm_database',
                                                                     'composer_web_database.csv'))
        df_entrie_database.to_csv(web_database_csv_path, index=False, sep=';')

    return df_entrie_database


def search_database_intern(df, search_keywords, apply_precise_keyword_search, do_print=False):

    db_list = []
    if do_print:
        print("-------------------------------------------------")
        print("Searching...")
        print("-------------------------------------------------")

    for col in search_keywords:
        sepra = ", "
        if search_keywords[col] == None:
            s = 'None'
            if do_print:
                print(f"{col : <20} {s: <20}")
        else:
            if do_print:
                if type(search_keywords[col]) == str:
                    print(f"{col : <20} { search_keywords[col] : <20}")
                else:
                    print(f"{col : <20} { sepra.join(search_keywords[col]) : <20}")
    if do_print:
        print("-------------------------------------------------")

    for col in search_keywords:

        if search_keywords[col] == None:

            continue

        else:
            if 'Composer' == col:
                #search_key = '|'.join(search_keywords[col])
                search_key = '(.*)|(.*)'.join(search_keywords[col])
                search_key = '(.*)' + search_key + '(.*)'
                s_df = df[df[col].str.contains(search_key, flags=re.IGNORECASE, case=False)]
                db_list.append(s_df)
            elif 'Movement Number' == col:

                search_key = '(.*)|(.*)'.join(search_keywords[col])
                search_key = '(.*)' + search_key + '(.*)'
                s_df = df[df[col].str.contains(search_key, flags=re.IGNORECASE, case=False)]
                db_list.append(s_df)

            elif 'Key' == col:
                search_key = '|'.join(search_keywords[col])
                s_df = df[df[col].str.contains(search_key)]
                s_df = s_df.drop_duplicates()
                db_list.append(s_df)

            elif 'Title' == col:
                #search_key = '|'.join(search_keywords[col])
                #search_key = '(.*)' + search_key + '(.*)'
                search_key = '(.*)|(.*)'.join(search_keywords[col])
                search_key = '(.*)' + search_key + '(.*)'

                s_df = df[df[col].str.contains(search_key, flags=re.IGNORECASE, case=False)]
                db_list.append(s_df)

            elif 'Life Time Year' == col:
                s_df = search_year(df=df, yr_keys=search_keywords['Life Time Year'])
                db_list.append(s_df)

            elif 'Life Time Range' == col:
                s_df = search_lifetime_range(df=df, yr_keys=search_keywords['Life Time Range'])
                db_list.append(s_df)


            elif 'Year Range' == col:
                s_df = search_year_range(df=df, yr_keys=search_keywords['Year Range'])
                db_list.append(s_df)




    if len(db_list) != 0:
        if apply_precise_keyword_search:
            df_s_db = reduce(lambda x ,y: pd.merge(x,y, how='inner', indicator=False), db_list)
        else:
            df_s_db = pd.concat(db_list,
                                ignore_index=True,
                                verify_integrity=False,
                                copy=False)
        df_s_db_drop = df_s_db.drop_duplicates()


    else:
        print(" ERROR : NO SEARCH HAS BEEN PERFORMED")
        df_s_db_drop = df
    return df_s_db_drop





def run_search(search_keywords,
               extract_database=True,
               apply_precise_keyword_search=True,
               extract_extire_database=False,
               save_extracted_database_path='', do_save_csv=False, do_print=False,
               save_search_output_path='search_output.csv'):
    """

    :param search_keywords: dict
    :param extract_database: bool
    :param apply_precise_keyword_search: bool
    :param save_extracted_database_path: str
    :param save_search_output_path: str
    :return: pandas dataframe


    'Composer': list, (example: ['Composer1', 'Composer2'])
    'Movement Number': list, (example: ['Mov1', 'Mov2'])
    'Title': list, (example: ['Title1', 'Title2'])
    'Key': list,  (example: ['C', 'G'])
    'Life Time Year': list, (example: [1300, 1600, 1880])
    'Life Time Range': str, (example: '1300-1600')
    'Year Range': str, (example: '1300-1600')


    -----------------------------------------------------
    Example
    -----------------------------------------------------
    search_keywords = {'Composer': ['liszt',],
                       'Movement Number': None,
                       'Title': ['Grandes', ],
                       'Key': ['A', ],
                        'Life Time Year': None,
                        'Life Time Range': None,
                        'Year Range': None}


    df_s = run_search(search_keywords=search_keywords,
                      extract_database=False,
                      apply_precise_keyword_search=True,
                      save_extracted_database_path=database_csv_path,
                      save_search_output_path='search_output.csv')

    """
    save_extracted_database_path = os.path.join(save_extracted_database_path, 'composer_web_database.csv')

    if not os.path.isfile(save_extracted_database_path):
        print("File not found therefore scraping database", save_extracted_database_path)
        extract_database =True

    if not extract_database:
        df_entrie_database = pd.read_csv(save_extracted_database_path, sep=';')
        if df_entrie_database.empty:
            df_entrie_database = scrape_database(save_extracted_database_path, do_print=do_print)
        print("saved at ", save_extracted_database_path)

    else:
        df_entrie_database = scrape_database(save_extracted_database_path, do_print=do_print)
    df_entrie_database = df_entrie_database.replace(np.nan, '', regex=True)
    if extract_extire_database:
        df_entrie_database.reset_index(drop=True, inplace=True)
        if do_save_csv:
            df_entrie_database.to_csv(save_search_output_path, index=False,  sep=';')
        return df_entrie_database
    else:
        df_s = search_database_intern(df=df_entrie_database,
                                      search_keywords=search_keywords,
                                      apply_precise_keyword_search=apply_precise_keyword_search,
                                      do_print=do_print)
        df_s.reset_index(drop=True, inplace=True)
        if do_save_csv:
            df_s.to_csv(save_search_output_path, index=False,  sep=';')
        return df_s



if __name__=='__main__':
    search_keywords = {'Composer': ['liszt', ],
                       'Movement Number': None,
                       'Title': ['Grandes', ],
                       'Key': ['A', ],
                       'Life Time Year': None,
                       'Life Time Range': None,
                       'Year Range': None}

    # database_csv_path = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/'
    #
    database_csv_path = os.getcwd().replace('core', os.path.join('data', 'xmls_to_parse', 'hfm_database'))

    assert os.path.isdir(database_csv_path), "File not Found"

    df_s = run_search(search_keywords=search_keywords,
                      extract_database=True,
                      apply_precise_keyword_search=True,
                      extract_extire_database=False,
                      save_extracted_database_path=database_csv_path,
                      save_search_output_path='search_output.csv')

