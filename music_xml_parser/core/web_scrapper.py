"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""
import os
import warnings

import requests

warnings.filterwarnings("ignore", 'This pattern has match groups')


def get_file_from_server(xml_link, save_at=None):
    """
    :param xml_link:
    :param save_at:
    :return:
    """
    if save_at == None:
        curr_dir = os.getcwd()
        # try:
        #     rootdir = curr_dir.replace(os.path.basename(curr_dir), 'data')
        #     os.mkdir(rootdir)
        #     os.chdir(rootdir)
        #
        #     for sub_dir_l1 in ['exports', 'xmls_to_parse']:
        #         os.mkdir(sub_dir_l1)
        #     os.chdir(os.path.join(rootdir, 'xmls_to_parse'))
        #     for sub_dir_l2 in ['hfm_database', 'xml_pool']:
        #         os.mkdir(sub_dir_l2)
        # except:
        #     pass

        if 'core' in curr_dir:
            save_at = curr_dir.replace('core', os.path.join('data', 'xmls_to_parse', 'hfm_database'))
        elif 'ipynb' in curr_dir:
            save_at = curr_dir.replace('ipynb', os.path.join('data', 'xmls_to_parse', 'hfm_database'))
        else:
            raise Exception("Error in Saving directory")

    file_name = xml_link.split('/')[-1]
    s = os.path.join(save_at, file_name)

    if os.path.isfile(s):
        return s
    else:
        print(f"Downloading file:{file_name}")
        response = requests.get(xml_link)
        with open(s, 'wb') as file:
            file.write(response.content)
        print(">>> %s downloaded!\n" % file_name)
        return s


def get_files_from_server(xml_link, save_at=None):
    """

    :param xml_link:
    :param save_at:
    :return:
    """
    if type(xml_link) == str:
        xml_link = [xml_link, ]
    if save_at == None:
        save_at = os.getcwd().replace('core', os.path.join('data', 'xmls_to_parse', 'hfm_database'))

    saved_links = []
    t = len(xml_link)

    for i, link in enumerate(xml_link):
        file_name = link.split('/')[-1]
        s = save_at + file_name

        if os.path.isfile(s):
            saved_links.append(s)
            continue
        else:
            print(f"{i} /{t}  Downloading file:{file_name}")
            response = requests.get(link)
            with open(s, 'wb') as file:
                file.write(response.content)
            saved_links.append(s)
            print(">>> %s downloaded!\n" % file_name)

    return saved_links
