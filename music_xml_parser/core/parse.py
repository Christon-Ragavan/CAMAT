"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""

try:
    from .parser_utils import _get_file_path, set_up_logger
    from .plot import pianoroll_parts,_create_pianoroll_single_parts
    from .analyse import *

    from .xml_parser import XMLParser
except:
    from parser_utils import _get_file_path, set_up_logger
    from plot import pianoroll_parts, _create_pianoroll_single_parts
    from analyse import *
    from xml_parser import XMLParser
from os.path import isdir, basename, isfile,join

np.seterr(all="ignore")


logger = set_up_logger(__name__)


@pianoroll_parts
def with_xml_file(file: str,
                  plot_pianoroll: bool = False,
                  plot_inline_ipynb: bool=True,
                  save_at: str = None,
                  save_file_name: str = None,
                  do_save: bool = False,
                  x_axis_res=2,
                  get_measure_onset:bool=False,
                  get_upbeat_info=False,
                  filter_dict=None,
                  *args,
                  **kwargs):

    def get_parts(df):
        n_df = df[['PartID', 'PartName']].to_numpy(dtype=str)
        u = np.unique(n_df, axis=0)
        pn = [str(i[1]) + '-' + str(i[0]) for idx, i in enumerate(u)]
        pid = [int(i)for i in u[:,0]]
        return pid, pn

    file = _get_file_path(file=file)

    if '.xml' not in basename(file):
        e = "Not a .XML file, Only .xml file is supported. Use Musescore or finale to convert to .xml format"
        logger.error(e)
        raise Exception(e)
    save_at_fn = ''

    if do_save:
        if save_file_name is not None:
            if '.csv' not in basename(save_file_name):
                e = f"{save_file_name} format not supported. File name must end with '.csv'. E.g. file_name.csv"
                logger.error(e)
                raise Exception(e)
        if save_at is None:
            save_at = join(os.getcwd().replace(basename(os.getcwd()), 'data'), 'exports')
            if isdir(save_at) == False: os.makedirs(save_at)
            if save_file_name is None:
                save_file_name = basename(file).replace('.xml', '.csv')
            save_at_fn = join(save_at, save_file_name)
        else:
            if save_file_name is None:
                save_file_name = basename(file).replace('.xml', '.csv')
            save_at_fn = join(save_at, save_file_name)

    logger.info("Extracting")
    parser_o = XMLParser(path=file, logger=logger)
    df_xml = parser_o.xml_parse()
    t_pid, t_pn = get_parts(df_xml)
    logger.info("Successful")
    if filter_dict is not None:
        df_xml = filter(df_xml, filter_dict)

    # upbeat_info = parser_o.upbeat_measure_info
    upbeat_info = []

    measure_onset_data = parser_o._compute_measure_n_onset()
    if do_save:
        df_xml.to_csv(save_at_fn, sep=';')
    return df_xml, \
           plot_pianoroll,\
           parser_o.measure_onset_list,\
           upbeat_info,\
           x_axis_res,\
           get_measure_onset, \
           measure_onset_data, \
           plot_inline_ipynb, t_pid,\
           t_pn


if __name__=='__main__':
    xml_file = 'https://analyse.hfm-weimar.de/database/04/BaJoSe_BWV7_COM_7-7_CantataChr_004_00043.xml'
    m_df = with_xml_file(file=xml_file,
                                  do_save=False)
    print(m_df)