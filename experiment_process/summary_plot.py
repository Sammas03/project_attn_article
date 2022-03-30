from typing import Union, List, Dict

from util import dataframe_reader as dfr


def summary_data_plot(predict_col: Union[str, int], result_files: Dict[str, List]):
    link = {}
    for key, value in result_files.item():
        df = dfr.easy_read_data(value)
        link[key] = df[predict_col] if isinstance(predict_col, str) else df.iloc[:, predict_col]

    # plot


if __name__ == '__main__':
    summary_data_plot('hello', [''])
