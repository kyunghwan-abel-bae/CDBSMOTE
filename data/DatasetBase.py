import pandas as pd
import numpy as np


class DatasetBase:
    """ A base class for data preprocessing
    """

    def __init__(self, file_path):
        self.__data = pd.DataFrame()
        self.__data_label = pd.DataFrame()
        self.__file_path = file_path

    def file_path(self):
        return self.__file_path

    def set_data(self, data):
        self.__data = data

    def set_data_label(self, data_label):
        self.__data_label = data_label

    def set_file_path(self, file_path):
        self.__file_path = file_path

    def get_pandas_data(self):
        return self.__data

    def get_pandas_data_label(self):
        return self.__data_label

    def get_dummy_eps_data(self):
        ''' dummy data for an eps of DBSCAN

        :return: [np.linspace(0.001, 1, 100)]
        '''
        return np.linspace(0.001, 1, 100).tolist()

    def normalize_data(self, data):
        ''' convert data to 0~1

        :param data:
        :return:
        '''
        list_columns = data.columns.tolist()

        for column in list_columns:
            if column[len(column) - 3:] == "_Code":
                continue

            list_data_column = data[column].to_list()

            data_max = max(list_data_column)
            data_min = min(list_data_column)

            if data_max-data_min == 0:
                list_data_column_normalize = [0.5] * len(list_data_column)
            else:
                list_data_column_normalize = [(item-data_min)/(data_max-data_min) for item in list_data_column]

            del data[column]

            data_column_normalize = pd.DataFrame(list_data_column_normalize)
            data_column_normalize.columns = [column]
            data = pd.concat([data, data_column_normalize], axis=1)

        return data