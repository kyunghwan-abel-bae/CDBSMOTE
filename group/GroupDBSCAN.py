import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN


class GroupDBSCAN:
    """ Expert Group of DBSCAN
    """
    def __init__(self, eps=0.1, min_samples=3):
        self.__data_exists = False

        self.__eps = eps
        self.__min_samples = min_samples

        self.__list_border = []
        self.__list_cluster = [] # cluster == not noisy data == core + border

        self.__list_label_accuracy = []

        self.__df_x_train = pd.DataFrame()
        self.__df_predict = pd.DataFrame()

        self.__dbscan = DBSCAN(eps=self.__eps, min_samples=self.__min_samples)

    def set_data_exists(self, value):
        ''' setter of self.__data_exists

        :param value:
        :return:
        '''
        self.__data_exists = value

    def data_exists(self):
        ''' getter of self.__data_exists

        :return:
        '''
        return self.__data_exists

    def df_x_train(self):
        ''' getter of self.__df_x_train

        :return:
        '''
        return self.__df_x_train

    def list_border(self):
        ''' getter of self.__list_border

        :return:
        '''
        return self.__list_border

    def list_cluster(self):
        ''' getter of self.__list_cluster

        :return:
        '''
        return self.__list_cluster

    def set_data(self, df_x_train):
        ''' set data for the dbscan. Plus, it's setter of df_x_train

        :param df_x_train:
        :return:
        '''
        if len(df_x_train) <= 30:
            raise Exception("Data Size must be bigger than 30")

        self.__df_x_train = df_x_train.copy()

        self.set_data_exists(True)

    def train(self, e, m):
        ''' create a dbscan

        :param e: eps
        :param m: min_samples
        :return: dbscan's prediction(result cluster of dbscan)
        '''
        if self.data_exists() is False:
            raise Exception("No Data Set")

        self.__dbscan = DBSCAN(eps=e, min_samples=m)
        self.__dbscan.fit(self.__df_x_train)

        return pd.DataFrame(self.__dbscan.fit_predict(self.__df_x_train))

    def quiz_optimize(self, df_x_quiz, df_y_quiz):
        ''' found all clusters which meet the number of labels(of df_y_quiz)

        :param df_x_quiz:
        :param df_y_quiz:
        :return:
        '''
        if self.data_exists() is False:
            raise Exception("No Data Set")

        list_eps = np.linspace(0.001, 1, 100).tolist()
        list_min_samples = np.arange(3, int(len(self.__df_x_train)*0.1))

        self.__list_border = []
        self.__list_cluster = []
        for m in list_min_samples:
            for e in list_eps:
                df_predict = self.train(e, m)
                if len(df_predict.drop_duplicates()) == (len(df_y_quiz.drop_duplicates())+1):
                    df_predict.columns = ['predict']

                    df_not_noisy_result = df_predict.loc[df_predict['predict'] != -1]
                    indices_core = self.__dbscan.core_sample_indices_
                    indices_border = list(df_not_noisy_result.index.difference(indices_core))

                    if len(indices_border) == 0:
                        print("No border data")
                        continue

                    df_result = pd.concat([self.__df_x_train, df_predict], axis=1)

                    df_border = df_result.loc[indices_border]
                    df_cluster = df_result.loc[df_not_noisy_result.index]

                    self.__list_border.append(df_border)
                    self.__list_cluster.append(df_cluster)