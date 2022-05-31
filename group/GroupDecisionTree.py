import pandas as pd

from sklearn.tree import DecisionTreeClassifier


class GroupDecisionTree:
    """ Expert Group of Decision Tree
    """
    def __init__(self, max_depth=1):
        self.__data_exists = False

        self.__max_depth = max_depth

        self.__list_label_accuracy = []

        self.__df_x_train = pd.DataFrame()
        self.__df_y_train = pd.DataFrame()
        self.__df_predict = pd.DataFrame()

        self.__df_core_indices_by_count = pd.DataFrame()

        self.__tree = DecisionTreeClassifier(max_depth=self.__max_depth, random_state=0)

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

    def df_y_train(self):
        ''' getter of self.__df_y_train

        :return:
        '''
        return self.__df_y_train

    def df_predict(self):
        ''' getter of self.__df_predict

        :return:
        '''
        return self.__df_predict

    def max_depth(self):
        ''' getter of self.__max_depth

        :return:
        '''
        return self.__max_depth

    def set_data(self, df_x_train, df_y_train):
        ''' set data for the decision tree. Plus, it's setter of df_x_train, df_y_train

        :param df_x_train:
        :param df_y_train:
        :return:
        '''
        if len(df_y_train) <= 30:
            raise Exception("Data Size must be bigger than 30")

        if len(df_x_train) != len(df_y_train):
            raise Exception("Data row count must be same")

        self.__df_x_train = df_x_train.copy()
        self.__df_y_train = df_y_train.copy()

        self.set_data_exists(True)

    def train(self, md=4):
        ''' create a decision tree

        :param md: max_depth for a decision tree
        :return: tree's prediction(result of a decision tree) of stored train data as a pandas data frame.
        '''
        if self.data_exists() is False:
            raise Exception("No Data Set")

        self.__tree = DecisionTreeClassifier(max_depth=md, random_state=0)
        self.__tree.fit(self.__df_x_train, self.__df_y_train)

        return pd.DataFrame(self.__tree.predict(self.__df_x_train))

    def score(self, df_x, df_y):
        ''' evaluate a created decision tree

        :param df_x: x data for test
        :param df_y: y data for test
        :return: accuracy score of df_x, df_y
        '''
        return self.__tree.score(df_x, df_y)

    def quiz1_optimize(self, df_x_quiz, df_y_quiz):
        ''' optimize a stored decision tree using quiz data. As a result, it has an optimized max depth for decision tree

        :param df_x_quiz:
        :param df_y_quiz:
        :return:
        '''
        if self.data_exists() is False:
            raise Exception("No Data Set")

        max_score = 0
        for md in range(1, 10):
            self.train(md)
            score = self.score(df_x_quiz, df_y_quiz)
            if score > max_score:
                max_score = score
                self.__max_depth = md

        self.__df_predict = self.train(self.__max_depth)