from DatasetBase import *

dict_code_class = {
    'Hernia': 0,
    'Spondylolisthesis': 1,
    'Normal': 2
}

class DataBFOP(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """
    def __init__(self, file_path):
        super(DataBFOP, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_class = data['class'].to_list()
        list_code_class = []

        for class_item in list_str_class:
            code = dict_code_class[class_item]
            if code is not None:
                list_code_class.append(code)

        df_class = pd.DataFrame(list_code_class)
        df_class.columns = ['class_Code']

        data = pd.concat([data, df_class], axis=1)
        data.reset_index()

        label = data['class_Code']

        del data['class']
        del data['class_Code']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))