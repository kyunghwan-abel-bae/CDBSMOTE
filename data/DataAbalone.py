from DatasetBase import *

dict_code_sex = {
    'M': 0,
    'I': 1,
    'F': 2
}


class DataAbalone(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """

    def __init__(self, file_path):
        super(DataAbalone, self).__init__(file_path)
        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_sex = data['sex'].to_list()
        list_code_sex = []

        for sex in list_str_sex:
            code = dict_code_sex[sex]
            if code is not None:
                list_code_sex.append(code)

        df_sex = pd.DataFrame(list_code_sex)
        df_sex.columns = ["Sex_Code"]

        data = pd.concat([data, df_sex], axis=1)

        data.reset_index()

        label = data['rings']

        del data['sex']
        del data['rings']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))