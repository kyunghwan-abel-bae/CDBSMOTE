from DatasetBase import *

dict_code_species = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}

class DataIris(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """

    def __init__(self, file_path):
        super(DataIris, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_species = data['Species'].to_list()
        list_code_species = []

        for species in list_str_species:
            code = dict_code_species[species]
            if code is not None:
                list_code_species.append(code)

        df_species = pd.DataFrame(list_code_species)
        df_species.columns = ["Species_Code"]

        data = pd.concat([data, df_species], axis=1)
        data.reset_index()

        label = data['Species_Code']

        del data['Id']
        del data['Species']
        del data['Species_Code']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))
