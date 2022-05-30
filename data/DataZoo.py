from DatasetBase import *


class DataZoo(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """

    def __init__(self, file_path):
        super(DataZoo, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())
        label = data['class_type']

        del data['animal_name']
        del data['class_type']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))