from DatasetBase import *


class DataHeart(DatasetBase):
    def __init__(self, file_path):
        super(DataHeart, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path_)

        label = data['target']

        del data['target']

        self.data_ = self.normalize_data(data)
        self.data_label_ = label.rename('labels')
