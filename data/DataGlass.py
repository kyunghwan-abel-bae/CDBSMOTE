import pandas as pd
import numpy as np

from DatasetBase import *

class DataGlass(DatasetBase):
    def __init__(self, file_path):
        super(DataGlass, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path_)
        label = data['Type']

        del data['Type']

        self.data_ = self.normalize_data(data)
        self.data_label_ = label.rename('labels')