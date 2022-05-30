
from DatasetBase import *

dict_code_class = {
    'DERMASON': 0,
    'SIRA': 1,
    'SEKER': 2,
    'HOROZ': 3,
    'CALI': 4,
    'BARBUNYA': 5,
    'BOMBAY': 6
}


class DataDryBean(DatasetBase):
    def __init__(self, file_path):
        super(DataDryBean, self).__init__(file_path)
        self.init()

    def init(self):
        data = pd.read_csv(self.file_path_)

        list_str_class = data['Class'].to_list()
        list_code_class = []

        for c in list_str_class:
            code = dict_code_class[c]
            if code is not None:
                list_code_class.append(code)

        df_class = pd.DataFrame(list_code_class)
        df_class.columns = ["class_Code"]

        data = pd.concat([data, df_class], axis=1)

        data.reset_index()

        label = data['class_Code']

        del data['Class']

        self.data_ = self.normalize_data(data)
        self.data_label_ = label.rename('labels')