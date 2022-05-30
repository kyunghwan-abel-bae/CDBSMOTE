from DatasetBase import *

dict_code_color = {
    'Red': 0,
    'Blue White': 1, 'Blue white': 1, 'Blue-White': 1, 'Blue-white': 1,
    'White': 2, 'Whitish': 2, 'white': 2,
    'Yellowish White': 3, 'yellow-white': 3, 'White-Yellow': 3, 'yellowish': 3, 'Yellowish': 3,
    'Pale yellow orange': 4,
    'Blue': 5,
    'Orange': 6,
    'Orange-Red': 7
}

dict_code_spectral_class = {
    'M': 0,
    'B': 1,
    'A': 2,
    'F': 3,
    'O': 4,
    'K': 5,
    'G': 6
}


class DataStars(DatasetBase):
    """ For handling data(Star Type Classification/NASA - https://www.kaggle.com/brsdincer/star-type-classification)
    """

    def __init__(self, file_path):
        super(DataStars, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_color = data['Color'].to_list()
        list_code_color = []

        for color in list_str_color:
            code = dict_code_color[color]
            if code is not None:
                list_code_color.append(code)

        list_str_spectral_class = data['Spectral_Class'].to_list()
        list_code_spectral_class = []

        for spec in list_str_spectral_class:
            code = dict_code_spectral_class[spec]
            if code is not None:
                list_code_spectral_class.append(code)

        df_color = pd.DataFrame(list_code_color)
        df_color.columns = ["Color_Code"]
        df_spec = pd.DataFrame(list_code_spectral_class)
        df_spec.columns = ["Spectral_Class_Code"]

        data = pd.concat([data, df_color, df_spec], axis=1)

        data.reset_index()

        label = data['Type']

        del data['Color']
        del data['Spectral_Class']
        del data['Type']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))
