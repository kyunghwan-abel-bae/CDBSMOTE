from DatasetBase import *

dict_code_diagnosis = {
    'M': 0,
    'B': 1
}


class DataBreastCancerDiagnostic(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """

    def __init__(self, file_path):
        super(DataBreastCancerDiagnostic, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_diagnosis = data['diagnosis'].to_list()
        list_code_diagnosis = []

        for diagnosis in list_str_diagnosis:
            code = dict_code_diagnosis[diagnosis]
            if code is not None:
                list_code_diagnosis.append(code)

        df_diagnosis = pd.DataFrame(list_code_diagnosis)
        df_diagnosis.columns = ["diagnosis_Code"]

        data = pd.concat([data, df_diagnosis], axis=1)
        data.reset_index()

        label = data['diagnosis_Code']

        del data['id']
        del data['diagnosis']
        del data['diagnosis_Code']
        del data['Unnamed: 32']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))
