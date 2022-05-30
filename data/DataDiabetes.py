from DatasetBase import *

dict_code_Gender = {
    'Male': 0,
    'Female': 1
}

dict_code_Polyuria = {
    'Yes': 0,
    'No': 1
}

dict_code_Polydipsia = {
    'Yes': 0,
    'No': 1
}

dict_code_SWL = {
    'Yes': 0,
    'No': 1
}
dict_code_weakness = {
    'Yes': 0,
    'No': 1
}

dict_code_Polyphagia = {
    'Yes': 0,
    'No': 1
}

dict_code_GT = {
    'Yes': 0,
    'No': 1
}

dict_code_VB = {
    'Yes': 0,
    'No': 1
}

dict_code_Itching = {
    'Yes': 0,
    'No': 1
}

dict_code_Irritability = {
    'Yes': 0,
    'No': 1
}

dict_code_DH = {
    'Yes': 0,
    'No': 1
}

dict_code_PP = {
    'Yes': 0,
    'No': 1
}

dict_code_MS = {
    'Yes': 0,
    'No': 1
}

dict_code_Alopecia = {
    'Yes': 0,
    'No': 1
}

dict_code_Obesity = {
    'Yes': 0,
    'No': 1
}

dict_code_class = {
    'Positive': 0,
    'Negative': 1
}


class DataDiabetes(DatasetBase):
    """ For handling data(UCI Repository of Machine Learning Databases [Online] - https://archive.ics.uci.edu/ml/index.php)
    """

    def __init__(self, file_path):
        super(DataDiabetes, self).__init__(file_path)

        self.init()

    def init(self):
        data = pd.read_csv(self.file_path())

        list_str_Gender = data['Gender'].to_list()
        list_str_Polyuria = data['Polyuria'].to_list()
        list_str_Polydipsia = data['Polydipsia'].to_list()
        list_str_SWL = data['sudden weight loss'].to_list()
        list_str_weakness = data['weakness'].to_list()
        list_str_Polyphagia = data['Polyphagia'].to_list()
        list_str_GT = data['Genital thrush'].to_list()
        list_str_VB = data['visual blurring'].to_list()
        list_str_Itching = data['Itching'].to_list()
        list_str_Irritability = data['Irritability'].to_list()
        list_str_DH = data['delayed healing'].to_list()
        list_str_PP = data['partial paresis'].to_list()
        list_str_MS = data['muscle stiffness'].to_list()
        list_str_Alopecia = data['Alopecia'].to_list()
        list_str_Obesity = data['Obesity'].to_list()
        list_str_class = data['class'].to_list()

        list_code_Gender = []
        list_code_Polyuria = []
        list_code_Polydipsia = []
        list_code_SWL = []
        list_code_weakness = []
        list_code_Polyphagia = []
        list_code_GT = []
        list_code_VB = []
        list_code_Itching = []
        list_code_Irritability = []
        list_code_DH = []
        list_code_PP = []
        list_code_MS = []
        list_code_Alopecia = []
        list_code_Obesity = []
        list_code_class = []

        for gender in list_str_Gender:
            code = dict_code_Gender[gender]
            if code is not None:
                list_code_Gender.append(code)

        for polyuria in list_str_Polyuria:
            code = dict_code_Polyuria[polyuria]
            if code is not None:
                list_code_Polyuria.append(code)

        for polydipsia in list_str_Polydipsia:
            code = dict_code_Polydipsia[polydipsia]
            if code is not None:
                list_code_Polydipsia.append(code)

        for swl in list_str_SWL:
            code = dict_code_SWL[swl]
            if code is not None:
                list_code_SWL.append(code)

        for w in list_str_weakness:
            code = dict_code_SWL[w]
            if code is not None:
                list_code_weakness.append(code)

        for polyphagia in list_str_Polyphagia:
            code = dict_code_Polyphagia[polyphagia]
            if code is not None:
                list_code_Polyphagia.append(code)

        for gt in list_str_GT:
            code = dict_code_GT[gt]
            if code is not None:
                list_code_GT.append(code)

        for vb in list_str_VB:
            code = dict_code_VB[vb]
            if code is not None:
                list_code_VB.append(code)

        for itching in list_str_Itching:
            code = dict_code_Itching[itching]
            if code is not None:
                list_code_Itching.append(code)

        for irritability in list_str_Irritability:
            code = dict_code_Irritability[irritability]
            if code is not None:
                list_code_Irritability.append(code)

        for dh in list_str_DH:
            code = dict_code_DH[dh]
            if code is not None:
                list_code_DH.append(code)

        for pp in list_str_PP:
            code = dict_code_PP[pp]
            if code is not None:
                list_code_PP.append(code)

        for ms in list_str_MS:
            code = dict_code_MS[ms]
            if code is not None:
                list_code_MS.append(code)

        for alopecia in list_str_Alopecia:
            code = dict_code_Alopecia[alopecia]
            if code is not None:
                list_code_Alopecia.append(code)

        for obesity in list_str_Obesity:
            code = dict_code_Obesity[obesity]
            if code is not None:
                list_code_Obesity.append(code)

        for c in list_str_class:
            code = dict_code_class[c]
            if code is not None:
                list_code_class.append(code)

        df_gender = pd.DataFrame(list_code_Gender)
        df_gender.columns = ["gender_Code"]

        df_polyuria = pd.DataFrame(list_code_Polyuria)
        df_polyuria.columns = ["polyuria_Code"]

        df_polydipsia = pd.DataFrame(list_code_Polydipsia)
        df_polydipsia.columns = ["polydipsia_Code"]

        df_swl = pd.DataFrame(list_code_SWL)
        df_swl.columns = ["swl_Code"]

        df_weakness = pd.DataFrame(list_code_weakness)
        df_weakness.columns = ["weakness_Code"]

        df_polyphagia = pd.DataFrame(list_code_Polyphagia)
        df_polyphagia.columns = ["polyphagia_Code"]

        df_gt = pd.DataFrame(list_code_GT)
        df_gt.columns = ["gt_Code"]

        df_vb = pd.DataFrame(list_code_VB)
        df_vb.columns = ["vb_Code"]

        df_itching = pd.DataFrame(list_code_Itching)
        df_itching.columns = ["itching_Code"]

        df_irritability = pd.DataFrame(list_code_Irritability)
        df_irritability.columns = ["irritability_Code"]

        df_dh = pd.DataFrame(list_code_DH)
        df_dh.columns = ["dh_Code"]

        df_pp = pd.DataFrame(list_code_PP)
        df_pp.columns = ["pp_Code"]

        df_ms = pd.DataFrame(list_code_MS)
        df_ms.columns = ["ms_Code"]

        df_alopecia = pd.DataFrame(list_code_Alopecia)
        df_alopecia.columns = ["alopecia_Code"]

        df_obecity = pd.DataFrame(list_code_Obesity)
        df_obecity.columns = ["obecity_Code"]

        df_class = pd.DataFrame(list_code_class)
        df_class.columns = ["class_Code"]

        data = pd.concat([data, df_gender, df_polyuria, df_polydipsia, df_swl, df_weakness, df_polyphagia, df_gt, df_vb, df_itching, df_irritability, df_dh, df_pp, df_ms, df_alopecia, df_obecity, df_class], axis=1)
        data.reset_index()

        label = data['class_Code']

        del data['Gender']
        del data['Polyuria']
        del data['Polydipsia']
        del data['sudden weight loss']
        del data['weakness']
        del data['Polyphagia']
        del data['Genital thrush']
        del data['visual blurring']
        del data['Itching']
        del data['Irritability']
        del data['delayed healing']
        del data['partial paresis']
        del data['muscle stiffness']
        del data['Alopecia']
        del data['Obesity']
        del data['class']

        self.set_data(self.normalize_data(data))
        self.set_data_label(label.rename('labels'))