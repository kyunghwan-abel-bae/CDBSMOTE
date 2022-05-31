import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Data processing
from data.DataStars import *
from data.DataGlass import *


def calc_macro_average(list_data):
    ''' getting a macro average for a precision and a recall

    :param list_data: [[elem1, elem2, elem3], [elem4, elem5, elem6], ...]
    :return:
    '''

    if len(list_data) == 0:
        return None

    total = 0
    for index, list_row in enumerate(list_data):
        tp = 0
        fp = 0
        for r_index, data in enumerate(list_row):
            if index == r_index:
                tp = data
            else:
                fp = fp + data

        if (tp + fp) != 0:
            total = total + (tp/(tp+fp))

    result = total / len(list_data)

    return result


def calc_f1_score_for_dbscan_result(pd_table):
    ''' calculate a f1-score for a pandas table

    :param pd_table: a pandas table for a confusion matrix
    :return:
    '''

    list_row_data = pd_table.values.tolist()
    calculated_recall = calc_macro_average(list_row_data)

    # Transposition and store data as a list
    list_col_data = [pd_table[n].tolist() for n in np.arange(0, len(pd_table.columns))]
    calculated_precision = calc_macro_average(list_col_data)

    calculated_f1_score = -1
    if calculated_precision > 0 and calculated_recall > 0:
        calculated_f1_score = 2 * (calculated_precision * calculated_recall) / (calculated_precision + calculated_recall)

    return calculated_f1_score


def plot_3rd_pca(graph, title, df_data, df_label):
    ''' plotting multidimensional data as a 3rd dimension graph

    :param graph:
    :param title:
    :param df_data:
    :param df_label:
    :return:
    '''
    graphic_data = df_data
    graphic_data_scaled = StandardScaler().fit_transform(graphic_data)

    pca = PCA(n_components=3)

    pca.fit(graphic_data_scaled)
    graphic_data_pca = pca.transform(graphic_data_scaled)

    pca_columns = ['pca_component_1', 'pca_component_2', 'pca_component_3']
    graphic_df_pca = pd.DataFrame(graphic_data_pca, columns=pca_columns)

    graph.rect = [0, 0, .95, 1]
    graph.elev = 48
    graph.azim = 134

    clear_labels_for_c = np.array(df_label.values).flatten().tolist()

    graph.scatter(graphic_df_pca['pca_component_1'], graphic_df_pca['pca_component_2'],
                        graphic_df_pca['pca_component_3'], c=clear_labels_for_c, alpha=0.5)

    graph.set_xlabel('pca_component_1')
    graph.set_ylabel('pca_component_2')
    graph.set_zlabel('pca_component_3')
    graph.set_title(title)

################ GROUP DECLARED

group_dt = GroupDecisionTree()
group_dbscan = GroupDBSCAN()

########## TARGET DATA
obj_data = DataStars('csv/Stars.csv')
# obj_data = DataGlass('csv/glass.csv')
# obj_data = DataIris('csv/Iris.csv')
# obj_data = DataBreastCancerDiagnostic('csv/breast_cancer_diagnostic.csv')
# obj_data = DataWineQualityRed('csv/winequality-red.csv')
# obj_data = DataZoo('csv/zoo.csv') # NO BORDER DATA
# obj_data = DataBFOP('csv/biomechanical_features_of_orthopedic_patients.csv')
# obj_data = DataAbalone('csv/abalone_original.csv') # too many classes
# obj_data = DataBank('csv/BankNoteAuthentication.csv')
# obj_data = DataDiabetes('csv/diabetes.csv')
# obj_data = DataDryBean('csv/Dry_Bean.csv')
# obj_data = DataHeart('csv/heart.csv')

df = obj_data.get_pandas_data()
df_label = obj_data.get_pandas_data_label()

dict_labels_value_counts = df_label.value_counts().to_dict()

########## FIND MIN/MAX VALUES (FOR REGULARIZATION) & DISTINGUISH NUMERIC CATEGORIES
list_columns = df.columns.tolist()
list_is_category_code = [] # 0 : not code, 1: code
list_columns_max = []
list_columns_min = []

for column in list_columns:
    if column[len(column) - 5:] == "_Code":
        list_is_category_code.append(1)
    else:
        list_is_category_code.append(0)

    data_column = df[column].to_list()

    list_columns_max.append(max(data_column))
    list_columns_min.append(min(data_column))

################ EXPERT GROUP(A) - DECISION TREE
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df, df_label, test_size=0.3, random_state=11)
df_x_test, df_x_quiz, df_y_test, df_y_quiz = train_test_split(df_x_test, df_y_test, test_size=0.3, random_state=11)

df_x_train = df_x_train.reset_index(drop=True)
df_x_test = df_x_test.reset_index(drop=True)
df_x_quiz = df_x_quiz.reset_index(drop=True)

df_y_train = df_y_train.reset_index(drop=True)
df_y_test = df_y_test.reset_index(drop=True)
df_y_quiz = df_y_quiz.reset_index(drop=True)


tree = DecisionTreeClassifier(max_depth=4, random_state=0)

tree.fit(df_x_train, df_y_train)
print("======================")
print("PURE DECISION TREE")
print("[Tree] Accuracy for the train set: {:.3f}".format(tree.score(X_train, y_train)))
print("[Tree] Accuracy for the test set: {:.3f}".format(tree.score(X_test, y_test)))
print("======================")

quit()


fig = plt.figure(figsize=(15,7))
first_graph = fig.add_subplot(221, projection='3d')

title_first_graph = "DECISION TREE(Accuracy : {:.3f})".format(tree.score(X_test, y_test))

plot_3rd_pca(first_graph, title_first_graph, X_train, y_train)

################ EXPERT GROUP(B) - DBSCAN
df_y_train = pd.DataFrame(y_train)
df_y_train.columns = ['labels']

list_labels = df_y_train.drop_duplicates('labels')['labels'].tolist()
num_labels = len(list_labels)

count_min_category = df_y_train['labels'].value_counts().min()

eps_data = obj_data.get_dummy_eps_data()

if count_min_category > 3:
    min_samples_data = np.arange(3, (count_min_category+1), 1)
else:
    min_samples_data = [3]

'''
Finding proper eps and min_samples
- min_samples : high -> low
- eps : narrow -> wide
'''
result_dbscan = None

# find a first cluster which fits to the data classes
for m in reversed(min_samples_data):
    for e in eps_data:
        # DBSCAN Options
        model = DBSCAN(eps=e, min_samples=m)
        # model = DBSCAN(eps=e, min_samples=m, algorithm='auto', metric="dice", leaf_size=90, p=2)
        # model = DBSCAN(eps=e, min_samples=m, algorithm='auto', metric="rogerstanimoto", leaf_size=90, p=2)
        # model = DBSCAN(eps=e, min_samples=m, algorithm='auto', metric="sokalmichener", leaf_size=90, p=2)
        # model = DBSCAN(eps=e, min_samples=m, algorithm='auto', metric="sokalsneath", leaf_size=90, p=2)
        predict = pd.DataFrame(model.fit_predict(X_train))
        predict_duplication_dropped = predict.drop_duplicates()
        if len(predict_duplication_dropped) == (num_labels+1):
            result_dbscan = predict
            print("--FOUND CLUSTER--")
            print("e : ", e, "min_samples : ", m)
            break

    if result_dbscan is not None:
        break

if result_dbscan is None:
    print("--DBSCAN Failed--")
    quit()

result_dbscan.columns = ['predict']

########## FINDING BORDER ITEMS AT THE EACH CLUSTERS
not_noisy_data = result_dbscan.loc[result_dbscan['predict'] != -1]
indices_core = model.core_sample_indices_
indices_border = list(not_noisy_data.index.difference(indices_core))

if len(indices_border) == 0:
    print("--NO BORDER ITEM--")
    quit()

X_train_border = X_train.loc[indices_border]
y_train_border = y_train.loc[indices_border]
predict_border = not_noisy_data.loc[indices_border]

data_border = pd.concat([X_train_border, y_train_border, predict_border], axis=1)
data_border = data_border.reset_index()

data_not_noisy = pd.concat([X_train, y_train, result_dbscan], axis=1)
data_not_noisy = data_not_noisy.loc[data_not_noisy['predict'] != -1]

data_not_noisy_X = data_not_noisy[list_columns]
data_not_noisy_predict = data_not_noisy[['predict']]

second_graph = fig.add_subplot(222, projection='3d')
plot_3rd_pca(second_graph, "DBSCAN(without noise)", data_not_noisy_X, data_not_noisy_predict)

################ DATA AUGMENTATION PROCESS(1) ~ Finding the closest item of the each cluster at the border data
data_temp = data_not_noisy.groupby('predict')
list_found_clusters = [data_temp.get_group(x).reset_index() for x in data_temp.groups]

dict_closest_data = {}

index_border = 0
while index_border <= data_border.index[-1]:
    list_closest_data = []
    for cluster in list_found_clusters:
        index_row = 0
        if cluster.loc[index_row, 'predict'] == data_border.loc[index_border, 'predict']:
            index_row = index_row + 1
        else:
            shortest_dist = int(2147483647) # maximum integer value

            index_shortest_dist_row = -1

            while index_row <= cluster.index[-1]:
                dist_total = 0
                for index, column in enumerate(list_columns):
                    if column[len(column) - 5:] == "_Code":
                        continue

                    dist = cluster.loc[index_row, column] - data_border.loc[index_border, column]
                    dist_total = dist_total + dist

                if dist_total < shortest_dist:
                    index_shortest_dist_row = index_row
                    shortest_dist = dist_total

                index_row = index_row + 1

            list_closest_data.append(cluster.loc[index_shortest_dist_row])

    dict_closest_data[data_border.loc[index_border, 'index']] = list_closest_data
    index_border = index_border + 1

################ DATA AUGMENTATION PROCESS(2) ~ Data augemtation using closest items
list_augmented_container = []
list_total_container = []

list_key_closest_data = list(dict_closest_data.keys())
df_augmented_data = pd.DataFrame()

list_closest_data_labels = []

# The key below is the index of borders. border items are important because the data is augmented in every border items
# So, the below loop makes the sum of the closest items properties, and the case of code items is just an appended list
# Plus, the label of the closest data are stored as a list
# result : list_total_container, list_closest_data_labels
for key in list_key_closest_data:
    list_cluster_closest_data = dict_closest_data[key]

    list_cluster_closest_data_code_total = []
    list_cluster_closest_data_not_code_total = []
    for index, column in enumerate(list_columns):
        if column[len(column) - 5:] == "_Code":
            list_cluster_closest_data_code_total.append([])
        else:
            list_cluster_closest_data_not_code_total.append(0)

    list_closest_data_labels_temp = []
    for closest_data in list_cluster_closest_data:
        count_code = 0
        count_not_code = 0

        for column in list_columns:
            if column[len(column) - 5:] == "_Code":
                list_cluster_closest_data_code_total[count_code].append(closest_data.loc[column])
                count_code = count_code + 1
            else:
                list_cluster_closest_data_not_code_total[count_not_code] = list_cluster_closest_data_not_code_total[count_not_code] + closest_data.loc[column]
                count_not_code = count_not_code + 1

        list_closest_data_labels_temp.append(closest_data['labels'])

    list_closest_data_labels.append(list_closest_data_labels_temp)

    list_temp = []
    count_code = 0
    count_not_code = 0
    for column in list_columns:
        if column[len(column) - 5:] == "_Code":
            list_temp.append(list_cluster_closest_data_code_total[count_code])
            count_code = count_code + 1
        else:
            list_temp.append(list_cluster_closest_data_not_code_total[count_not_code])
            count_not_code = count_not_code + 1

    list_total_container.append(list_temp)

# properties of data augmentation are prepared at the 'list_augmented_container'
list_augmented_container = []
for list_total in list_total_container:
    list_temp = []

    for index, column in enumerate(list_columns):
        if column[len(column) - 5:] == "_Code":
            list_total[index] = int(max(list_total[index], key=list_total[index].count))
        else:
            divisor = (len(list_found_clusters)-1)
            if divisor == 0:
                divisor = 1
            list_total[index] = list_total[index] / divisor

    list_augmented_container.append(list_total)

df_augmented_data = pd.DataFrame(list_augmented_container, columns=list_columns)

# labels of data augmentation are prepared at the 'list_augmented_container'
list_augmented_labels = []
for closest_data_labels in list_closest_data_labels:
    label_min = None
    count_min = int(2147483647) # maximum integer value
    for label_item in closest_data_labels:
        label_item = int(label_item)
        if dict_labels_value_counts[label_item] < count_min:
            count_min = dict_labels_value_counts[label_item]
            label_min = label_item

    dict_labels_value_counts[label_min] = dict_labels_value_counts[label_min] + 1
    list_augmented_labels.append(label_min)

df_augmented_data_label = pd.DataFrame(list_augmented_labels)

X_train_augmented = pd.concat([X_train, df_augmented_data])
y_train_augmented = pd.concat([y_train, df_augmented_data_label])

X_train_augmented = X_train_augmented.reset_index(drop=True)
y_train_augmented = y_train_augmented.reset_index(drop=True)

third_graph = fig.add_subplot(223, projection='3d')
third_graph.set_title("AUGMENTED")

plot_3rd_pca(third_graph, "AUGMENTED", X_train_augmented, y_train_augmented)

################ RESULT - DECISION TREE WITH AUGMENTED DATA
tree.fit(X_train_augmented, y_train_augmented)
print("======================")
print("DECISION TREE(with augmented data)")
print("[Tree] Accuracy for the train set: {:.3f}".format(tree.score(X_train_augmented, y_train_augmented)))
print("[Tree] Accuracy for the test set: {:.3f}".format(tree.score(X_test, y_test)))
print("======================")

fourth_graph = fig.add_subplot(224, projection='3d')

title_fourth_graph = "RESULT(Accuracy : {:.3f})".format(tree.score(X_test, y_test))

plot_3rd_pca(fourth_graph, title_fourth_graph, X_test, y_test)

plt.show()
