import itertools

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Expert groups
from group.GroupDecisionTree import *
from group.GroupDBSCAN import *

# Data processing
from data.DataStars import *
from data.DataGlass import *
from data.DataIris import *
from data.DataBFOP import *
from data.DataBreastCancerDiagnostic import *
from data.DataWineQualityRed import *
from data.DataZoo import *
from data.DataAbalone import *
from data.DataBank import *
from data.DataDiabetes import *
from data.DataDryBean import *
from data.DataHeart import *

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

'''
######################################################
(A)
'''
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

group_dt.set_data(df_x_train, df_y_train)
group_dbscan.set_data(df_x_train)

'''
######################################################
(B)
'''

# group_dt.train(1)
# group_dbscan.train(0.1, 3)


'''
######################################################
(C)
'''

random_df_x_train = df_x_train.sample(frac=0.3, random_state=60)
random_df_y_train = df_y_train.loc[random_df_x_train.index.tolist()]

df_x_quiz = pd.concat([df_x_quiz, random_df_x_train])
df_y_quiz = pd.concat([df_y_quiz, random_df_y_train])

df_x_quiz = df_x_quiz.reset_index(drop=True)
df_y_quiz = df_y_quiz.reset_index(drop=True)

group_dt.quiz_optimize(df_x_quiz, df_y_quiz)
group_dbscan.quiz_optimize(df_x_quiz, df_y_quiz)

fig = plt.figure(figsize=(15, 7))

first_graph = fig.add_subplot(121, projection='3d')
plot_3rd_pca(first_graph, "Expert Group(A)", group_dt.df_x_train(), group_dt.df_predict())

'''
######################################################
(D) Mutual teaching
'''

list_border = group_dbscan.list_border()
list_cluster = group_dbscan.list_cluster()

f1_score_max = -1
table_f1_score_max = pd.DataFrame()
cluster_score_max = pd.DataFrame()
border_score_max = pd.DataFrame()

df_dt_y_predict = group_dt.df_predict().copy()
for index_cluster, cluster in enumerate(list_cluster):
    temp_y_predict = df_dt_y_predict.copy()
    temp_y_predict = temp_y_predict.loc[cluster.index]

    temp_cluster_predict = cluster['predict']

    temp_y_predict = temp_y_predict.reset_index(drop=True)
    temp_cluster_predict = temp_cluster_predict.reset_index(drop=True)

    c_table = confusion_matrix(temp_y_predict, temp_cluster_predict)
    pd_table = pd.DataFrame(c_table)
    pd_table_columns = list(pd_table.columns.values)

    for temp_column in list(itertools.permutations(pd_table_columns, len(pd_table_columns))):
        pd_table = pd_table[list(temp_column)]
        temp_f1_score = calc_f1_score_for_dbscan_result(pd_table)

        if temp_f1_score > f1_score_max:
            table_f1_score_max = pd_table.copy()
            f1_score_max = temp_f1_score
            cluster_score_max = cluster.copy()
            border_score_max = list_border[index_cluster]

print("Best matching cluster - f1 score : ", f1_score_max)
print("Best matching cluster - table : ")
print(table_f1_score_max)
print("Best matching cluster : ")
print(cluster_score_max)

if len(border_score_max) == 0:
    print("[ABORTED] Best matching cluster has no border data")
    quit()

border_score_max_labels = df_dt_y_predict.loc[border_score_max.index.values.tolist()]
border_score_max_labels.columns = ['labels']

cluster_score_max_labels = df_dt_y_predict.loc[cluster_score_max.index.values.tolist()]
cluster_score_max_labels.columns = ['labels']

border_score_max = pd.concat([border_score_max, border_score_max_labels], axis=1)
cluster_score_max = pd.concat([cluster_score_max, cluster_score_max_labels], axis=1)

border_score_max = border_score_max.reset_index()

num_cluster = len(cluster_score_max['predict'].drop_duplicates())

data_temp = cluster_score_max.groupby('predict')
list_data_not_noisy_clusters = [data_temp.get_group(x).reset_index() for x in data_temp.groups]

################ LABELING THE FOUND CLUSTERS
df_dbscan_predict = cluster_score_max['predict']

list_columns_table_f1_score_max = table_f1_score_max.columns.values.tolist()
list_dbscan_label = df_dbscan_predict.values.flatten().tolist()
for index_dbscan_predict, elem_dbscan_predict in enumerate(list_dbscan_label):
    list_dbscan_label[index_dbscan_predict] = list_columns_table_f1_score_max.index(elem_dbscan_predict)

df_dbscan_label = pd.DataFrame(list_dbscan_label)

del cluster_score_max['predict']
df_dbscan_x = cluster_score_max

second_graph = fig.add_subplot(122, projection='3d')
plot_3rd_pca(second_graph, "Expert Group(B)", df_dbscan_x, df_dbscan_label)


################ DATA AUGMENTATION PROCESS(1) ~ Finding the closest item of the each cluster at the border data
dict_closest_data = {}

index_border = 0
while index_border <= border_score_max.index[-1]:
    list_closest_data = []
    for cluster in list_data_not_noisy_clusters:
        index_row = 0
        if cluster.loc[index_row, 'predict'] == border_score_max.loc[index_border, 'predict']:
            index_row = index_row + 1
        else:
            shortest_dist = int(2147483647)
            index_shorteds_dist_row = -1

            while index_row <= cluster.index[-1]:
                dist_total = 0
                for index, column in enumerate(list_columns):
                    if column[len(column) - 5:] == "_Code" or column == "labels":
                        continue

                    dist = abs(cluster.loc[index_row, column] - border_score_max.loc[index_border, column])
                    dist_total = dist_total + dist

                if dist_total < shortest_dist:
                    index_shortest_dist_row = index_row
                    shortest_dist = dist_total

                index_row = index_row + 1

            list_closest_data.append(cluster.loc[index_shortest_dist_row])

    dict_closest_data[border_score_max.loc[index_border, 'index']] = list_closest_data
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
                list_cluster_closest_data_not_code_total[count_not_code] = \
                    list_cluster_closest_data_not_code_total[count_not_code] + closest_data.loc[column]
                count_not_code = count_not_code + 1

        list_closest_data_labels_temp.append([closest_data['labels']])

    list_closest_data_labels_temp = sum(list_closest_data_labels_temp, [])
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
for list_total in list_total_container:
    list_temp = []

    for index, column in enumerate(list_columns):
        if column[len(column) - 5:] == "_Code":
            # MAX COUNT
            list_total[index] = int(max(list_total[index], key=list_total[index].count))
        else:
            divisor = num_cluster
            if divisor < 1:
                print("[ABORTED] The number of cluster is zero")
                quit()
            # AVERAGE
            list_total[index] = list_total[index] / divisor

    list_augmented_container.append(list_total)

# labels of data augmentation are prepared at the 'list_augmented_labels'
list_augmented_labels = []

count = 0
list_index_excluded = []
for index_closest_data_labels, closest_data_labels in enumerate(list_closest_data_labels):
    # NOTE : closet_data_labels are also list
    label_min = None
    count_min = int(2147483647)

    df_value_counts = pd.DataFrame(closest_data_labels).value_counts()
    df_value_counts = df_value_counts.sort_values()

    df_index_value_counts = df_value_counts.sort_values().index.to_frame(index=False)

    list_df_value_counts = df_value_counts.values.tolist()
    list_df_index_counts = df_index_value_counts.values.flatten().tolist()

    list_temp_indices_need_to_check = []

    # Neighbours might be consist of [2, 1, 3, 4, 5, 5]. In this case, labels(2, 1, 3, 4) are needed to check
    for index, item in enumerate(list_df_value_counts):
        if list_df_value_counts[0] == item: # list_df_value_counts[0] ~ minimum counts among the neighbours
            list_temp_indices_need_to_check.append(list_df_index_counts[index])

    label_min_final_count = int(2147483647)
    label_min_chosen = None
    # Among labels(2, 1, 3, 4), which label has the least count.
    for label_min in list_temp_indices_need_to_check:
        if dict_labels_value_counts[label_min] < label_min_final_count:
            label_min_chosen = label_min
            label_min_final_count = dict_labels_value_counts[label_min]

    # Final chosen label might be the most count of labels.
    # In this case, this item(value+label) should be excluded.
    if label_min_final_count == max(dict_labels_value_counts.values()):
        if min(dict_labels_value_counts.values()) != max(dict_labels_value_counts.values()): # when label counts are balanced
            list_index_excluded.append(index_closest_data_labels)
            continue

    dict_labels_value_counts[label_min_chosen] = dict_labels_value_counts[label_min_chosen] + 1
    list_augmented_labels.append(label_min_chosen)


# filtering data using list_index_excluded
index_excluded = 0
list_filtered_augmented_container = []
for index_filetered_augmented_container, augmented_container in enumerate(list_augmented_container):
    if index_excluded != len(list_index_excluded):
        if list_index_excluded[index_excluded] == index_filetered_augmented_container:
            index_excluded = index_excluded + 1
            continue

    list_filtered_augmented_container.append(augmented_container)


df_augmented_data = pd.DataFrame(list_filtered_augmented_container, columns=list_columns)
df_augmented_data_label = pd.DataFrame(list_augmented_labels)

X_train_augmented = pd.concat([df_x_train, df_augmented_data])
y_train_augmented = pd.concat([df_y_train, df_augmented_data_label])

X_train_augmented = X_train_augmented.reset_index(drop=True)
y_train_augmented = y_train_augmented.reset_index(drop=True)


print("======================")
print("DECISION TREE")
print("[ORIGINAL] Accuracy for the test set: {:.3f}".format(group_dt.score(df_x_test, df_y_test)))

group_dt.set_data(X_train_augmented, y_train_augmented)
group_dt.quiz_optimize(df_x_quiz, df_y_quiz)

print("[AUGMENTED] Accuracy for the test set: {:.3f}".format(group_dt.score(df_x_test, df_y_test)))
print("======================")