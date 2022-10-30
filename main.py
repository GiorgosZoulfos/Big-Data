import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# # ------- main starts here -----------

# ------------- PART A -----------------

# Import clinical dataset
clinical_data = pd.read_csv('clinical_dataset.csv', sep=';', encoding="ISO-8859-1", low_memory=False,
                            header=0)

# Convert nominal features to numerical
clinical_data = clinical_data.replace(
    {'Frail': 2, 'Pre-frail': 1, 'Non frail': 0, 'F': 1, 'M': 0
        , 'Yes': 1, 'No': 0, 'Sees well': 2, 'Sees moderately': 1
        , 'Sees poorly': 0, 'Hears well': 2
        , 'Hears moderately': 1, 'Hears poorly': 0, '>5 sec': 1, '<5 sec': 0
        , True: 1, False: 0, 'Permanent sleep problem': 2
        , 'Occasional sleep problem': 1, 'No sleep problem': 0
        , '5 - Excellent': 5
        , '4 - Good': 4, '3 - Medium': 3, '2 - Bad': 2, '1 - Very bad': 1
        , '5 - A lot better': 5
        , '4 - A little better': 4, '3 - About the same': 3
        , '2 - A little worse': 2
        , '1 - A lot worse': 1, '> 5 h per week': 3
        , '> 2 h and < 5 h per week': 2
        , '< 2 h per week': 1, 'Never smoked': 2
        , 'Past smoker (stopped at least 6 months)': 1
        , 'Current smoker': 0})

# Remove erroneous values
clinical_data = clinical_data.replace({'test non realizable': np.NaN
                                       , 'test non adequate': np.NaN
                                       , 'Test non adequate': np.NaN
                                       , 'test non applicable': np.NaN
                                       , 999: np.NaN, -391.96400: np.NaN})

# Handle missing values
# find nan cells and fill them with the mean value of the column
size = clinical_data.shape

for row in range(size[0]):
    for col in range(size[1]):
        if pd.isna(clinical_data.iloc[row, col]):
            clinical_data.iloc[row, col] = clinical_data.iloc[:, col].mean()

# Classification
# - keep part of the features
# - split the data in 80% - 20% manner
# - train svm model for classification
# - predict the fried parameter
labels = clinical_data.iloc[:, 1].to_numpy().astype(int)

corr = clinical_data.corr()
data = clinical_data.filter(['age', 'raise_chair_time', 'gait_get_up'
                             , 'gait_speed_4m'
                             , 'depression_total_score', 'pain_perception'
                             , 'comordibities_count'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

clf_model = svm.SVC(kernel='linear', C=1)
clf_model.fit(X_train, y_train)

print(clf_model.score(X_test, y_test))

predictions = clf_model.predict(X_test)

# -------------- PART B -------------------

# Import beacons dataset
beacons_data = pd.read_csv('beacons_dataset.csv', sep=';', encoding="ISO-8859-1", low_memory=False,
                           header=0)

# Correct room labels
room_names = beacons_data.room.unique()

beacons_data = beacons_data.replace(
    dict.fromkeys(['Kitcheb', 'Kitch', 'Kitcen', 'DinerRoom'
                   , 'DiningRoom', 'DinningRoom'
                   , 'Kithen', 'Kitchen2', 'Kitvhen', 'kitchen'
                   , 'Kichen', 'Kiychen'
                   , 'DinnerRoom', 'Dinerroom'], 'Kitchen'))

beacons_data = beacons_data.replace(
    dict.fromkeys(['Bed', 'Bedroom1', 'Bedroom2'
                   , 'bedroom', 'Bedroom1st'
                   , 'Bedroom-1'], 'Bedroom'))

beacons_data = beacons_data.replace(
    dict.fromkeys(['Barhroom', 'Bathroon', 'Washroom'
                   , 'Baghroom', 'Bsthroom', 'Bathroim'
                   , 'Bqthroom', 'Bathroom-1', 'Bathroom1'], 'Bathroom'))

beacons_data = beacons_data.replace(
    dict.fromkeys(['Livroom', 'Living', 'Livingroom2', 'Livingroom1'
                      , 'Sitingroom', 'Leavingroom', 'Sittingroom'
                      , 'LivingRoom', 'Sittigroom', 'Luvingroom1'
                      , 'SittingRoom', 'LeavingRoom', 'SeatingRoom'
                      , 'LuvingRoom', 'LivingRoom2', 'Liningroom'
                      , 'SittingOver', 'Livingroon', 'livingroom'
                      , 'Sittingroom', 'Sittinroom', 'LivibgRoom'
                      , 'Leavivinroom'], 'Livingroom'))

beacons_data = beacons_data.replace(
    dict.fromkeys(['Workroom', 'Office1', 'Office2'
                      , 'Office1st', 'Office-2'], 'Office'))

# Remove erroneous users
beacons_data = beacons_data.loc[(beacons_data['part_id'].astype(str).str.isnumeric())
                                & (beacons_data['part_id'].astype(str).str.len() == 4)]

# Generate features
fmt = '%H:%M:%S'

beacons_data = beacons_data.sort_values(by=['part_id'])

# generated_data = {'part_id': [], 'Bedroom': [], 'Bathroom': [], 'Livingroom': [], 'Kitchen': []}
generated_data = pd.DataFrame(columns=['part_id', 'Bedroom', 'Bathroom', 'Livingroom', 'Kitchen'])

shape = beacons_data.shape

current_user = beacons_data.iloc[0, 0]
other_rooms = 0
data_to_add = {'part_id': int(current_user), 'Bedroom': 0, 'Bathroom': 0, 'Livingroom': 0, 'Kitchen': 0}
rooms = ['Bedroom', 'Bathroom', 'Livingroom', 'Kitchen']
for row in range(shape[0] - 1):
    if beacons_data.iloc[row + 1, 0] == current_user:

        room = beacons_data.iloc[row, 3]
        time = (datetime.strptime(beacons_data.iloc[row + 1][2], fmt) -
                datetime.strptime(beacons_data.iloc[row][2], fmt)).total_seconds()

        if time < 0:
            time += 24*60*60

        if room in rooms:
            data_to_add[room] = data_to_add[room] + time
        else:
            other_rooms += time

    else:

        total_time = sum(data_to_add.values()) - int(data_to_add['part_id']) + other_rooms

        if total_time != 0:
            data_to_add['Bedroom'] = (data_to_add['Bedroom'] / total_time) * 100
            data_to_add['Bathroom'] = (data_to_add['Bathroom'] / total_time) * 100
            data_to_add['Livingroom'] = (data_to_add['Livingroom'] / total_time) * 100
            data_to_add['Kitchen'] = (data_to_add['Kitchen'] / total_time) * 100

        generated_data = generated_data.append(data_to_add, ignore_index=True)
        current_user = beacons_data.iloc[row + 1, 0]
        other_rooms = 0
        data_to_add = {'part_id': int(current_user), 'Bedroom': 0, 'Bathroom': 0, 'Livingroom': 0, 'Kitchen': 0}

# Merge the 2 datasets into one
data_for_clustering1 = pd.merge(generated_data, clinical_data, left_on='part_id', right_on='ï»¿part_id', how='inner')
data_for_clustering2 = data_for_clustering1.drop(columns=['ï»¿part_id', 'fried'])

# Cluster analysis
# - perform PCA (down to 2 dimensions - ~ 98%)
# - cluster analysis with k-means

pcadata = PCA(n_components=2)
pcadata.fit(data_for_clustering2)
print(pcadata.explained_variance_ratio_)
data_for_clustering3 = pcadata.fit_transform(data_for_clustering2)

data_for_clustering3 = pd.DataFrame(data_for_clustering3)

f = data_for_clustering1.fried

d = data_for_clustering3
d['category'] = f

sns.scatterplot(d[0], d[1], hue=d['category'], palette='deep')

# Cluster analysis with PCA

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data_for_clustering3)

clusters = kmeans.labels_
clusters = pd.DataFrame(clusters)

category = pd.DataFrame(data_for_clustering1.fried)

d1 = data_for_clustering3
d1['category'] = f
d1['predicted_cluster'] = clusters

sns.scatterplot(d1[0], d1[1], hue = d1['predicted_cluster'], palette='deep')
silhouette_with_PCA = silhouette_score(data_for_clustering3, clusters, metric='euclidean', sample_size=None, random_state=None)

# Cluster analysis without PCA
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data_for_clustering2)

clusters = kmeans.labels_
clusters = pd.DataFrame(clusters)


silhouette_without_PCA = silhouette_score(data_for_clustering2, clusters, metric='euclidean', sample_size=None, random_state=None)

# Save the pre-processed datasets
clinical_data.to_csv('clinical_dataset_pre_processed')
data_for_clustering2.to_csv('beacons_dataset_pre_processed')
