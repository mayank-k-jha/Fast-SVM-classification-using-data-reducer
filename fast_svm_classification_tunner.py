
# ************************** Author : Mayank Kumar Jha (mk9440) https://github.com/mk9440 ******************************
# ************************************************Read Me **************************************************************
# **********************************************************************************************************************
#                                       Parameters description
# **********************************************************************************************************************
# data  = must be provided as Pandas DataFrame with all numerical values
# target = must be provided as Pandas 1D DataFrame with all numerical values 
# no_of_cluster = 'auto' means function will automatically select no of optimal cluster for clustering based on target values
#                  otherwise can be set to any numerical values >=2, see auto implementation for more info
# scaling = indicates whether to apply scaling on data or not, as KMeans are sensitive to larger values, so default parameter 
#           value is True to normalize data before Applying KMeans
# n_jobs  = No of CPU cores to use, default is set to -1 ie use all cores
# no_of_iteration = indicates how many iteration to allow for KMeans to converge
# seed = value for Random State
#verbose = whether to print message during processing or not
# return type is dataframe, So function will return both data and target chopped after samples removal
# **********************************************************************************************************************



def sample_reducer_for_fast_SVM_classification(data, target, no_of_cluster = 'auto', scaling = True, 
                                               n_jobs = -1,no_of_iteration=200,seed = 2464616, verbose = True):

    # Reseting index value as index might get shuffled during cross_validation or data splitting
    data = data.reset_index().drop(['index'],1)
    target = target.reset_index().drop(['index'],1)
    
    data_copy = data.copy()
    # storing columns of data
    cols = list(data.columns)

    if scaling:
        sc = MinMaxScaler()
        data = sc.fit_transform(data)
        data = pd.DataFrame(data,columns=cols)

    if no_of_cluster == 'auto':
        # chosing 4 times, because smaller the cluster would be, better are the chances for crisp clusters 
        # (Think practically :D )
        no_of_cluster = 4 * len(set(target))

    if verbose:
        print(no_of_cluster," clusters are going to be formed")
        print('Started Clustering Algorithm, Please wait.....')
    
    # Defining KMeans clustering classifier
    clf = KMeans(n_clusters=no_of_cluster,n_jobs=n_jobs,random_state=seed,max_iter=no_of_iteration)

    labels = clf.fit_predict(data)

    # assigning distance of each samples from its cluster
    all_data = clf.transform(data)

    # finding clusters having only single class around its radius
    all_data = pd.DataFrame(all_data,columns=['distance_'+str(i) for i in range(all_data.shape[1])])
    all_data['target'] = labels


    # Grouping each clustered dataframe for further individual operations on each cluster
    g = all_data.groupby('target')


    # getting radius size for each cluster
    crisp_clusters = []
    for i in range(no_of_cluster):
        radius = g.get_group(i).reset_index().max()['distance_'+str(i)]
        is_eligible =True
        for j in [k for k in range(no_of_cluster) if k!=i]:
            if g.get_group(j).reset_index().min()['distance_'+str(i)] <= radius:
                is_eligible = False
                break
        if is_eligible:
            crisp_clusters.append(i)

    
    if verbose:
        print("Clustering Completed Successfully")
        print("Total Number of Crisp Clusters Found = ",len(crisp_clusters),' Clusters')
        print("Started Sampling for samples removal")

    # getting radius of the crisp clusters, 
    # If the outer radius is small (say ≤ 1 unit), threshold = 25 % and if the outer radius is small (say > 1 unit),
    # threshold = 50 %.
    # here threshold is the no of samples to be selected out of total samples in it's cluster for removal
    index_to_remove = []
    for i in crisp_clusters:
        if g.get_group(i).reset_index().max()['distance_'+str(i)] <= 1:
            # remove 25% samples closest to it's centroid
            bound = np.percentile(g.get_group(i).reset_index()['distance_'+str(i)].values,q=(0,25),axis=0)[1]
            index_to_remove += list(all_data[(all_data['target']==i) & (all_data['distance_'+str(i)] <= bound)].index.values)
            
        else:
            #remove 50% samples closest to it's centroid
            bound = np.percentile(g.get_group(i).reset_index()['distance_'+str(i)].values,q=(0,50),axis=0)[1]
            index_to_remove += list(all_data[(all_data['target']==i) & (all_data['distance_'+str(i)] <= bound)].index.values) 
            
    if verbose:
        print("Sampling Completed, Total ",len(index_to_remove),' samples are freed!')
        
    return data_copy.drop(index_to_remove),target.drop(index_to_remove)
	

#***************************************************************************************************************************************
# An example showing above function in action on Iris Dataset

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


#loading iris data
iris = load_iris()

# converting features and target to Pandas DataFrame
x, y = pd.DataFrame(iris.data), pd.DataFrame(iris.target)

# Splitting data for cross validation
x_tr,x_ts,y_tr,y_ts = train_test_split(x,y,test_size = 0.5)

#Defining SVM Classifier
clf = svm.SVC()

#fitting SVM classifier on data without removal of samples
clf.fit(x_tr,y_tr)

print('\nBefore applying sample reduction,\n')
#Getting performance Report 
print(classification_report(y_true=y_ts,y_pred = clf.predict(x_ts)))

#Now removing unwanted samples with our algorithm
x_test_new,y_test_new = sample_reducer_for_fast_SVM_classification(data=x_tr,target=y_tr)

#Re Defining new SVM Classifier
clf = svm.SVC()

#fitting new SVM classifier on data after removal of samples
clf.fit(x_test_new,y_test_new)

print('\nAfter applying sample reduction,\n')
#Getting performance Report 
print(classification_report(y_true=y_ts,y_pred = clf.predict(x_ts)))
