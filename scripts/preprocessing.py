from matplotlib.pyplot import axis
import numpy as np

def normalize(x):
    return (x-np.mean(x))/np.std(x)

def drop_corr(x):
    "Calculate correlation and delete one of the 2 features if abs(correlation) is above 0.95"
    #calculate correlation between different features, corr is the pearson correlation between 2 features
    cor = np.corrcoef(x.T)
    #Find index where abs(r) is close to 1
    p = np.argwhere(np.abs(cor) > 0.95)
    ind = []
    for i in range (np.size(p, axis=0)):
        if (p[i,0] < p[i,1]) and p[i,1] not in ind: 
            ind = np.append(ind, p[i,1])

    return ind.astype(int)

def missing_values(x):
    """
        Change every -999 values to the median value of the corresponding column 
        and take the index of columns having more than a third of these values to drop them later
    """
    counts=[]
    medians = np.median(x, axis = 0)
    clean_x = np.copy(x)
    for j in range(np.size(x, axis=1)):
        count = 0
        for i in range(np.size(x, axis=0)):
            if clean_x[i, j] == -999:
                clean_x[i, j] = medians[j]
                count += 1
        counts = np.append(counts, count)
    counts = counts/np.size(x, axis=0)
    #Removing columns where more than a thirs of values are missing
    ind = np.argwhere(counts > 1/3)  

    return ind, clean_x 

def removed_index(x):
    "Concatenate the final column indexes to remove"
    ind_miss, clean_x = missing_values(x)
    ind_corr = drop_corr(clean_x)
    ind = np.concatenate((np.squeeze(ind_miss), ind_corr))
    ind = np.unique(ind)
    return ind, clean_x

def preprocessing(x, ind):
    "Removing columns"
    clean_x = np.copy(x)
    clean_x = np.delete(clean_x, ind, axis = 1)
    return clean_x

def standardize(tX, tX_to_standardize):
    mean_tX = tX.mean(0)
    std_tX = tX.std(0)
    return (tX_to_standardize-mean_tX)/std_tX

def full_preprocessing(tX, tX_test):
    "Step by step full preprocessing of the data"
    #Taking indexes of columns to remove and creating clean_x a dataset where -999 values are replaced
    #by the median of the column
    ind, clean_tX = removed_index(tX)  
    #droping columns form tX
    tX_preprocessed = preprocessing(clean_tX, ind)

    tX_normalized = standardize(tX_preprocessed, tX_preprocessed)

    #Same operations on tX_test but we use mean, std deviation and medians computed on the train dataset
    _, clean_tX_test = removed_index(tX_test)
    tX_test_preprocessed = preprocessing(clean_tX_test, ind)
    tX_test_normalized = standardize(tX_preprocessed,tX_test_preprocessed)
    return tX_normalized, tX_test_normalized
    