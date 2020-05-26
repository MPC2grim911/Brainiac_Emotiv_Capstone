#machine learning

#import statements
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from IPython.display import display

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from time import time

from sklearn.metrics import f1_score

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report




#visualization code for displaying skewed distributions of features
def distribution(data, value, transformed = False):

    fig = plt.figure(figsize = (11, 5)); #create figure

    for i , feature in enumerate([value]): #skewed feature plotting
        ax = fig.add_subplot(1, 1, i+1)
        ax.hist(data[data.columns[feature-1]], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(data.columns[feature-1]), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 500))
        ax.set_yticks([0,100,200,300,400,500])


    if transformed: #plot aesthetics
        fig.suptitle("Log-transformed Distributions of Continuous EEG Data Features", fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous EEG Data Features", fontsize = 16, y = 1.03)

    fig.tight_layout()
    return




#calculate and return the performance score between true and
#predicted values based on the chosen metric
def performance_metric(y_tru, y_pred):

    score = accuracy_score(y_tru, y_pred) #calculate the performance score

    return score #return score value




#perform grid search over the 'max_depth' parameter for a decision 
#tree regressor trained on the input data (x, y)
def fit_model(x, y): #FIXME: may have to change paameter dictionary

    #create cross validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)


    #create a linear discriminant analysis object
    clf = SVC(kernel = 'rbf', class_weight='balanced')

    #create a dictionary for the parameters 
    paraDict = {'C':[1e3,5e3,1e4,5e4,1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]} #FIXME here: modify 

    #Transform 'performance_metric' into scoring function
    score_func = make_scorer(performance_metric) 

    #create the grid search cv object
        #Include parameters: (estimator, param_grid, scoring, cv)
        #which is equivalent to (clf, paraDict, score_func, cv_sets) 
    grid = GridSearchCV(clf, paraDict, scoring=score_func, cv=cv_sets)

    #fit grid search object to the data to compute optimal model
    grid = grid.fit(x, y) 

    return grid.best_estimator_ #return optimal model after fit data



def main():

    #Data exploration    #FIXME HERE!!!

    raw_data = pd.read_csv('data_set_file.csv') #read in data #FIXME: change name as needed

    #split data into features and target labels
    target_raw = raw_data[raw_data.columns[-1]]
    features_raw = raw_data.drop(raw_data.columns[-1], axis = 1)

    print("The shape of the data: {}".format(raw_data.shape)) #print data shape

    display(raw_data.head(n=15)) #display first 15 records of data



    #Data preparation
    distribution(raw_data, 6) #visualize skewed continuous features of the original data



    #log transform the skewed features
    feature_log_trans = features_raw.apply(lambda x: np.log(x+1))

    #visualize the new log distributions
    distribution(feature_log_trans, 6, transformed = True)



    #outlier detection

    #calculate q1 (25th quantile of the data) for all features
    Q1 = feature_log_trans.quantile(0.25)

    #calculate q3 ( 90th quantile of the data) for all features
    Q3 = feature_log_trans.quantile(0.90)


    #use interquartile range to calculate an outlier step
    #(1.5 times the interquartile range)
    IQR = Q3 - Q1
    out_Step = 1.5 * IQR


    #Remove the outlier from the dataset
    feature_log_trans_out = feature_log_trans[~((feature_log_trans < (Q1 - out_Step)) | (feature_log_trans > (Q3 + out_Step))).any(axis = 1)]


    #join features and target after removing outliers
    preprocessed_data_out = feature_log_trans_out.join(target_raw)
    target_raw_out = preprocessed_data_out[preprocessed_data_out.columns[-1]]

    #print data shape after removing outliers
    print("The shape of the data after removing outliers: {}".format(preprocessed_data_out.shape))

    display(preprocessed_data_out.head(n=10)) #display first 10 records



    #Normalizing numerical features

    #Initialize a scaler and apply to features
    scaler = MinMaxScaler() #default = (0, 1)

    feature_log_minmax_trans_out = pd.DataFrame(scaler.fit_transform(feature_log_trans_out), columns=features_raw.columns)

    display(feature_log_minmax_trans_out.head()) #show example scaling record




    #shuffle and split data

    #Assign features to bands variable and labels to state varaable
    bands = np.array(feature_log_minmax_trans_out)
    state = np.array(target_raw_out)

    #shuffle and split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(bands, state, test_size=0.2, random_state=42, shuffle=True)

    print("Training and testing split was successful")




    #PCA Transformation

    pca = PCA().fit(x_train) #fit the PCA algorithm with data

    #plot the cumulative summation of the explained variance
    plt.figure(figsize = (14,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()



    n_comp = 33 #from the explained variance graph

    print("Estracting the top %d eigenfaces from %d faces"%(n_comp, x_train.shape[0]))
    t0 = time()

    #create an instance of PCA, initialized with n_comp
    pca = PCA(n_components=n_comp)

    # Pass the training dataset to pca's fit method
    pca = pca.fit(x_train)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    print("Explained variance ratios: ", pca.explained_variance_ratio_*100)
    print("done in %0.3fs"%(time() - t0))




    #fitting the model

    #fit training data to model using grid search
    model = fit_model(x_train_pca, y_train) #FIXME

    #produce value for gamma and C #FIXME: may have to change parameters
    print("Parameter 'gamma' is {} for the original model.".format(model.get_params()['gamma']))
    print("Parameter 'C' is {} for the optimal model.".format(model.get_params()['C']))





    #making predictions 
    y_pred = model.predict(x_test_pca) #make predictions

    #Lable states class. 
    states_class = ['Triangle', 'Circle', 'Square']

    for i, state in enumerate(y_pred): #show predictions
        print("Predicted object id for test {}'s bands: {}".format(i+1, states_class[state-1]))

        



    #final model Evaluation #FIXME here

    #calculate f1 score and assign to variable
    score = f1_score(y_test, y_pred, average='micro')

    print("F1 score: %0.1f %%"%(score*100)) #print score


    #calculate the confusion matrix and assign to variable
    matrix = confusion_matrix(y_test, y_pred)

    #plot the confusion matrix
    fig, ax = plot_confusion_matrix(matrix)
    labels = ['triangle', 'circle', 'square']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    
    plt.savefig('plot_confusion_matrix.png') #save results
    plt.show()



    #calculate classification report and assign to variable
    report = classification_report(y_test, y_pred)

    print(report)#print classification report

    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('report.csv')




if __name__ == "__main__":
    main()




