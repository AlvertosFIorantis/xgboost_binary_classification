from pandas import DataFrame
import datetime
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score



df_2 = pd.read_csv("./titanic.csv", sep="\t")


df_2.head()


del df_2['Name']
del df_2['PassengerId']
del df_2['Ticket']
del df_2['Cabin']




'''Removing duplicates'''
df_2 = DataFrame.drop_duplicates(df_2)

numeric_columns = list(df_2.select_dtypes(
    include="number").columns.values)

'''Filing missing values in numeric columns with 0'''
for col in numeric_columns:
    df_2[col] = df_2[col].fillna(0)


cat_columns = list(df_2.select_dtypes(
    include="object").columns.values)

'''Filling missing values in categorical columns using "UNKNOWN" token'''
for col in cat_columns:
    df_2[col] = df_2[col].fillna('UNKNOWN')


'''Check for missing values '''
df_2.isnull().values.any()

'''Split dataset to train and test '''

train_1, test = train_test_split(
    df_2, test_size=0.1, random_state=123)


'''Now split the train to a smaller train and validation dataset '''
train, validation = train_test_split(train_1, test_size=0.15, random_state=123)


def one_hot_encoding_train(dataset, normalize=False, levels_limit=200):
    if normalize == True:
        '''Normalize numeric data'''
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.externals import joblib
        '''Get numeric columns'''
        numeric_columns = list(dataset.select_dtypes(
            include="number").columns.values)
        scaler = MinMaxScaler()
        dataset[numeric_columns] = scaler.fit_transform(
            dataset[numeric_columns])
        joblib.dump(scaler, './scaler.pkl')
    import pickle
    '''Collect all the categorical columns'''
    cat_columns = list(dataset.select_dtypes(include="object").columns.values)
    for col in cat_columns:
        column_length = (len(dataset[col].unique()))
        if column_length > levels_limit:
            dataset.drop(str(col), axis=1, inplace=True)
            cat_columns.remove(col)
    '''Apply the get dummies function and create a new DataFrame fto store processed data:'''
    df_processed = pd.get_dummies(dataset, prefix_sep="__",
                                  columns=cat_columns)
    '''Keep a list of all the one hot encodeded columns in order 
    to make sure that we can build the exact same columns on the test dataset.'''
    cat_dummies = [col for col in df_processed
                   if "__" in col
                   and col.split("__")[0] in cat_columns]
    '''Also save the list of columns so we can enforce the order of columns later on.'''
    processed_columns = list(df_processed.columns[:])
    '''Save all the nesecarry lists into pickles'''
    with open('cat_columns.pkl', 'wb') as f:
        pickle.dump(cat_columns, f)
    with open('cat_dummies.pkl', 'wb') as f:
        pickle.dump(cat_dummies, f)
    with open('processed_columns.pkl', 'wb') as f:
        pickle.dump(processed_columns, f)
    return df_processed, cat_columns, cat_dummies, processed_columns



def one_hot_encoding_test(test_dataset, normalize=False):
    if normalize == True:
        '''Normalize numeric data'''
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.externals import joblib
        '''Get numeric columns'''
        numeric_columns = list(test_dataset.select_dtypes(
            include="number").columns.values)
        scaler = joblib.load('scaler.pkl')
        test_dataset[numeric_columns] = scaler.fit_transform(
            test_dataset[numeric_columns])
    import pickle
    '''Process the unseen (test) data!'''
    '''Load nessecary lists from pickles'''
    with open('cat_columns.pkl', 'rb') as f:
        cat_columns = pickle.load(f)
    with open('cat_dummies.pkl', 'rb') as f:
        cat_dummies = pickle.load(f)
    with open('processed_columns.pkl', 'rb') as f:
        processed_columns = pickle.load(f)
    df_test_processed = pd.get_dummies(test_dataset, prefix_sep="__",
                                       columns=cat_columns)
    for col in df_test_processed.columns:
        if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
            print("Removing (not in training) additional feature  {}".format(col))
            df_test_processed.drop(col, axis=1, inplace=True)
    for col in cat_dummies:
        if col not in df_test_processed.columns:
            print("Adding missing feature {}".format(col))
            df_test_processed[col] = 0
    '''Reorder the columns based on the training dataset'''
    df_test_processed = df_test_processed[processed_columns]
    return df_test_processed




target_idx = train.columns.get_loc("Survived")

X_train = train.loc[:, train.columns != 'Survived']
Y_train = train[train.columns[target_idx]]

X_validation = validation.loc[:, validation.columns != 'Survived']
Y_validation = validation[validation.columns[target_idx]]

X_test = test.loc[:, test.columns != 'Survived']
Y_test = test[test.columns[target_idx]]


X_train, cat_columns, cat_dummies, processed_columns = one_hot_encoding_train(
    X_train, True)

X_validation = one_hot_encoding_test(X_validation, True)

X_test = one_hot_encoding_test(X_test, True)



'''Baysian optimization'''

from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def xgb_evaluate(min_child_weight, gamma, subsample, colsample_bytree, max_depth, learning_rate, scale_pos_weight, num_round,reg_lambda,reg_alpha): #max_depth
    max_depth = int(max_depth)
    best = xgb.XGBClassifier(min_child_weight = min_child_weight, gamma = gamma, subsample = subsample, colsample_bytree = colsample_bytree, 
    max_depth = int(max_depth), learning_rate = learning_rate, num_round = num_round, scale_pos_weight = scale_pos_weight,reg_lambda=reg_lambda,reg_alpha=reg_alpha, seed=16)
    best.fit(X_train, Y_train)

    valid_y_pred = best.predict(X_validation)
    cm = confusion_matrix(Y_validation, valid_y_pred)
    #presision = cm[1][1]/(cm[1][1]+cm[1][0])
    #Accuracy
    #presision=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])
    #presision=(cm[1][1])/(cm[1][1]+cm[0][1])
    #presision=((cm[1][1]/cm[0][1]))+0.000000001#otherwise i can get in inf value in which case the optimizer will crush
    #presision=cm[1][1]
    #presision=(recall_score(Y_validation, valid_y_pred)+precision_score(Y_validation, valid_y_pred))
    #presision=precision_score(Y_validation, valid_y_pred)
    presision = accuracy_score(Y_validation,valid_y_pred)
    #presision=recall_score(Y_validation, valid_y_pred)
    return presision


from bayes_opt import BayesianOptimization
xgb_bo = BayesianOptimization(xgb_evaluate,
    {'min_child_weight': (0.5,100),
    'gamma': (0,30),#30
    'subsample': (0.6, 1),
    'colsample_bytree': (0.4, 1),
    'max_depth': (1, 30),
    'learning_rate': (0.05, 0.3),
    'scale_pos_weight': (1, 3),
    'num_round': (100, 500),
    'reg_lambda':(0,100),
    'reg_alpha':(0,100)
    })
    


start_time = timer(None) # timing starts from this point for "start_time" variableper
xgb_bo.maximize(init_points=10, n_iter=5, acq='ei')
timer(start_time) # timing ends here for "start_time" variable


params = xgb_bo.max['params']
num_round=int(params["num_round"])
params["max_depth"]=int(params["max_depth"])
params['eval_metric']='error'
params['objective']='binary:logistic'


data_dmatrix = xgb.DMatrix(data=X_train, label=Y_train)
validation_matrix = xgb.DMatrix(data=X_validation, label=Y_validation)
watchlist = [(validation_matrix, 'eval'), (data_dmatrix, 'train')]


bst = xgb.train(params, data_dmatrix,num_round,
                watchlist)


from xgboost import plot_importance
plot_importance(bst, max_num_features=10,importance_type='weight')

'''Create confusion matrix for churn model '''
valid_y_pred = bst.predict(validation_matrix)
valid_y_pred[valid_y_pred > 0.50] = 1
valid_y_pred[valid_y_pred <= 0.50] = 0

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print(recall_score(Y_validation, valid_y_pred))
print(precision_score(Y_validation, valid_y_pred))
print(accuracy_score(Y_validation,valid_y_pred))
print(confusion_matrix(Y_validation, valid_y_pred))


import shap
explainer  = shap.Explainer(bst,X_validation)
shap_values = explainer(X_validation)

shap.summary_plot(shap_values, X_validation)

obs = 10
#10 the recrod in the X_validation
base_value = Y_train.mean()
shap_values_observation = shap_values[obs]
def obs_to_explain(data,base_value ,shap_values,n):
    
        for_plot = pd.DataFrame({'data':np.round(data,2),
                                 'shap':shap_values,
                                 'shap_abs': np.abs(shap_values),
                                 'label': data.index
                                })
        for_plot = for_plot.sort_values(by='shap_abs',ascending=False)

        # Split the variables into n and the rest. Only show the top n
        for_plot1 = for_plot.iloc[0:n,:]
        for_plot2 = for_plot.iloc[n:,:]

        # Sum up the rest as 'others'
        rest = pd.DataFrame({'data': '','shap':for_plot2['shap'].sum(), 'label': 'Others'},index=['others'])
        for_plot = for_plot1.append(rest)

        # Sum up the rest into 'others'
        base = pd.DataFrame({'data': np.round(base_value,2),'shap':base_value, 'label': 'Base value'},index=['base_value'])
        for_plot = base.append(for_plot)

        for_plot['blank'] = for_plot['shap'].cumsum().shift(1).fillna(0) # +  base_value
        for_plot['label'] = for_plot['label'] + " =" + for_plot['data'].map(str) 
        for_plot = for_plot.drop(['data','shap_abs'],axis=1)
        

        
        return(for_plot ) 
    
#n = number of most important features
most_important_features= obs_to_explain(X_validation.iloc[obs,:],
                              base_value = base_value,
                              shap_values = rf_shap_values[obs],n=8)





#Check that the values matches the values in the plot
obs=10
shap.plots.waterfall(shap_values[obs])
most_important_features= obs_to_explain(X_validation.iloc[obs,:],
                              base_value = shap_values[obs].base_values,
                              shap_values = shap_values[obs].values ,n=8)
print(most_important_features)
print(X_validation.iloc[obs,:])


