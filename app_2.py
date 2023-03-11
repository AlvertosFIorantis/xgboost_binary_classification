#%%
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
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime
from bayes_opt import BayesianOptimization

from xgboost import plot_importance


#%%
df_2 = pd.read_csv("./titanic.csv", sep="\t")


df_2.head()


del df_2['Name']
del df_2['PassengerId']
del df_2['Ticket']
del df_2['Cabin']

def numeric_fill_na(dataset_input):
    '''
    Input: Pandas DataFrame
    Operation: fill the missing values on a numeric column using 0
    Output: returns a Pandas DataFrame
    '''
    dataset = dataset_input.copy()
    numeric_columns = list(dataset.select_dtypes(
    include="number").columns.values)
    for col in numeric_columns:
        dataset[col] = dataset[col].fillna(0)  
    return dataset

def categorical_fill_na(dataset_input):
    '''
    Input: Pandas DataFrame
    Operation: fill the missing values on a categorical column using UNKNOWN
    Output: returns a Pandas DataFrame
    '''
    dataset = dataset_input.copy()
    cat_columns = list(dataset.select_dtypes(
        include="object").columns.values)
    for col in cat_columns:
        dataset[col] = dataset[col].fillna('UNKNOWN')
    return dataset


'''Removing duplicates'''
df_2 = DataFrame.drop_duplicates(df_2)


df_2 = numeric_fill_na(df_2)
df_2 = categorical_fill_na(df_2)


'''Check for missing values '''
df_2.isnull().values.any()

'''Split dataset to train and test '''

train_1, test = train_test_split(
    df_2, test_size=0.1, random_state=123)


'''Now split the train to a smaller train and validation dataset '''
train, validation = train_test_split(train_1, test_size=0.15, random_state=123)


def one_hot_encoding_train(dataset, normalize=False, levels_limit=200):
    '''
    Input: Pandas DataFrame ,normalize boolean Default False, levels_limit intiger
            default value 200
    Operation: Main operation is to  scale numerical columns and one hot encode 
            categorical ones. If normalize=False then we dont apply MinMax Scaler
            to numerical columns. If a categorical column has more than levels_limit
            levels then that columns is been dropped and not used in one hot encoding
    Output: Pickles all the files that being returned. Returns processed dataFrame 
            list of all categorical columns levels of categorical columns and order
    '''
    if normalize == True:
        numeric_columns = list(dataset.select_dtypes(
            include="number").columns.values)
        scaler = MinMaxScaler()
        dataset[numeric_columns] = scaler.fit_transform(
            dataset[numeric_columns])
        with open('./scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    cat_columns = list(dataset.select_dtypes(include="object").columns.values)
    for col in cat_columns:
        column_length = (len(dataset[col].unique()))
        if column_length > levels_limit:
            dataset.drop(str(col), axis=1, inplace=True)
            cat_columns.remove(col)

    df_processed = pd.get_dummies(dataset, prefix_sep="__",
                                  columns=cat_columns)
    cat_dummies = [col for col in df_processed
                   if "__" in col
                   and col.split("__")[0] in cat_columns]

    processed_columns = list(df_processed.columns[:])
    with open('cat_columns.pkl', 'wb') as f:
        pickle.dump(cat_columns, f)
    with open('cat_dummies.pkl', 'wb') as f:
        pickle.dump(cat_dummies, f)
    with open('processed_columns.pkl', 'wb') as f:
        pickle.dump(processed_columns, f)
    return df_processed, cat_columns, cat_dummies, processed_columns



def one_hot_encoding_test(test_dataset, normalize=False):
    '''
    Input: Pandas DataFrame ,normalize boolean Default False, Also pickels from 
          the trained function
    Operation: Main operation is to  scale numerical columns and one hot encode 
            categorical ones. If normalize=False then we dont apply MinMax Scaler
            to numerical columns. If a categorical column has more than levels_limit
            levels then that columns is been dropped and not used in one hot encoding
    Output: Processed dataset
    '''
    if normalize == True:
        numeric_columns = list(test_dataset.select_dtypes(
            include="number").columns.values)
        scaler = pickle.load(open('./scaler.pkl', 'rb'))
        test_dataset[numeric_columns] = scaler.transform(
            test_dataset[numeric_columns])
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

data_dmatrix = xgb.DMatrix(data=X_train, label=Y_train)
validation_matrix = xgb.DMatrix(data=X_validation, label=Y_validation)
watchlist = [(validation_matrix, 'eval'), (data_dmatrix, 'train')]


def timer(start_time=None):
    '''
    just a function to time things.
    '''
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def xgb_evaluate(min_child_weight, gamma, subsample, colsample_bytree, max_depth, learning_rate, scale_pos_weight, num_round,reg_lambda,reg_alpha): 
    '''
    Inputs: hyperParameters used to tune an xgboost model
    Operation: Tuning a classification Xgboost model using recall on the top
                decile on the validation dataset
    Returns: the Recall value
    '''
    max_depth = int(max_depth)
    num_round=int(num_round)
    params = {
        "objective":"binary:logistic"
        ,"eval_metric":"error"
        ,"colsample_bytree":colsample_bytree
        ,"min_child_weight":min_child_weight
        ,"subsample":subsample
        ,"eta":learning_rate
        ,"max_depth":max_depth
        ,"gamma":gamma
        ,"scale_pos_weight":scale_pos_weight
        ,"reg_lambda":reg_lambda
        ,"reg_alpha":reg_alpha
    }

    bst =xgb.train(params,data_dmatrix,num_round)

    valid_y_pred = bst.predict(validation_matrix)
    threshold = np.percentile(valid_y_pred,np.arange(0,100,10))[-1]

    predictions=[]
    for value in valid_y_pred:
        if value>=threshold:
            predictions.append(1.0)
        else:
            predictions.append(0.0)
    recall = recall_score(Y_validation,predictions)
    return recall

xgb_bo = BayesianOptimization(xgb_evaluate,
    {'min_child_weight': (0.5,10),
    'gamma': (0,0),#30
    'subsample': (0.6, 1),
    'colsample_bytree': (0.4, 1),
    'max_depth': (2, 10),
    'learning_rate': (0.05, 0.3),
    'scale_pos_weight': (1, 1),
    'num_round': (100, 500),
    'reg_lambda':(0,1),
    'reg_alpha':(0,1)
    })
    


start_time = timer(None) # timing starts from this point for "start_time" variableper
xgb_bo.maximize(init_points=1, n_iter=20, acq='ei')
timer(start_time) # timing ends here for "start_time" variable


params = xgb_bo.max['params']
num_round=int(params["num_round"])
params["max_depth"]=int(params["max_depth"])
params['eval_metric']='error'
params['objective']='binary:logistic'



#%%
bst = xgb.train(params, data_dmatrix,num_round,
                watchlist)


#plot_importance(bst, max_num_features=10,importance_type='weight')

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

#%%

'''
Plot recall per decile
'''

test_y_pred = bst.predict(validation_matrix)

deciles = []
recalls = []
precisions = []
thresholds = []
for n in range(1,11):
    deciles.append(n)

    threshold = np.percentile(test_y_pred,np.arange(0,100,10))[-n]
    thresholds.append(threshold)

    predictions = []
    for value in test_y_pred:
        if value >= threshold:
            predictions.append(1.0)
        else:
            predictions.append(0.0)
    
    recall = recall_score(Y_validation,predictions)
    recalls.append(recall)

    precission = precision_score(Y_validation,predictions)
    predictions.append(precission)

fig,ax = plt.subplots(figsize=(10,5))
plt.plot(deciles,[round(x*100,1) for x in recalls],marker='o',c='Indigo',markersize=15,linewidth=3)
plot.xlabel('Decile',fontsize=15)
plot.ylabel('Fraction of total conversions recovered [%]',fontsize=15)
plt.xticks(np.arange(1.10+1,1),fontsize=15)
plt.yticks(np.arange(10,100+1,10),fontsize=15)
plt.grid()
plt.show()


pickle.dump(thresholds,open("./thresholds.pkl","wb"))

#%%
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


#%%




#%%
#Check that the values matches the values in the plot
obs=10
shap.plots.waterfall(shap_values[obs])
most_important_features= obs_to_explain(X_validation.iloc[obs,:],
                              base_value = shap_values[obs].base_values,
                              shap_values = shap_values[obs].values ,n=8)
print(most_important_features)
print(X_validation.iloc[obs,:])



# %%
''''
Converting log odds to probabilties
'''
explainerModel_prob = shap.TreeExplainer(bst,data = X_train)

shap_values_model_prob  = explainerModel_prob.shap_values(X_train)

 


 



explainerModel_prob=shap.TreeExplainer (bst, data = X_train)
with open('explainer.pkl', 'wb') as f:
    pickle.dump(explainerModel_prob, f)


with open('explainer.pkl', 'rb') as f:
    explainer_loaded = pickle.load(f)
shap_values_model_prob = explainer_loaded.shap_values (X_test)


test_dmatrix = xgb.DMatrix(data=X_test, label=Y_test)
final_predictions= bst.predict(test_dmatrix)


#########################################################


def xgb_shap_transform_scale(shap_values, model_prediction):
    #Compute the transformed base value, which consists in applying the logit function to the base value
    from scipy.special import expit #Importing the logit function for the base value transformation
    untransformed_base_value = shap_values[-1]
    base_value = expit(untransformed_base_value )
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = sum(shap_values)
    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = (model_prediction - base_value)
    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient
    #Finally resetting the base_value as it does not need to be transformed
    shap_values_transformed [-1] = base_value
    #Now returning the transformed array
    return shap_values_transformed


test_dmatrix = xgb.DMatrix(data=X_test, label=Y_test)
final_predictions = bst.predict(test_dmatrix)

#Prodcution dataframe

Production_dataframe = pd.DataFrame(columns=['PROBABILITY', 'explanations'])

 

 

number_of_explanations=10

with open('explainer.pkl', 'rb') as f:
    explainer_loaded = pickle.load(f)


shap_values_model_prob  = explainer_loaded.shap_values(X_test)


 
final_list_of_dictionaries = []
for j in range(X_test.shape[0]):
    shap_values=shap_values_model_prob[j]
    #instead of scorring the model all the time i should save the preidcitons
    model_prediction=final_predictions[j]
    shap_proba=xgb_shap_transform_scale(shap_values, model_prediction)
    #Get the position in array of the 10 larger values
    arr =shap_values
    large_indexes=arr.argsort()[-number_of_explanations:][::-1] #get the position of the 10 largest values
    largest_values=np.sort(arr)[-number_of_explanations:][::-1]# 10 largest values
    smallest_values=np.sort(arr)[:number_of_explanations] # 10 lowest values
    smallest_indexes=arr.argsort()[:number_of_explanations] #indexes of the 10 lowest vlues
    large_values= {}
    for A, B in zip(large_indexes, largest_values):
        large_values[A] = B
    small_values= {}
    for A, B in zip(smallest_indexes, smallest_values):
        small_values[A] = B
    #merge the 2 dictionaries
    combined_dictionary = {**large_values, **small_values}
    main_keys = sorted(combined_dictionary, key=lambda dict_key: abs(combined_dictionary[dict_key]))[-10:]
    #the 10 keys with the largest absolute values
    final_dictioanry = dict((k, combined_dictionary[k]) for k in main_keys if k in combined_dictionary)
    DSS_dictionary_output= {}
    for key, value in final_dictioanry.items():
        colname = X_test.columns[key]
        DSS_dictionary_output[colname] = value
    DSS_dataframe = pd.DataFrame([DSS_dictionary_output])
    #Try to convert the row to json instead of creating new columns
    DSS_dataframe.astype(str)
    row_json_similar_to_DSS = DSS_dataframe.to_json(orient='records')[1:-1].replace('},{', '} {')
    row_json_similar_to_DSS_clean = row_json_similar_to_DSS.replace("/", "_")
    row_json_similar_to_DSS_clean = row_json_similar_to_DSS_clean.replace("\\", "")
    #Need to convert the predcition to score by using the deciles
    row_list=[model_prediction,row_json_similar_to_DSS_clean]
    row_df = pd.DataFrame([row_list],columns=['PROBABILITY', 'explanations'],dtype=str) # i foce all columns to be strings helps with the json
    Production_dataframe = Production_dataframe.append(row_df, ignore_index=True)
    #to mono pou alaka einai to eplaantions edo kai pano pou ftiaxno proti fora to dataframe apo kefalea se mikra
    ## Extra append the dictioanry to my list 
    final_list_of_dictionaries.append(DSS_dictionary_output)

### Load thresholds

loaded_thresholds=pickle.load(open('./thresholds.pkl', "rb"))

def probability_to_risk_score(probability, model_thresholds):
    '''
    Converts the probability of a model to risk scores by using the decile threshold of a sepcifc model

    The function recieves 2 arguments, the first argument is the probability of the model and the second

    argument is an ordered array (from larger to smallest)
    '''

    if probability>=model_thresholds[0]:
        return 10
    elif probability>=model_thresholds[1] and probability<model_thresholds[0]: 
        return 9 
    elif probability>=model_thresholds[2] and probability<model_thresholds[1]:
        return 8

    elif probability>=model_thresholds[3] and probability<model_thresholds[2]: 
        return 7 
    elif probability>=model_thresholds[4] and probability<model_thresholds[3]:
        return 6

    elif probability>=model_thresholds[5] and probability<model_thresholds[4]:
        return 5 
    elif probability>=model_thresholds[6] and probability<model_thresholds[5]:
        return 4 
    elif probability>=model_thresholds[7] and probability<model_thresholds[6]:
        return 3
    elif probability>=model_thresholds[8] and probability<model_thresholds[7]:
        return 2
    else:
        return 1
    

Production_dataframe[ "PROBABILITY"]= Production_dataframe["PROBABILITY"].astype (float) 
Production_dataframe[ "Score"]=Production_dataframe.apply(lambda x: probability_to_risk_score(x[ 'PROBABILITY'], loaded_thresholds), axis=1) 

Production_dataframe[ "PROBABILITY"]= Production_dataframe[ "PROBABILITY"].astype(str)


##############



### I do that so i can have the shap values fo all the test dataset into 1 dictiaonry just the total
# adding the values with common key

dict1={}

for dict2 in final_list_of_dictionaries:
    for key in dict2:
        if key in dict1:
            dict1[key] = dict2[key] + dict1[key]
        else:
            dict1[key] = dict2[key]
         
highest_5=dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True)[:5]) 
lowest_5=dict(sorted(dict1.items(), key=lambda x: x[1], reverse=True)[5:])   



combined_dictionary = {**highest_5, **lowest_5}
main_keys = sorted(combined_dictionary, key=lambda dict_key: abs(combined_dictionary[dict_key]))[-10:]
#the 10 keys with the largest absolute values
DSS_dictionary_output = dict((k, combined_dictionary[k]) for k in main_keys if k in combined_dictionary)
DSS_dataframe = pd.DataFrame([DSS_dictionary_output])
#Try to convert the row to json instead of creating new columns
DSS_dataframe.astype(str)
row_json_similar_to_DSS = DSS_dataframe.to_json(orient='records')[1:-1].replace('},{', '} {')
row_json_similar_to_DSS_clean = row_json_similar_to_DSS.replace("/", "_")
row_json_similar_to_DSS_clean = row_json_similar_to_DSS_clean.replace("\\", "")



