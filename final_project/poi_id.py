#!/usr/bin/python
random = 13

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# Preparing all features and data to test.

all_features_list = [
    'poi',
    'bonus',
    'salary',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'total_payments',
    'total_stock_value',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi',
    'to_messages'
]

features_list = [
    'poi',
    'bonus',
    'salary',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'from_poi_to_this_person_ratio',
    'from_this_person_to_poi_ratio',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi'
]

### Task 2: Remove outliers

if 'TOTAL' in data_dict:
    del data_dict['TOTAL']

data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 3: Create new feature(s)
for name, item in data_dict.items():
    data_dict[name]['from_this_person_to_poi_ratio'] = 0
    data_dict[name]['from_poi_to_this_person_ratio'] = 0

    # Set to 0
    for key in all_features_list:
        if item[key] == 'NaN':
            data_dict[name][key] = 0

    if item['from_messages'] > 0:
        data_dict[name]['from_this_person_to_poi_ratio'] = float(item['from_this_person_to_poi']) / float(item['from_messages'])
    if item['to_messages'] > 0:
        data_dict[name]['from_poi_to_this_person_ratio'] = float(item['from_poi_to_this_person']) / float(item['to_messages'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import StratifiedShuffleSplit
folds = 1000
cv = StratifiedShuffleSplit(
     labels, folds, random_state=random)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Join all the features here
estimators = [
    ('scale', MinMaxScaler()),
    ('classifier', DecisionTreeClassifier(random_state=random))
]


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV

# Parameters used in GridSearchCV
parameters = [
    {
        'classifier__max_features': [None],
        'classifier__criterion': ['entropy'],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [2],
        'classifier__min_weight_fraction_leaf': [0],
        'classifier__max_depth': [2, 3, 4, 5],
        'classifier__class_weight': [{1: 0.8, 0: 0.3}, {1: 0.8, 0: 0.35}, {1: 0.8, 0: 0.25}],
        'classifier__splitter': ['best']
    }
]

init_clf = Pipeline(estimators)


grid = GridSearchCV(init_clf, parameters, cv = cv, scoring='f1', verbose=1)
grid.fit(features, labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = init_clf.set_params(**grid.best_params_)    
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)