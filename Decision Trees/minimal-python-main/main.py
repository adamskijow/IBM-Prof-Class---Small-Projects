import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
    my_data = pd.read_csv(path)
    my_data.info()

#   This tells us that 4 out of the 6 features of this dataset are categorical, which will have to be converted into numerical ones to be used for modeling.
#  For this, we can make use of __LabelEncoder__ from the Scikit-Learn library.
    label_encoder = LabelEncoder()
    my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
    my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
    my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
    my_data.isnull().sum() # check if there are any missing values in the dataset. 

    custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
    my_data['Drug_num'] = my_data['Drug'].map(custom_map)

    category_counts = my_data['Drug'].value_counts()

# Plot the count plot
    plt.bar(category_counts.index, category_counts.values, color='blue')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.title('Category Distribution')
    plt.xticks(rotation=45)  # Rotate labels for better readability if needed
    plt.show()

#  Now, we can split our dataset into features and labels, and then into training and testing sets.
    y = my_data['Drug']
    X = my_data.drop(['Drug','Drug_num'], axis=1)
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    drugTree.fit(X_trainset,y_trainset)
#  Now that we have trained our model, we can make predictions.
    tree_predictions = drugTree.predict(X_testset)
    print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
#   Finally, we can visualize the decision tree.
    plot_tree(drugTree)
    plt.show()






if __name__ == "__main__":
    main()
