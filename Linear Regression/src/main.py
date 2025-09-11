# Simplest form project to demonstrate linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

if __name__ == "__main__":
    url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df=pd.read_csv(url) #Pull sample data from the IBM course
    print(df.sample(5)) #Show a sample in terminal
    print(df.describe()) #show description in terminal
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    print(df.sample(9)) #print out a subset that might be indicative of CO2 emissions
    # Now we want to display scatter plots of engine features against CO2 emissions to see how linear their relationship is
    viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    viz.hist() 
    plt.show() # display the histograms.
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("Emission")
    plt.show()
    #Now lets see how engine size and emission are correlated
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.xlim(0,27)
    plt.show()

    #Now that we've done analysis, lets start with the regression
    X = cdf.ENGINESIZE.to_numpy()
    y = cdf.CO2EMISSIONS.to_numpy()   

    from sklearn.model_selection import train_test_split # import the train test split function
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    type(X_train), np.shape(X_train), np.shape(X_train) #check the type and shape of X_train
    
    from sklearn import linear_model

    # create a model object
    regressor = linear_model.LinearRegression()

    #train the model on the training data
    # X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
    # So we need to reshape it. We can let it infer the number of observations using '-1'.
    regressor.fit(X_train.reshape(-1, 1), y_train)

    # Print the coefficients
    print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
    print ('Intercept: ',regressor.intercept_)
    plt.scatter(X_train, y_train,  color='blue')
    plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Use the predict method to make test predictions
    y_test_ = regressor.predict(X_test.reshape(-1,1))

    # Evaluation
    print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
    print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
    print("R2-score: %.2f" % r2_score(y_test, y_test_))


