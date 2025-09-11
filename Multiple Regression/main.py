import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df = pd.read_csv(url)

    # verify successful load with some randomly selected records
    df.sample(5)
    # Drop categoricals and any garbage columns - like MODELYEAR is all the same
    df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
    # You want to eliminate any strong dependencies or correlations between features by selecting the best one from each correlated group.
    df.corr()
    df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
    df.head(9) # take a look at the dataset
    axes = pd.plotting.scatter_matrix(df, alpha=0.2)
    # To help with selecting predictive features that are not redundant, 
    # consider the following scatter matrix, which shows the scatter plots for each pair of input features. 
    # The diagonal of the matrix shows each feature's histogram.

    # need to rotate axis labels so we can read them
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()
    #Extract the required columns and convert the resulting dataframes to NumPy arrays.
    X = df.iloc[:,[0,1]].to_numpy()
    y = df.iloc[:,[2]].to_numpy()

    from sklearn import preprocessing
    #subtract the mean and divide by the standard deviation. This is called standardization.
    std_scaler = preprocessing.StandardScaler()
    X_std = std_scaler.fit_transform(X)

    pd.DataFrame(X_std).describe().round(2)

    from sklearn import linear_model

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)

    # create a model object
    regressor = linear_model.LinearRegression()

    # train the model in the training data
    regressor.fit(X_train, y_train)

    # Print the coefficients
    coef_ =  regressor.coef_
    intercept_ = regressor.intercept_

    print ('Coefficients: ',coef_)
    print ('Intercept: ',intercept_)
    # Get the standard scaler's mean and standard deviation parameters
    means_ = std_scaler.mean_
    std_devs_ = np.sqrt(std_scaler.var_)

    # The least squares parameters can be calculated relative to the original, unstandardized feature space as:
    coef_original = coef_ / std_devs_
    intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

    print ('Coefficients: ', coef_original)
    print ('Intercept: ', intercept_original)

    # Ensure X1, X2, and y_test have compatible shapes for 3D plotting
    X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
    X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

    #   Create a mesh grid for plotting the regression plane
    x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                               np.linspace(X2.min(), X2.max(), 100))

    y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf

    # Predict y values using trained regression model to compare with actual y_test for above/below plane colors
    y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
    above_plane = y_test >= y_pred
    below_plane = y_test < y_pred
    above_plane = above_plane[:,0]
    below_plane = below_plane[:,0]

    # Plotting
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points above and below the plane in different colors
    ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
    ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

    # Plot the regression plane
    ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

    # Set view and labels   
    ax.view_init(elev=10)

    ax.legend(fontsize='x-large',loc='upper center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect(None, zoom=0.75)
    ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
    ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
    ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
    ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
    plt.tight_layout()
    plt.show()
    #Instead of making a 3D plot, which is difficult to interpret, 
    #you can look at vertical slices of the 3D plot by plotting each variable separately
    #as a best-fit line using the corresponding regression parameters.
    plt.scatter(X_train[:,0], y_train,  color='blue')
    plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    plt.scatter(X_train[:,1], y_train,  color='blue')
    plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
    plt.xlabel("FUELCONSUMPTION_COMB_MPG")
    plt.ylabel("Emission")
    plt.show()


    

if __name__ == "__main__":
    main()
