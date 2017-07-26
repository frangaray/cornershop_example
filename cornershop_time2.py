print("HOLA")
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import model_selection

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

import sys

from csv import reader

import numpy as np

from enum import Enum

from itertools import izip

from math import sin, cos, sqrt, atan2, radians

class fileName(Enum):
    ORDERS = 0
    ORDER_PRODUCT = 1
    SHOPPERS = 2
    STOREBRANCH = 3

def cleaning_shoppers(dataset):

    # Changing the string code into numbers
    
    dataset.iloc[0::,1] = dataset.iloc[0::,1].replace('REVIEW',0)
    dataset.iloc[0::,1] = dataset.iloc[0::,1].replace('BEGINNER',1)
    dataset.iloc[0::,1] = dataset.iloc[0::,1].replace('INTERMEDIATE',2)
    dataset.iloc[0::,1] = dataset.iloc[0::,1].replace('ADVANCED',3)

    # Replacing NaN by the mean of the variable
    dataset.iloc[0::,2].fillna(dataset.iloc[0::,2].mean(), inplace=True)
    dataset.iloc[0::,3].fillna(dataset.iloc[0::,3].mean(), inplace=True)
    dataset.iloc[0::,4].fillna(dataset.iloc[0::,4].mean(), inplace=True)
    dataset.iloc[0::,5].fillna(dataset.iloc[0::,5].mean(), inplace=True)

    return dataset

def shoppers_info_for_orders(dataset):

    # Info from orders data
    column_picker_id = dataset[fileName.ORDERS]['picker_id']
    column_driver_id = dataset[fileName.ORDERS]['driver_id']

    # Info from shoppers data
    column_shopper_id = dataset[fileName.SHOPPERS]['shopper_id']
    column_seniority_sh = dataset[fileName.SHOPPERS]['seniority']
    column_found_rate_sh = dataset[fileName.SHOPPERS]['found_rate']
    column_picking_speed_sh = dataset[fileName.SHOPPERS]['picking_speed']
    column_accepted_rate_sh = dataset[fileName.SHOPPERS]['accepted_rate']
    column_rating_sh = dataset[fileName.SHOPPERS]['rating']

    # Will save the info into new columns per order
    column_seniority_driver = []
    column_found_rate_driver = []
    column_picking_speed_driver = []
    column_accepted_rate_driver = []
    column_rating_driver = []

    column_seniority_picker = []
    column_found_rate_picker = []
    column_picking_speed_picker = []
    column_accepted_rate_picker = []
    column_rating_picker = []

    for dr_id,pi_id in izip(column_driver_id,column_picker_id):
        for num,sh_id in enumerate(column_shopper_id,start=0):
            if dr_id == sh_id:
                column_seniority_driver.append(column_seniority_sh[num])
                column_found_rate_driver.append(column_found_rate_sh[num])
                column_picking_speed_driver.append(column_picking_speed_sh[num])
                column_accepted_rate_driver.append(column_accepted_rate_sh[num])
                column_rating_driver.append(column_rating_sh[num])

            if pi_id == sh_id:
                column_seniority_picker.append(column_seniority_sh[num])
                column_found_rate_picker.append(column_found_rate_sh[num])
                column_picking_speed_picker.append(column_picking_speed_sh[num])
                column_accepted_rate_picker.append(column_accepted_rate_sh[num])
                column_rating_picker.append(column_rating_sh[num])

    # Adding the new info extracted from shoppers:
    
    dataset[fileName.ORDERS].insert(10, 'seniority_picker',pandas.Series(column_seniority_picker).to_frame('seniority_picker'))
    dataset[fileName.ORDERS].insert(10, 'seniority_driver',pandas.Series(column_seniority_driver).to_frame('seniority_driver'))
    
    dataset[fileName.ORDERS].insert(10, 'found_rate_picker',pandas.Series(column_found_rate_picker).to_frame('found_rate_picker'))
    dataset[fileName.ORDERS].insert(10, 'found_rate_driver',pandas.Series(column_found_rate_driver).to_frame('found_rate_driver'))

    dataset[fileName.ORDERS].insert(10, 'picking_speed_picker',pandas.Series(column_picking_speed_picker).to_frame('picking_speed_picker'))
    dataset[fileName.ORDERS].insert(10, 'picking_speed_driver',pandas.Series(column_picking_speed_driver).to_frame('picking_speed_driver'))

    dataset[fileName.ORDERS].insert(10, 'accepted_rate_picker',pandas.Series(column_accepted_rate_picker).to_frame('accepted_rate_picker'))
    dataset[fileName.ORDERS].insert(10, 'accepted_rate_driver',pandas.Series(column_accepted_rate_driver).to_frame('accepted_rate_driver'))

    dataset[fileName.ORDERS].insert(10, 'rating_picker',pandas.Series(column_rating_picker).to_frame('rating_picker'))
    dataset[fileName.ORDERS].insert(10, 'rating_driver',pandas.Series(column_rating_driver).to_frame('rating_driver'))

    #print dataset[fileName.ORDERS]

    return dataset[fileName.ORDERS]

def time_min(dataset):
    tim1 = pandas.DatetimeIndex(dataset[fileName.ORDERS]['promised_time'])
    tim2 = pandas.DatetimeIndex(dataset[fileName.ORDERS]['actual_time'])
    
    time1 = tim1.hour * 60. + tim1.minute + tim1.second/60.
    time2 = tim2.hour * 60. + tim2.minute + tim2.second/60.
    #print time2

    return time1,time2
    

def column_to_radians(column_torad):
    column_inrad = []
    for i in range(0,len(column_torad)):
        column_inrad.append(radians(column_torad[i]))
        column_inrad.append(radians(column_torad[i]))

    return column_inrad

def column_distance(column_lat_start,column_lat_end,column_lng_start,column_lng_end):
    
    column_lat_dif = []
    column_lng_dif = []
    
    for i in range(0,len(column_lat_start)):
        column_lat_dif.append(column_lat_end[i]-column_lat_start[i])
        column_lng_dif.append(column_lng_end[i]-column_lng_start[i])

    column_a = []
    
    for i in range(0,len(column_lat_dif)):
        column_a.append(sin(column_lat_dif[i]/2)*sin(column_lat_dif[i]/2) + cos(column_latrad_start[i]) * cos(column_latrad_end[i]) * sin(column_lng_dif[i]/2)*sin(column_lng_dif[i]/2))

    column_c = []

    for i in range(0,len(column_a)):
        column_c.append(2 * atan2(sqrt(column_a[i]), sqrt(1 - column_a[i])))

    column_distance = []
    R = 6373.0
    
    for i in range(0,len(column_c)):
        column_distance.append(column_c[i]*R)

    return column_distance


def plot_1D(x_train,y_train,x_validation, y_validation, pred_outcome,x_est,pred_outcome_est, model_name):

    plt.scatter(x_train, y_train,  color='red', label='training data')
    plt.scatter(x_validation, y_validation,  color='black', label='validation data')
    plt.scatter(x_est, pred_outcome_est,  color='green', label='estimation data')
    plt.plot(x_validation, pred_outcome, color='blue',linewidth=3, label=model_name)

    plt.ylabel('total time [min]')
    plt.xlabel('Distance [Km]')
    plt.legend()
    #plt.xticks(())
    #plt.yticks(())

    plt.show()


def plot_multi(x_train,y_train,x_validation,y_r,y_l,y_p):
    lw = 2
    plt.scatter(x_train, y_train, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(x_validation, y_r, color='navy', lw=lw, label='RBF model')
    plt.plot(x_validation, y_l, color='c', lw=lw, label='Linear model')
    plt.plot(x_validation, y_p, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

def corr_matrix(dataset):

    names = ['dow', 'pro_ti','ac_ti','on_dem', 'rat_dri','rat_pi', 'ac_rat_dri', 'ac_rat_pi','pic_spe_dri', 'pic_spe_pi','fo_rat_dri', 'fo_rat_pi','sen_dri', 'sen_pic', 'tot_dist', 'tot_min']
    correlations = dataset.corr()
    print correlations
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ticks = np.arange(0,16,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

# Different ML models
def LR(x_train,y_train,x_validation,y_validation,x_for_est):
    print 'Training Linear Regression...'
    regr = LinearRegression(fit_intercept=True, normalize=False)
    regr.fit(x_train, y_train)
    predict_outcome = regr.predict(x_validation)

    predict_outcome_es = regr.predict(x_for_est)

    #The mean squared error
    print("Mean absolute error: %.2f"
        % np.mean(regr.predict(x_validation) - y_validation))
    #Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x_validation, y_validation))

    return predict_outcome, predict_outcome_es    

def svm(x_train,y_train,x_validation,y_validation):
    # Takes a very long time in 1D, impossible for many variables...
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    y_rbf = svr_rbf.fit(x_train, y_train).predict(x_validation)
    y_lin = svr_lin.fit(x_train, y_train).predict(x_validation)
    y_poly = svr_poly.fit(x_train, y_train).predict(x_validation)

    return y_rbf,y_lin,y_poly

def DTR(x_train,y_train,x_validation,y_validation,depth,x_for_est):
    
    print 'Training Decision Tree...'
    regr = DecisionTreeRegressor(max_depth=depth)
    regr.fit(x_train, y_train)
    predict_outcome = regr.predict(x_validation)
    predict_outcome_est = regr.predict(x_for_est)

    #The mean squared error
    print("Mean absolute error: %.2f" % np.mean(regr.predict(x_validation) - y_validation))
    #Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x_validation, y_validation))

    return predict_outcome,predict_outcome_est

def RFR(x_train,y_train,x_validation,y_validation,depth,x_for_est):

    print 'Training Random Forest Regression...'
    regr_rf = RandomForestRegressor(n_estimators=100,max_depth=depth, random_state=2)
    regr_rf.fit(x_train, y_train)
    pred_out = regr_rf.predict(x_validation)
    pred_out_est = regr_rf.predict(x_for_est)

    #The mean squared error
    print("Mean absolute error: %.2f" % np.mean(regr_rf.predict(x_validation) - y_validation))
    #Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr_rf.score(x_validation, y_validation))
    return pred_out, pred_out_est

def final_csv(x_est,predict_1,predict_2,predict_3,is1D):

    if is1D:

        data_est = pandas.DataFrame(data=x_est,columns=['total_distance'])
        data_LR = pandas.DataFrame(data=predict_1,columns=['total_minutes_LR'])
        data_DTR = pandas.DataFrame(data=predict_2,columns=['total_minutes_DTR'])
        data_RFR = pandas.DataFrame(data=predict_3,columns=['total_minutes_RFR'])

        frames = [data_est,data_LR,data_DTR,data_RFR]
        dataset_final  = pandas.concat(frames, axis=1)
        #print dataset_final
        dataset_final.to_csv('time_estimation_final_1D.csv')

    else:
        data_est = pandas.DataFrame(data=x_est,columns=['dow','promised_time','actual_time','on_demand','rating_driver','rating_picker','accepted_rate_driver','accepted_rate_picker','picking_speed_driver','picking_speed_picker','found_rate_driver', 'found_rate_picker','seniority_driver','seniority_picker','total_distance'])
        data_LR = pandas.DataFrame(data=predict_1,columns=['total_minutes_LR'])
        data_DTR = pandas.DataFrame(data=predict_2,columns=['total_minutes_DTR'])
        data_RFR = pandas.DataFrame(data=predict_3,columns=['total_minutes_RFR'])

        frames = [data_est,data_LR,data_DTR,data_RFR]
        dataset_final  = pandas.concat(frames, axis=1)
        #print dataset_final
        dataset_final.to_csv('time_estimation_final.csv')

    
if __name__ == "__main__":
    # Set this variable for 1D ML algorithm True or False
    is_1D = True

    
    # Loading datasets
    paths = []
    paths.append("/Users/frangaray/Desktop/Cornershop_exercise/datascience-test-master/data/orders.csv")
    paths.append("/Users/frangaray/Desktop/Cornershop_exercise/datascience-test-master/data/order_product.csv")
    paths.append("/Users/frangaray/Desktop/Cornershop_exercise/datascience-test-master/data/shoppers.csv")
    paths.append("/Users/frangaray/Desktop/Cornershop_exercise/datascience-test-master/data/storebranch.csv")

    datasets =[]
    name = []

    for name in paths:
        #print name
        datasets.append(pandas.read_csv(name, low_memory = False))

    # Getting the lat long info for each order and branch id to calculate the total distance:
    # Transforming latitude and longitude into a distance (assuming that they are in degrees):
    # approximate radius of earth in km, assuming the distance is always a straight line (which is not, we have blocks!)

    
    column_order_branch_id = datasets[fileName.ORDERS]['store_branch_id']
    column_store_id = datasets[fileName.STOREBRANCH]['store_branch_id']
    column_lat = datasets[fileName.STOREBRANCH]['lat']
    column_lng = datasets[fileName.STOREBRANCH]['lng']


    column_lat_rad = column_to_radians(column_lat)
    column_lng_rad = column_to_radians(column_lng)

    column_latrad_start = []
    column_lngrad_start = []
    #print column_order_branch_id

    for id in column_order_branch_id:
        #print id
        for num,id_comp in enumerate(column_store_id,start=0):
            if id == id_comp:
               # print id, id_comp
                column_latrad_start.append(column_lat_rad[num])
                column_lngrad_start.append(column_lng_rad[num])


    column_lat_end = datasets[fileName.ORDERS]['lat']
    column_lng_end = datasets[fileName.ORDERS]['lng']

    column_latrad_end = column_to_radians(column_lat_end)
    column_lngrad_end = column_to_radians(column_lng_end)

    column_dist = column_distance(column_latrad_start,column_latrad_end,column_lngrad_start,column_lngrad_end)

    #print(column_dist)
    # Filling the empty spaces of total_minutes to zero so I can identify them
    datasets[fileName.ORDERS].iloc[0::,10].fillna(0, inplace=True)


    # Adding the distance column to the dataset
    datasets[fileName.ORDERS].insert(10, 'total_distance',pandas.Series(column_dist).to_frame('total_distance'))

    if is_1D:
        # Separating the dataset in the group that actually has the total_minutes values and the other one that will be used for estimation.
    
        dataset_est = datasets[fileName.ORDERS][datasets[fileName.ORDERS].total_minutes == 0]
        dataset_train = datasets[fileName.ORDERS][datasets[fileName.ORDERS].total_minutes != 0]

        array = dataset_train.values

        array_est = dataset_est.values
    
        X_est = array_est[0::,10:11]

        print dataset_train
        
        # Slicing the dataset into variable and target
        X = array[0::,10:11]
        Y = array[0::,11]
        
        #print('What we have on X=')
        print(X)
        #sys.exit()
        #print('What we have on Y=')
        print(Y)
        #sys.exit()

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

        print(X_train)

        # Linear Regression for a 1D dataset (distance in kilometers)
    
        predict_out,predict_out_est = LR(X_train,Y_train,X_validation,Y_validation,X_est)
        #print predict_out
        #plot_1D(X_train,Y_train,X_validation,Y_validation,predict_out,X_est,predict_out_est, 'Linear Model')

        # SVM regression 1D (takes a long time!)
        #pred_rbf,pred_lin,pred_poly = svm(X_train,Y_train,X_validation,Y_validation)

        # Descision Tree Regressor 1D
        pred_outcome,pred_outcome_est = DTR(X_train,Y_train,X_validation,Y_validation, 2,X_est)
        #plot_1D(X_train,Y_train,X_validation,Y_validation,pred_outcome,X_est,predict_out_est, 'Descision Tree Regressor')

        # Random Forest Regressor 1D
    
        pred_out, pred_out_est = RFR(X_train,Y_train,X_validation,Y_validation, 2,X_est)
        plot_1D(X_train,Y_train,X_validation,Y_validation,pred_out,X_est,pred_out_est, 'Random Forest Regressor')
        
        final_csv(X_est,predict_out_est,pred_outcome_est,pred_out_est,is_1D)

       
    else:
        
        # PREPARING THE COMPLETE DATASET:
        
        # Looking at SHOPPERS data first and cleaning it:
        datasets[fileName.SHOPPERS] = cleaning_shoppers(datasets[fileName.SHOPPERS])
        
        # Adding info from shoppers list according to the orders list
        datasets[fileName.ORDERS] = shoppers_info_for_orders(datasets)
        
        #Looking into ORDERS data: changing time into minutes for promised time and actual time. I'm going to do a new variable which is the difference in between promised_time and actual_time (actually did not behave very well, leaving it as it is):
        datasets[fileName.ORDERS]['promised_time'],datasets[fileName.ORDERS]['actual_time'] = time_min(datasets)

        #print datasets[fileName.ORDERS]
        #sys.exit()
        
        # Droping the columns that I don't need (doing a simple 1D exercise):
        datasets[fileName.ORDERS].drop(['order_id','lat','lng','picker_id','driver_id','store_branch_id'], axis=1, inplace=True)
        
        datasets[fileName.ORDERS].iloc[0::,3] = datasets[fileName.ORDERS].iloc[0::,3].astype(int)

        # Separating the dataset in the group that actually has the total_minutes values and the other one that will be used for estimation.
    
        dataset_est = datasets[fileName.ORDERS][datasets[fileName.ORDERS].total_minutes == 0]
        dataset_train = datasets[fileName.ORDERS][datasets[fileName.ORDERS].total_minutes != 0]

        #print datasets[fileName.ORDERS]
        
        # Scatter matrix to see how all the variables distribute
        #scatter_matrix(dataset_train)
        #plt.show()
        
        # Correlation matrix
        #corr_matrix(dataset_train)
        
        array = dataset_train.values

        array_est = dataset_est.values
    
        X_est = array_est[0::,0:15]

        
        # Slicing the dataset into variable and target
        X = array[0::,0:15]
        Y = array[0::,15]
        
        #print('What we have on X=')
        #print(X)

        #print('What we have on Y=')
        #print(Y)
        

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

        #print(X_train)
        
        # Linear Regression for the dataset
    
        predict_out,predict_out_est = LR(X_train,Y_train,X_validation,Y_validation,X_est)

        #print predict_out_est

         # Descision Tree Regressor 
        pred_outcome,pred_outcome_est = DTR(X_train,Y_train,X_validation,Y_validation, 100,X_est)

        # Random Forest Regressor 
    
        pred_out, pred_out_est = RFR(X_train,Y_train,X_validation,Y_validation, 100,X_est)
        print dataset_est

        final_csv(X_est,predict_out_est,pred_outcome_est,pred_out_est,is_1D)
        






    
 

    
 



    
