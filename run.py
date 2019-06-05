from flask import Flask, request, jsonify
import pandas as pd
from sklearn import svm
import numpy as np
#from sklearn import cross_validation
from sklearn import preprocessing as pre
from datetime import datetime

app = Flask(__name__)

@app.route('/billgenerate',methods=['POST'])
def home():
    units = int( request.args['power'] )
    if units <= 50 :
        amount = units * 3.85
    elif units <= 150:
        amount = 182.5 + (units-50) * 6.10
    elif units <= 300:
        amount = 182.5 + 610 + (units-150) * 6.40
    elif units <= 500:
        amount = 182.5 + 610 + 960 + (units-300) * 6.70
    else:
        amount = 182.5 + 610 + 960 + 1340 + (units-500) * 7.15
    return str(amount)

# power consumption till now
@app.route('/usage')
def usage():
    data = pd.read_csv('.\\Data\\final.csv',parse_dates=['timestamp'],index_col='timestamp')
    final = data.groupby(data.index.month)
    m_total = []
    d_total = []
    h_total = []

    # monthly consumption
    for i,d in final:
        m_total.append([i,round(d.USAGE.sum(),2)])

    # daily consumption
    for i,d in final:
        for day in set(d.index.day):
            d_total.append([day,d[d.index.day==day]['USAGE'].sum()])
        break
    
    # hourly consumption
    for i,d in final:
        for day in set(d.index.day):
            #l=[]
            #print(i,day)
            for hr in set(d.index.hour):
                h_total.append([hr,d[(d.index.hour==hr) & (d.index.day==day)].USAGE.sum()])
            break
        break 
    

    # prediction of power consumption for the month
    old = pd.read_csv('.\\Data\\final.csv',parse_dates=['timestamp'],index_col='timestamp')
    train_start = '1-march-2014'
    train_end = '25-march-2014'
    test_start = '26-march-2014'
    test_end = '30-april-2014'

    X_train_df = old[train_start:train_end]
    del X_train_df['USAGE']
    del X_train_df['timestamp_end']
    del X_train_df['wspdm']

    y_train_df = old['USAGE'][train_start:train_end]

    X_test_df = old[test_start:test_end]
    del X_test_df['USAGE']
    del X_test_df['timestamp_end']
    del X_test_df['wspdm']

    y_test_df = old['USAGE'][test_start:test_end]

    # Numpy arrays for sklearn
    X_train = np.array(X_train_df)
    X_test = np.array(X_test_df)
    y_train = np.array(y_train_df)
    y_test = np.array(y_test_df)

    scaler = pre.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.001).fit(X_train_scaled,y_train)
    #print('Testing R^2 =', round(SVR_model.score(X_test_scaled,y_test),3))

    # Use SVR model to calculate predicted next-hour usage
    predict_y_array = SVR_model.predict(X_test_scaled)
    # Put it in a Pandas dataframe for ease of use
    predict_y = pd.DataFrame(predict_y_array,columns=['USAGE'])
    predict_y.index = X_test_df.index

    predict = list(zip(predict_y.index,predict_y_array))    
    return jsonify({'month':m_total,'day':d_total,'hour':h_total,'prediction':predict})
# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)