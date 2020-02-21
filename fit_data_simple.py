from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import csv
import numpy as np
from commons.common_tools import def_log
from joblib import dump, load

# input_path = 'data_from_7.csv'
input_path = 'data_to_7.csv'

with open(input_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    x = []
    y = []
    for i, row in enumerate(spamreader):
        x_, y_ = row
        x.append(x_)
        y.append(y_)

x = np.array(x)[:, np.newaxis]
y = np.array(y)[:, np.newaxis]

def_log.d(x.shape, y.shape)

predict = [[-45], [-30], [-15]]

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
predict_ = poly.transform(predict)

clf = linear_model.LinearRegression()
clf.fit(X_, y)
print(clf.predict(predict_))

dump(poly, 'models/self_poly_to_7.joblib')
dump(clf, 'models/self_linear_to_7.joblib')

# poly = load('models/poly_from_7.joblib')
# X_ = poly.transform(x)
# predict_ = poly.fit_transform(predict)
#
# clf = load('models/linear_from_7.joblib')
# print(clf.predict(predict_))
