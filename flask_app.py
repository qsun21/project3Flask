from flask import Flask
from flask import Response
from bson import json_util
import json
import os
import ssl
from sklearn import svm
import numpy as np

from pymongo import MongoClient

uri = os.environ.get('MONGO_URI', None)
client = MongoClient(uri, ssl_cert_reqs=ssl.CERT_NONE)
clf = svm.SVC(kernel='linear')

try:
    print(client.server_info())
except Exception as e:
    print(e)

database = client['project3']
col = database.get_collection("breast_cancer")
cols = col.find({'is_training': {'$eq': True}})
results = []
labels = []
for element in cols:
    row = []
    row.append(element.get("radius_mean"))
    row.append(element.get("texture_mean"))
    row.append(element.get("perimeter_mean"))
    row.append(element.get("area_mean"))
    row.append(element.get("smoothness_mean"))
    row.append(element.get("compactness_mean"))
    row.append(element.get("concavity_mean"))
    row.append(element.get("concave_points_mean"))
    row.append(element.get("symmetry_mean"))
    row.append(element.get("radius_se"))
    row.append(element.get("texture_se"))
    row.append(element.get("perimeter_se"))
    row.append(element.get("area_se"))
    row.append(element.get("smoothness_se"))
    row.append(element.get("compactness_se"))
    row.append(element.get("concavity_se"))
    row.append(element.get("concave_points_se"))
    row.append(element.get("symmetry_se"))
    row.append(element.get("fractal_dimension_se"))
    row.append(element.get("radius_worst"))
    row.append(element.get("texture_worst"))
    row.append(element.get("perimeter_worst"))
    row.append(element.get("area_worst"))
    row.append(element.get("smoothness_worst"))
    row.append(element.get("compactness_worst"))
    row.append(element.get("concavity_worst"))
    row.append(element.get("concave_points_worst"))
    row.append(element.get("symmetry_worst"))
    row.append(element.get("fractal_dimension_worst"))
    labels.append(element.get("diagnosis"))
    results.append(row)
np_samples = np.array(results)
np_labels = np.array(labels)
clf.fit(np_samples, np_labels)

app = Flask(__name__)
 
@app.route('/data')
def data():
    collections = col.find({'is_training': {'$eq': False}})
    collections_list = list(collections)
    test = []
    for element in collections_list:
        row = []
        row.append(element.get("radius_mean"))
        row.append(element.get("texture_mean"))
        row.append(element.get("perimeter_mean"))
        row.append(element.get("area_mean"))
        row.append(element.get("smoothness_mean"))
        row.append(element.get("compactness_mean"))
        row.append(element.get("concavity_mean"))
        row.append(element.get("concave_points_mean"))
        row.append(element.get("symmetry_mean"))
        row.append(element.get("radius_se"))
        row.append(element.get("texture_se"))
        row.append(element.get("perimeter_se"))
        row.append(element.get("area_se"))
        row.append(element.get("smoothness_se"))
        row.append(element.get("compactness_se"))
        row.append(element.get("concavity_se"))
        row.append(element.get("concave_points_se"))
        row.append(element.get("symmetry_se"))
        row.append(element.get("fractal_dimension_se"))
        row.append(element.get("radius_worst"))
        row.append(element.get("texture_worst"))
        row.append(element.get("perimeter_worst"))
        row.append(element.get("area_worst"))
        row.append(element.get("smoothness_worst"))
        row.append(element.get("compactness_worst"))
        row.append(element.get("concavity_worst"))
        row.append(element.get("concave_points_worst"))
        row.append(element.get("symmetry_worst"))
        row.append(element.get("fractal_dimension_worst"))
        test.append(row)
    print(test)
    np_test = np.array(test)
    pred_labels = clf.predict(np_test)
    
    final_list = []
    for i in range(0, len(collections_list)):
        el_dict = dict(element)
        el_dict['predicted_label'] = pred_labels[i]
        final_list.append(el_dict)

    return json_util.dumps(final_list)

if __name__ == '__main__':
   app.run(host="0.0.0.0")
