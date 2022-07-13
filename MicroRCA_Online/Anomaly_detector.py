
import numpy as np


from sklearn.cluster import Birch
from sklearn import preprocessing

smoothing_window = 12

# Anomaly Detection
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation. 
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    
    anomalies = []
    #counter = 0
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            n_clusters = np.unique(labels).size

            #print(l)
            #print("bbbb")
            #print(labels)
            #y = np.empty(len(X))
            #y.fill(counter)
            #plt.scatter(X[:, 0], y, c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')

            #counter += 1

            if n_clusters > 1:
                anomalies.append(svc)
    #plt.show()
    return anomalies

def Anomaly_detection_loop(latency_df, ad_threshold, anomaly_mode= None):
    
    anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
    #print("Anomalies")
    #print(anomalies)

    # get the anomalous service
    anomaly_nodes = []
    for anomaly in anomalies:
        edge = anomaly.split('_')
        anomaly_nodes.append(edge[1])

    #anomaly_nodes = set(anomaly_nodes)
    
    return latency_df, anomalies
        