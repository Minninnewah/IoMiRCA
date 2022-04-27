#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: li
"""
from random import random, choice, randrange

import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
#import csv
import yaml
import os

from sklearn.cluster import Birch
from sklearn import preprocessing
#import seaborn as sns

## =========== Data collection ===========

metric_step = '15s'
smoothing_window = 12

# kubectl get nodes -o wide | awk -F ' ' '{print $1 " : " $6":9100"}'
node_dict = {
                'k8s-cluster-work-1-k8-cluster-worker-vnfd-vm-0' : '10.161.2.129:9100',
                'k8-cluster-ns-1-k8-cluster-ns-vnfd-vm-0' : '10.161.2.161:9100'
        }


def prometheus_query(prom_url, start_time, end_time, query):
    response = requests.get(prom_url,
                            params={'query': query,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    return response.json()['data']['result']

def latency_source_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()

    ####Istio request duration

    query = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"source\", destination_workload_namespace=\"sock-shop\"}[2m])) by (destination_workload, source_workload, le))'
    results = prometheus_query(prom_url, start_time, end_time, query)

    #### Add all values to Dataframe
    for result in results:
        #print(result)
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['value']
        if(src_svc == 'unknown' or dest_svc == 'unknown'):
            #print("skip unknown")
            continue

        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = pd.Series(timestamp)
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = str(values[1])

        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')


    #### Istio get send bytes
    query = 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\"}[2m])) by (destination_workload, source_workload) / 1000'
    results = prometheus_query(prom_url, start_time, end_time, query)

    ###Replace latency with sent bytes total
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['value']
        if(src_svc == 'unknown' or dest_svc == 'unknown'):
            #print("skip unknown")
            continue

        metric = values[1]
        if name in latency_df:
            print("Fail the value from above would be overwritten")

        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    latency_df.set_index('timestamp')
    return latency_df


def latency_destination_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()

    query = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"sock-shop\"}[2m])) by (destination_workload, source_workload, le))'
    results = prometheus_query(prom_url, start_time, end_time, query)

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['value']
        if(src_svc == 'unknown' or dest_svc == 'unknown'):
            #print("skip unknown")
            continue

        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = pd.Series(timestamp)
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')

    query = 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\"}[2m])) by (destination_workload, source_workload) / 1000'
    results = prometheus_query(prom_url, start_time, end_time, query)

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['value']
        if(src_svc == 'unknown' or dest_svc == 'unknown'):
            #print("skip unknown")
            continue

        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    latency_df.set_index('timestamp')
    return latency_df

def get_metric_services(prom_url, start_time, end_time, faults_name):

    service_dict = {}

    query = 'sum(rate(container_memory_working_set_bytes{namespace="sock-shop", container!~\'POD|istio-proxy|\'}[10m])) by (pod, instance, container)'
    results = prometheus_query(prom_url, start_time, end_time, query)

    for result in results:
        df = pd.DataFrame()
        svc = result['metric']['container']
        pod = result['metric']['pod']
        nodename = result['metric']['instance']
        values = result['value']

        if 'timestamp' not in df:
            timestamp = values[0]
            df['timestamp'] = pd.Series(timestamp)
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        df['ctn_cpu'] = metric
        df['ctn_cpu'] = df['ctn_cpu'].astype('float64')

        df['ctn_network'] = ctn_network(prom_url, start_time, end_time, pod)
        df['ctn_network'] = df['ctn_network'].astype('float64')
        df['ctn_memory'] = ctn_memory(prom_url, start_time, end_time, pod)
        df['ctn_memory'] = df['ctn_memory'].astype('float64')

        #instance = node_dict[nodename]
        instance = nodename.replace('10250', '9100')

        df_node_cpu = node_cpu(prom_url, start_time, end_time, instance)
        df['node_cpu'] = df_node_cpu

        df_node_network = node_network(prom_url, start_time, end_time, instance)
        df['node_network'] = df_node_network

        df_node_memory = node_memory(prom_url, start_time, end_time, instance)
        df['node_memory'] = df_node_memory

        service_dict[svc] = df
    return service_dict


def svc_metrics(prom_url, start_time, end_time, faults_name):

    query = 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container!~\'POD|istio-proxy|\'}[10m])) by (pod, instance, container)'
    results = prometheus_query(prom_url, start_time, end_time, query)

    for result in results:
        df = pd.DataFrame()
        svc = result['metric']['container']
        pod = result['metric']['pod']
        nodename = result['metric']['instance']
        values = result['value']

        if 'timestamp' not in df:
            timestamp = values[0]
            df['timestamp'] = pd.Series(timestamp)
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        df['ctn_cpu'] = metric
        df['ctn_cpu'] = df['ctn_cpu'].astype('float64')

        df['ctn_network'] = ctn_network(prom_url, start_time, end_time, pod)
        df['ctn_network'] = df['ctn_network'].astype('float64')
        df['ctn_memory'] = ctn_memory(prom_url, start_time, end_time, pod)
        df['ctn_memory'] = df['ctn_memory'].astype('float64')

#        response = requests.get('http://localhost:9090/api/v1/query',
#                                params={'query': 'sum(node_uname_info{nodename="%s"}) by (instance)' % nodename
#                                        })
#        results = response.json()['data']['result']
#
#        print(results)
#
#
#        instance = results[0]['metric']['instance']
        instance = node_dict[nodename]

        df_node_cpu = node_cpu(prom_url, start_time, end_time, instance)
        df['node_cpu'] = df_node_cpu

        df_node_network = node_network(prom_url, start_time, end_time, instance)
        df['node_network'] = df_node_network

        df_node_memory = node_memory(prom_url, start_time, end_time, instance)
        df['node_memory'] = df_node_memory

        df.set_index('timestamp')
        return df

def ctn_network(prom_url, start_time, end_time, pod):

    query = 'sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[2m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod="%s"}[2m])) / 1000' % (pod, pod)
    results = prometheus_query(prom_url, start_time, end_time, query)
    values = [0, float('NaN')]
    if(len(results) == 0):
        print("test")
    else:
        values = results[0]['value']
        #print(values)

    metric = pd.Series(values[1])
    return metric


def ctn_memory(prom_url, start_time, end_time, pod):
    query = 'sum(rate(container_memory_working_set_bytes{namespace="sock-shop", pod="%s"}[2m])) / 1000 ' % pod
    results = prometheus_query(prom_url, start_time, end_time, query)

    values = results[0]['value']
    metric = pd.Series(values[1])
    return metric


def node_network(prom_url, start_time, end_time, instance):

    query = 'rate(node_network_transmit_packets_total{device="ens3", instance="%s"}[2m]) / 1000' % instance
    results = prometheus_query(prom_url, start_time, end_time, query)

    values = results[0]['value']

    return pd.Series(values[1])

def node_cpu(prom_url, start_time, end_time, instance):

    query = 'sum(rate(node_cpu_seconds_total{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%s" }[2m])) / count(node_cpu_seconds_total{mode="system", instance="%s"})' % (instance, instance)
    results = prometheus_query(prom_url, start_time, end_time, query)

    values = results[0]['value']

    return pd.Series(values[1])

def node_memory(prom_url, start_time, end_time, instance):

    query = '1 - sum(node_memory_MemAvailable_bytes{instance="%s"}) / sum(node_memory_MemTotal_bytes{instance="%s"})' % (instance, instance)
    results = prometheus_query(prom_url, start_time, end_time, query)

    values = results[0]['value']

    return pd.Series(values[0])

#add connection to dataframe and graph
def mpg_add_connection(df, DG, results):
    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
        if source != 'unknown' and destination != 'unknown':
            df = pd.concat([df, pd.DataFrame.from_records([{'source': source, 'destination': destination}])], ignore_index=True)
            DG.add_edge(source, destination)
            DG._node[source]['type'] = 'service'
            DG._node[destination]['type'] = 'service'
    return DG, df

def mpg_add_connection_host(df, DG, results):
    for result in results:
        metric = result['metric']
        if 'container' in metric:
            source = metric['container']
            destination = metric['instance']
            df = pd.concat([df, pd.DataFrame.from_records([{'source': source, 'destination': destination}])], ignore_index=True)
            DG.add_edge(source, destination)
            DG._node[source]['type'] = 'service'
            DG._node[destination]['type'] = 'host'
    return DG, df

# Create Graph
def mpg(prom_url, faults_name):
    DG = nx.DiGraph()
    df = pd.DataFrame(columns=['source', 'destination'])
    response = requests.get(prom_url,
                            params={'query': 'sum(istio_tcp_received_bytes_total) by (source_workload, destination_workload)'
                                    })
    results = response.json()['data']['result']
    DG, df = mpg_add_connection(df, DG, results)


    response = requests.get(prom_url,
                            params={'query': 'sum(istio_requests_total{destination_workload_namespace=\'sock-shop\'}) by (source_workload, destination_workload)'
                                    })
    results = response.json()['data']['result']

    DG, df = mpg_add_connection(df, DG, results)

    response = requests.get(prom_url,
                            params={'query': 'sum(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~\'POD|istio-proxy\'}) by (instance, container)'
                                    })
    results = response.json()['data']['result']

    DG, df = mpg_add_connection_host(df, DG, results)

    filename = faults_name + '_mpg.csv'
##    df.set_index('timestamp')
    df.to_csv(filename)
    return DG


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


def node_weight(svc, anomaly_graph, baseline_df, faults_name):

    #Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
#        print(u, v)
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg  = in_edges_weight_avg / num

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    node_cols = ['node_cpu', 'node_network', 'node_memory']
    max_corr = 0.01
    metric = 'node_cpu'
    for col in node_cols:
        temp = abs(baseline_df[svc].corr(df[col]))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric

def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col


    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if anomaly_graph.nodes[v]['type'] == 'service':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

    edges_weight_avg  = edges_weight_avg / num

    personalization = edges_weight_avg * max_corr

    return personalization, metric



def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha):
    # Get the anomalous subgraph and rank the anomalous services
    # input: 
    #   DG: attributed graph
    #   anomlies: anoamlous service invocations
    #   latency_df: service invocations from data collection
    #   agg_latency_dff: aggregated service invocation
    #   faults_name: prefix of csv file
    #   alpha: weight of the anomalous edge
    # output:
    #   anomalous scores 
    
    # Get reported anomalous nodes
    edges = []
    nodes = []
#    print(DG.nodes())
    baseline_df = pd.DataFrame()
    edge_df = {}
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
#        nodes.append(edge[0])
        svc = edge[1]
        nodes.append(svc)
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly

#    print('edge df:', edge_df)
    nodes = set(nodes)
#    print(nodes)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    for node in nodes:
#        print(node)
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u,v)
#            print(edge)
            if edge in edges:
                data = alpha
            else:
                normal_edge = u + '_' + v
                data = baseline_df[v].corr(latency_df[normal_edge])

            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

       # Set personalization with container resource usage
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u,v)
            if edge in edges:
                data = alpha
            else:

                if DG.nodes[v]['type'] == 'host':
                    data, col = node_weight(u, anomaly_graph, baseline_df, faults_name)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']


    for node in nodes:
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, faults_name)
        personalization[node] = max_corr / anomaly_graph.degree(node)
#        print(node, personalization[node])

    anomaly_graph = anomaly_graph.reverse(copy=True)
#
    edges = list(anomaly_graph.edges(data=True))

    for u, v, d in edges:
        if anomaly_graph.nodes[node]['type'] == 'host':
            anomaly_graph.remove_edge(u,v)
            anomaly_graph.add_edge(v,u,weight=d['weight'])

#    plt.figure(figsize=(9,9))
#    #nx.draw(DG, with_labels=True, font_weight='bold')
#    pos = nx.spring_layout(anomaly_graph)
#    nx.draw(anomaly_graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
#    labels = nx.get_edge_attributes(anomaly_graph,'weight')
#    nx.draw_networkx_edge_labels(anomaly_graph,pos,edge_labels=labels)
#    #plt.show()
#    plt.savefig('anomaly_graph.png')
#
##    personalization['shipping'] = 2
#    print('Personalization:', personalization)



    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score



def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--folder', type=str, required=False,
                        default='1',
                        help='folder name to store csv file')
    
    parser.add_argument('--length', type=int, required=False,
                    default=150,
                    help='length of time series')

    parser.add_argument('--url', type=str, required=False,
                    default='http://localhost:9090/api/v1/query',
                    help='url of prometheus query')

    return parser.parse_args()

def get_latency(prom_url, start_time, end_time, faults_name):

    latency_df_source = latency_source_50(prom_url, start_time, end_time, faults_name)

    latency_df_destination = latency_destination_50(prom_url, start_time, end_time, faults_name)

    # remove timestamp, then add the values and add the timestamp again
    timestamp = latency_df_source["timestamp"]
    latency_df_destination_2 = latency_df_destination.drop('timestamp', axis=1)
    latency_df_source_2 = latency_df_source.drop('timestamp', axis=1)
    latency_combined = latency_df_destination_2.add(latency_df_source_2, fill_value=0)  # fill_value=0
    latency_combined.insert(0, 'timestamp', timestamp)
    return latency_combined

def generate_latency_values(latency, amount_timestamps = 2, nan_values=False, fault_injection=False):
    latency = pd.DataFrame(np.zeros((amount_timestamps, len(latency.columns))), columns=latency.columns)
    base_value = 5
    for i in range(len(latency.columns)):
        for j in range(amount_timestamps):
            if j == 0:
                latency.loc[j, latency.columns[i]] = base_value * 2 + i + random() * 3
            else:
                latency.loc[j, latency.columns[i]] = latency.loc[j - 1, latency.columns[i]] + (random()-0.5) * 8

    if nan_values:
        column = choice(latency.columns)
        latency[column] = float("nan")

    if fault_injection:
        column = choice(latency.columns)
        row = randrange(amount_timestamps)
        while latency.loc[row, column] == float("nan"):
            column = choice(latency.columns)
        latency.loc[row, column] = 3 * latency.loc[row, column]

    return latency


def get_metrics_row(start_time, end_time, latency_df=pd.DataFrame(), service_dict={}):

    latency_df = pd.concat([latency_df, get_latency(prom_url, start_time, end_time, faults_name)], ignore_index=True)
    service_dict_temp = get_metric_services(prom_url, start_time, end_time, faults_name)
    for key in service_dict_temp.keys():
        if key in service_dict:
            service_dict[key] = pd.concat([service_dict[key], service_dict_temp[key]], ignore_index=True)
            
        else:
            service_dict[key] = service_dict_temp[key]

    return latency_df, service_dict


def wait_rest_of_interval_time(end_time, interval_time):
    while end_time + (interval_time) >= time.time():
        time.sleep(0.1)


def store_metrics_to_files(latency_df, service_dict):
    latency_df.to_csv("source")
    latency_df.to_html('temp.html')
    for key in service_dict.keys():
        service_dict[key].to_csv(faults_name + '_' + key + '.csv')



def one_time_RCA(prom_url, len_second, faults_name):
    # Tuning parameters
    alpha = 0.55  # 0.55
    ad_threshold = 0.085  # 0.045

    n = 15
    interval_time = 15
    latency_df = pd.DataFrame()
    service_dict = {}

    for i in range(n):
        print("Loop " + str(i), flush=True)
        end_time = time.time()
        start_time = end_time - len_second
        latency_df, service_dict = get_metrics_row(start_time, end_time, latency_df, service_dict)
        if i + 1 < n:
            wait_rest_of_interval_time(end_time, interval_time)

    # latency_df = generate_latency_values(latency_df, amount_timestamps=5, nan_values=True, fault_injection=True)
    store_metrics_to_files(latency_df, service_dict)

    DG = mpg(prom_url, faults_name)

    # anomaly detection on response time of service invocation
    anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
    print("Anomalies")
    print(anomalies)

    # get the anomalous service
    anomaly_nodes = []
    for anomaly in anomalies:
        edge = anomaly.split('_')
        anomaly_nodes.append(edge[1])

    anomaly_nodes = set(anomaly_nodes)

    anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
    print(anomaly_score)

    # anomaly_score_new = []
    # for anomaly_target in anomaly_score:
    #    node = anomaly_target[0]


#   #                     print(anomaly_target[0])
#    if DG.nodes[node]['type'] == 'service':
#        anomaly_score_new.append(anomaly_target)
# print(anomaly_score_new)


def infinite_rca(prom_url, len_second, faults_name, config):
    # Tuning parameters
    alpha = 0.55  # 0.55
    ad_threshold = 0.085  # 0.045

    interval_time = 15
    considered_timestamps = config['CONFIG']['N_TIMESTAMPS']#15
    anomaly_mode = config['CONFIG']['ANOMALY_MODE']
    latency_df = pd.DataFrame()
    service_dict = {}

    DG = mpg(prom_url, faults_name)
    event_counter = 0

    while True:
        print("--- Loop [" + str(len(latency_df)) + " timestamps, " + str(event_counter) + " event-counter]: " + str(time.asctime(time.localtime(time.time()))), flush=True)

        end_time = time.time()
        start_time = end_time - len_second
        latency_df, service_dict = get_metrics_row(start_time, end_time, latency_df, service_dict)

        if len(latency_df) >= considered_timestamps:
            if event_counter in config['EVENTS']:
                os.system("py chaos-mesh.py --fault " + config['EVENTS'][event_counter])
                print('***** Fault: ' + config['EVENTS'][event_counter] + ' started *****')

            anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
            print("Anomalies")
            print(anomalies)

            # get the anomalous service
            anomaly_nodes = []
            for anomaly in anomalies:
                if anomaly != "iot-handler_temperature-sensor":
                    edge = anomaly.split('_')
                    anomaly_nodes.append(edge[1])

            #anomaly_nodes = set(anomaly_nodes)

            anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
            print(anomaly_score)
            if anomaly_mode == 'sliding_window':
                latency_df.drop(latency_df.head(1).index, inplace=True)
            elif anomaly_mode == 'base_data':
                latency_df.drop(latency_df.tail(1).index, inplace=True)
            elif anomaly_mode =='sliding_window_improved':
                if len(anomalies) == 0:
                    latency_df.drop(latency_df.head(1).index, inplace=True)
                else:
                    latency_df.drop(latency_df.tail(1).index, inplace=True)
            else:
                print("Anomaly mode not supported")
            event_counter += 1

        wait_rest_of_interval_time(end_time, interval_time)



if __name__ == '__main__':
    args = parse_args()

    folder = args.folder
    len_second = args.length
    prom_url = args.url
      
    faults_name = folder

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #one_time_RCA(prom_url, len_second, faults_name)
    infinite_rca(prom_url, len_second, faults_name, config)
