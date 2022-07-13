import requests
import pandas as pd
from random import random, choice, randrange
import numpy as np

import networkx as nx

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

def latency_source_50(prom_url, start_time, end_time):

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


def latency_destination_50(prom_url, start_time, end_time):

    latency_df = pd.DataFrame()

    query = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"sock-shop\"}[2m])) by (destination_workload, source_workload, le))'
    results = prometheus_query(prom_url, start_time, end_time, query)

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
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')

    #for i in latency_df:
        #print(i)
    query = 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\"}[2m])) by (destination_workload, source_workload) / 1000'
    results = prometheus_query(prom_url, start_time, end_time, query)

    for result in results:
        #print(result)
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

def get_metric_services(prom_url, start_time, end_time):

    service_dict = {}

    query = 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container!~\'POD|istio-proxy|\'}[2m])) by (pod, instance, container)'
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
        df['node_cpu'] = df_node_cpu.astype('float64')

        df_node_network = node_network(prom_url, start_time, end_time, instance)
        df['node_network'] = df_node_network.astype('float64')

        df_node_memory = node_memory(prom_url, start_time, end_time, instance)
        df['node_memory'] = df_node_memory.astype('float64')

        service_dict[svc] = df
    return service_dict

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
def mpg(prom_url, folder_name):
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

    filename = 'mpg.csv'
##    df.set_index('timestamp')
    df.to_csv(folder_name + "\\" + filename)
    return DG


def get_latency(prom_url, start_time, end_time):

    latency_df_source = latency_source_50(prom_url, start_time, end_time)

    latency_df_destination = latency_destination_50(prom_url, start_time, end_time)

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

def get_latency_row(prom_url, start_time, end_time, latency_df=pd.DataFrame()):
    latency_df = pd.concat([latency_df, get_latency(prom_url, start_time, end_time)], ignore_index=True)
    return latency_df

def get_metrics_row(prom_url, start_time, end_time, service_dict={}):

    service_dict_temp = get_metric_services(prom_url, start_time, end_time)
    for key in service_dict_temp.keys():
        if key in service_dict:
            service_dict[key] = pd.concat([service_dict[key], service_dict_temp[key]], ignore_index=True)
            
        else:
            service_dict[key] = service_dict_temp[key]

    return service_dict