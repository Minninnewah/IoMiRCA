#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: li
"""

import argparse
import math
import time 
import pandas as pd
import yaml
import os

import MUD_handler
import RCA_detector
import Metrics_collector
import Anomaly_detector


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
    parser.add_argument('--mud', type=str, required=False,
                    default='MUD_files',
                    help='Folder of the MUD files')

    return parser.parse_args()

def wait_rest_of_interval_time(end_time, interval_time):
    while end_time + (interval_time) >= time.time():
        time.sleep(0.1)

def store_metrics_to_files(latency_df, service_dict, folder_name):
    latency_df.to_csv(folder_name + "\\" + "latency.csv")
    latency_df.to_html(folder_name + "\\" + 'latency.html')
    for key in service_dict.keys():
        service_dict[key].to_csv(folder_name + "\\" + key + '.csv')

def remove_IoT_device_metrics(data, mud_data):
    connections = []
    if isinstance(data, pd.DataFrame):
        for connection in data.keys():
            if any(device in connection for device in mud_data):
                #Don't add connections with only zeros to active connections
                #print(data[connection])
                #print(data[connection].isnull())
                
                if not (data[connection].isnull()).all():
                    #print("Add connection: " + connection)
                    connections.append(connection)
                data = data.drop(connection, axis=1)
    
    elif isinstance(data, dict):
        for mud_connection in mud_data:
            if mud_connection in data:
                data.pop(mud_connection)
    #print(connections)
    return data, connections

def adjust_metrics_data( latency_df, service_dict, anomaly_mode, anomalies):
    if anomaly_mode is not None:
    
        if anomaly_mode == 'sliding_window':
            latency_df.drop(latency_df.head(1).index, inplace=True)
            for svc in service_dict:
                service_dict[svc].drop(service_dict[svc].head(1).index, inplace=True)
        elif anomaly_mode == 'base_data':
            latency_df.drop(latency_df.tail(1).index, inplace=True)
            for svc in service_dict:
                service_dict[svc].drop(service_dict[svc].tail(1).index, inplace=True)
        elif anomaly_mode =='sliding_window_improved':
            if len(anomalies) == 0:
                latency_df.drop(latency_df.head(1).index, inplace=True)
                for svc in service_dict:
                    service_dict[svc].drop(service_dict[svc].head(1).index, inplace=True)
            else:
                latency_df.drop(latency_df.tail(1).index, inplace=True)
                for svc in service_dict:
                    service_dict[svc].drop(service_dict[svc].tail(1).index, inplace=True)
        else:
            print("Anomaly mode not supported")
    return latency_df, service_dict

def run(prom_url, len_second, folder, config, mud_data, infinit_run):

    #Anomaly detection
    # Tuning parameters
    alpha = 0.55  # 0.55
    ad_threshold = 0.085  # 0.045

    interval_time = 15
    considered_timestamps = config['CONFIG']['N_TIMESTAMPS']#15
    anomaly_mode = config['CONFIG']['ANOMALY_MODE']
    latency_df = pd.DataFrame()
    service_dict = {}

    
    event_counter = 0
    anomaly_free_base_data = False

    DG = Metrics_collector.mpg(prom_url, folder)

    while infinit_run or len(latency_df) < considered_timestamps:
        print("\n--- Loop [" + str(len(latency_df)) + " timestamps, " + str(event_counter) + " event-counter]: " + str(time.asctime(time.localtime(time.time()))) + ", Base data ready? " + str(anomaly_free_base_data), flush=True)

        end_time = time.time()
        start_time = end_time - len_second
        latency_df = Metrics_collector.get_latency_row(prom_url, start_time, end_time, latency_df)
        #for item in latency_df:
            #print(item)
        latency_df, iot_connections = remove_IoT_device_metrics(latency_df, mud_data)
        service_dict = Metrics_collector.get_metrics_row(prom_url, start_time, end_time, service_dict)
        service_dict, dumpster = remove_IoT_device_metrics(service_dict, mud_data)

        if len(latency_df) >= considered_timestamps:

            #wait until a clean base data is available
            if not anomaly_free_base_data:
                latency_df, anomalies = Anomaly_detector.Anomaly_detection_loop(latency_df, ad_threshold, anomaly_mode)
                latency_df, service_dict = adjust_metrics_data( latency_df, service_dict, 'sliding_window', anomalies)
                if "front-end_orders" in anomalies:
                    anomalies.remove("front-end_orders")
                if len(anomalies) == 0:
                    print("Clean base data available")
                    anomaly_free_base_data = True
            else:
                #Fault injection handling
                if event_counter in config['EVENTS']:
                    os.system("py chaos-mesh.py --fault " + config['EVENTS'][event_counter])
                    print('***** Fault: ' + config['EVENTS'][event_counter] + ' started *****')

                #Anomaly detection handling
                if infinit_run:
                    latency_df, anomalies = Anomaly_detector.Anomaly_detection_loop(latency_df, ad_threshold, anomaly_mode)
                else:
                    latency_df, anomalies = Anomaly_detector.Anomaly_detection_loop(latency_df, ad_threshold, None)
                
                print("Anomalies: " + str(anomalies))
                if "front-end_orders" in anomalies:
                    anomalies.remove("front-end_orders")

                if(len(anomalies) > 0):
                    store_metrics_to_files(latency_df, service_dict, folder)

                    #RCA
                    anomaly_score = RCA_detector.anomaly_subgraph_2(DG, anomalies, latency_df, alpha, service_dict, mud_data, iot_connections)
                    print("The three most likely root causes")
                    anomaly_score.sort(key=lambda x: x[1], reverse=True)
                    print(anomaly_score[:3])
                
                latency_df, service_dict = adjust_metrics_data( latency_df, service_dict, anomaly_mode, anomalies)
                event_counter += 1
            

        wait_rest_of_interval_time(end_time, interval_time)


if __name__ == '__main__':
    args = parse_args()

    folder = args.folder
    len_second = args.length
    prom_url = args.url
    mud_folder = args.mud

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mud_data = MUD_handler.read_MUD_files(mud_folder)

    run(prom_url, len_second, folder, config, mud_data, True)
    
