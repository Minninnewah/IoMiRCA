#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: li
"""

import argparse
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

    DG = Metrics_collector.mpg(prom_url, folder)

    while infinit_run or len(latency_df) < considered_timestamps:
        print("--- Loop [" + str(len(latency_df)) + " timestamps, " + str(event_counter) + " event-counter]: " + str(time.asctime(time.localtime(time.time()))), flush=True)

        end_time = time.time()
        start_time = end_time - len_second
        latency_df, service_dict = Metrics_collector.get_metrics_row(prom_url, start_time, end_time, latency_df, service_dict)

        if len(latency_df) >= considered_timestamps:
            #Fault injection handling
            if event_counter in config['EVENTS']:
                os.system("py chaos-mesh.py --fault " + config['EVENTS'][event_counter])
                print('***** Fault: ' + config['EVENTS'][event_counter] + ' started *****')

            #Anomaly detection handling
            if infinit_run:
                latency_df, anomalies = Anomaly_detector.Anomaly_detection_loop(latency_df, ad_threshold, anomaly_mode)
            else:
                latency_df, anomalies = Anomaly_detector.Anomaly_detection_loop(latency_df, ad_threshold, None)

            # latency_df = generate_latency_values(latency_df, amount_timestamps=5, nan_values=True, fault_injection=True)
            store_metrics_to_files(latency_df, service_dict, folder)
            
            if(len(anomalies) > 0):
                #RCA
                anomaly_score = RCA_detector.anomaly_subgraph(DG, anomalies, latency_df, alpha, mud_data)
                print(anomaly_score)

        wait_rest_of_interval_time(end_time, interval_time)

        
        event_counter += 1

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

    run(prom_url, len_second, folder, config, mud_data, False)
    
