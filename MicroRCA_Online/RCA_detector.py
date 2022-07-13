from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Anomaly_detector import birch_ad_with_smoothing
import networkx as nx
from sklearn import preprocessing
from sklearn.cluster import Birch

def node_weight(svc, anomaly_graph, baseline_df, service_dict):

    #Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
#        print(u, v)
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg  = in_edges_weight_avg / num

    if svc not in service_dict:
        return in_edges_weight_avg, None

    #print(baseline_df)
    #filename = folder + "\\" + svc + '.csv'
    #df = pd.read_csv(filename)
    df = service_dict[svc]
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

#birch
def svc_personalization_1(svc, anomaly_graph, baseline_df, service_dict):

    df = service_dict[svc]

    #Get max outgoing edge to detect if the error is coming from somewhere
    max_edge = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True): #in edged because of inverted graph
        print("Weight edge: " + str(data["weight"]))
        if anomaly_graph.nodes[u]['type'] == 'service':
            print("t")
            if data["weight"] > max_edge:
                max_edge = data["weight"]

    #Check if there is a high possibility of the node being the anomalous service
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    considered_anomalous = 0
    metric = 'ctn_cpu'
    for col in ctn_cols:
        df[col] = df[col].rolling(window=12, min_periods=1).mean()
        x = np.array(df[col])
        x = np.where(np.isnan(x), 0, x)
        normalized_x = preprocessing.normalize([x])

        X = normalized_x.reshape(-1,1)

        brc = Birch(branching_factor=50, n_clusters=None, threshold=0.085, compute_labels=True)
        brc.fit(X)
        brc.predict(X)

        labels = brc.labels_
        n_clusters = np.unique(labels).size
        if n_clusters > 1:
            considered_anomalous = 1
            metric = col
    weight = (considered_anomalous + (1-max_edge))/2
    print(weight)
    return weight, metric

#correlation
def svc_personalization_2(svc, anomaly_graph, baseline_df, service_dict):
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
    if svc not in service_dict:
        #print(svc)
        return edges_weight_avg, None

    #filename = folder + "\\" + svc + '.csv'
    #df = pd.read_csv(filename)
    df = service_dict[svc]
    #print(svc)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col
    personalization = max_corr
    #print(edges_weight_avg)
    print(svc + " (" + str(metric) + "): " + str(personalization))

    return personalization, metric

#correlation with max weight
def svc_personalization_3(svc, anomaly_graph, baseline_df, service_dict):
        #Get max outgoing edge to detect if the error is coming from somewhere
    max_edge = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True): #in edged because of inverted graph
        print("Weight edge: " + str(data["weight"]))
        if anomaly_graph.nodes[u]['type'] == 'service':
            print("t")
            if data["weight"] > max_edge:
                max_edge = data["weight"]
    if svc not in service_dict:
        #print(svc)
        return max_edge, None

    #filename = folder + "\\" + svc + '.csv'
    #df = pd.read_csv(filename)
    df = service_dict[svc]
    #print(svc)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col
    personalization = max_edge * max_corr
    print(svc + " (" + str(metric) + "): " + str(personalization))


    #Testing
    personalization = (max_corr + (1-max_edge))/2
    return personalization, metric

def svc_personalization_4(svc, anomaly_graph, baseline_df, service_dict):
    abr_limit = 2.5

    df = service_dict[svc]

    #Get max outgoing edge to detect if the error is coming from somewhere
    max_edge = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True): #in edged because of inverted graph
        print("Weight edge: " + str(data["weight"]))
        if anomaly_graph.nodes[u]['type'] == 'service':
            print("t")
            if data["weight"] > max_edge:
                max_edge = data["weight"]

    #Check if there is a high possibility of the node being the anomalous service
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    considered_anomalous = 0
    metric = 'ctn_cpu'
    for col in ctn_cols:
        print(df[col].iloc[-1])
        print(df[col].iloc[-2])
        if df[col].iloc[-1]> abr_limit * df[col].iloc[-2]:
            print("TTTTTTTTtTTTT")
            considered_anomalous = 1

    weight = (considered_anomalous + (1-max_edge))/2
    print(weight)
    return weight, metric

def printgraph(graph, name):
    plt.figure(figsize=(20,20))
    #nx.draw(DG, with_labels=True, font_weight='bold')
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
    labels = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
    #plt.show()
    plt.savefig(name + '.png')

def anomaly_subgraph(DG, anomalies, latency_df, alpha, service_dict, MUD_data):
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

    printgraph(DG, "graph")
    
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

    #print('edge df:', edge_df)
    nodes = set(nodes)
    #print(nodes)
    #print(edges)

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
                    data, col = node_weight(u, anomaly_graph, baseline_df, service_dict)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']


    for node in nodes:
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, service_dict)
        personalization[node] = max_corr / anomaly_graph.degree(node)
#        print(node, personalization[node])

    anomaly_graph = anomaly_graph.reverse(copy=True)
#
    edges = list(anomaly_graph.edges(data=True))

    for u, v, d in edges: #???? for all nodes
        if anomaly_graph.nodes[node]['type'] == 'host':
            anomaly_graph.remove_edge(u,v)
            anomaly_graph.add_edge(v,u,weight=d['weight'])

    printgraph(anomaly_graph, "anomaly_graph")

#    personalization['shipping'] = 2
    #print('Personalization:', personalization)



    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score

class Calculation_methods(Enum):
    MIN = 1
    MAX = 2
    SUM = 3
    MUL = 4 #multiplay with a integer like weight *= 2

def calc_IoT_values(calculation_method, old_value, new_value, is_node = False):
    if calculation_method == Calculation_methods.MAX:
        return max(old_value, new_value)
    elif calculation_method == Calculation_methods.MIN:
        return min(old_value, new_value)
    elif calculation_method == Calculation_methods.SUM:
        return old_value + new_value
    elif calculation_method == Calculation_methods.MUL: 
        return new_value * old_value

def replace_or_add_edge(graph, u, v, weight):
    if graph.has_edge(u,v):
        #type_u = graph.nodes[u]['type']
        #type_v = graph.nodes[v]['type']
        graph.remove_edge(u,v)
    graph.add_edge(u,v, weight=weight)
        #graph.nodes[u]['type'] = type_u
        #graph.nodes[v]['type'] = type_v

def create_anomalous_subgraph(DG, nodes, edges, alpha, baseline_df, service_dict, latency_df, iot_connections, mud_data):
    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    anomalous_iot_nodes = {}

    for node in nodes:
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u,v)
            if edge in edges:
                data = alpha
            else:
                if DG.nodes[v]['type'] == 'host':
                    data, col = node_weight(u, anomaly_graph, baseline_df, service_dict)
                elif u + '_' + v in iot_connections:
                    data = alpha
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

            # Add all connections to IoT devices from anomalous nodes
            for iot_connection in iot_connections:
                
                if v in iot_connection and v == node or u in iot_connection and u == node:
                    edges = iot_connection.split('_')

                    #TODO should no be based on MUD data, perhaps weight already define
                    if edges[1] in mud_data:
                        iot_device = edges[1]
                        iot_service = edges[0]
                        rules = mud_data[iot_device]["to"]
                    elif edges[0] in mud_data:
                        iot_device = edges[0]
                        iot_service = edges[1]
                        rules = mud_data[iot_device]["from"]
                    else:
                        print("Missing Mud file")

                    anomalous_iot_nodes[iot_service] = iot_device
    print("Anomalous iot nodes:")
    print(anomalous_iot_nodes)

    #TODO
    for node in anomalous_iot_nodes:
        weight = 0
        
        for u, v, data in anomaly_graph.in_edges(node, data=True):
            # v = node
            weight = calc_IoT_values(Calculation_methods.MAX, weight, data['weight'])
        
        if weight == 0 and node in nodes: #connection to an anomalous node without incoming edges
            weight = alpha
        replace_or_add_edge(anomaly_graph, node, anomalous_iot_nodes[node], weight)
    
    anomaly_graph = anomaly_graph.reverse(copy=True)
    
    # Change direction of edges to host nodes
    edges = list(anomaly_graph.edges(data=True))
    for node in nodes:
        for u, v, d in edges:
            if anomaly_graph.nodes[node]['type'] == 'host':
                anomaly_graph.remove_edge(u,v)
                anomaly_graph.add_edge(v,u,weight=d['weight'])

    return anomaly_graph, anomalous_iot_nodes

def calculate_service_personalization(anomaly_graph, nodes, baseline_df, service_dict, version):
    # Set personalization with container resource usage and MUD rules
    personalization = {}
    for node in nodes:
        if version == 0:
            max_corr, col = svc_personalization_1(node, anomaly_graph, baseline_df, service_dict)
        elif version == 1:
            max_corr, col = svc_personalization_2(node, anomaly_graph, baseline_df, service_dict)
        elif version == 2:
            max_corr, col = svc_personalization_3(node, anomaly_graph, baseline_df, service_dict)
        elif version == 3:
            max_corr, col = svc_personalization_4(node, anomaly_graph, baseline_df, service_dict)
        personalization[node] = max_corr #/ anomaly_graph.degree(node)

    return personalization

def iot_edge_weight_calculations(anomaly_graph, nodes, anomalous_iot_nodes, mud_data, alpha):
    for node in nodes:
        if node in anomalous_iot_nodes:
            edges_to_be_replaced = []
            for u, v in anomaly_graph.in_edges(anomalous_iot_nodes[node]):
                #v: temperature-sensor
                if u not in mud_data[anomalous_iot_nodes[node]]["to"]:
                    #print("Violation_1: " + u + "_" + v)
                    edges_to_be_replaced.append((u, v, alpha))

            for u, v in anomaly_graph.out_edges(anomalous_iot_nodes[node]):
                #v: temperature-sensor
                if v not in mud_data[anomalous_iot_nodes[node]]["from"]:
                    #print("Violation_2: " + u + "_" + v)
                    edges_to_be_replaced.append((u, v, alpha))

            #This has to be done this way because we cannot change the anomaly_graph while iterating through it
            for u, v, weight in edges_to_be_replaced:
                #print(u + v + str(weight))
                replace_or_add_edge(anomaly_graph, u, v, weight)
            
def iot_personalization_weight_calculations(anomaly_graph, nodes, anomalous_iot_nodes, personalization, mud_data, alpha):
    for node in nodes:
        if node in anomalous_iot_nodes:
            personalization_weight = 1 - personalization[node]
            old_weight = 0
            if anomalous_iot_nodes[node] in personalization:
                old_weight = personalization[anomalous_iot_nodes[node]]
            personalization_weight = calc_IoT_values(Calculation_methods.SUM, old_weight, personalization_weight)

            for u, v in anomaly_graph.in_edges(anomalous_iot_nodes[node]):
                #v: temperature-sensor
                if u not in mud_data[anomalous_iot_nodes[node]]["to"]:
                    #print("Violation_1: " + u + "_" + v)
                    personalization_weight = calc_IoT_values(Calculation_methods.SUM, personalization_weight, alpha) #penalty value
                    
            for u, v in anomaly_graph.out_edges(anomalous_iot_nodes[node]):
                #v: temperature-sensor
                if v not in mud_data[anomalous_iot_nodes[node]]["from"]:
                    #print("Violation_2: " + u + "_" + v)
                    personalization_weight = calc_IoT_values(Calculation_methods.SUM, personalization_weight, alpha) #penalty value
            
            personalization[anomalous_iot_nodes[node]] = personalization_weight
    return personalization

def anomaly_subgraph_2(DG, anomalies, latency_df, alpha, service_dict, mud_data, iot_connections):


    printgraph(DG, "graph_2")
    
    # Get reported anomalous nodes
    edges = []
    nodes = set()
    baseline_df = pd.DataFrame()
    edge_df = {}
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
        svc = edge[1]
        nodes.add(svc)
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly


    anomaly_graph, anomalous_iot_nodes = create_anomalous_subgraph(DG, nodes, edges, alpha, baseline_df, service_dict, latency_df, iot_connections, mud_data)
    iot_edge_weight_calculations(anomaly_graph, nodes, anomalous_iot_nodes, mud_data, alpha)
    for i in range(4):
        personalization = calculate_service_personalization(anomaly_graph, nodes, baseline_df, service_dict, i)
        personalization = iot_personalization_weight_calculations(anomaly_graph, nodes, anomalous_iot_nodes, personalization, mud_data, alpha)
        

        #printgraph(anomaly_graph, "anomaly_graph_2")
        print("Personalization" + str(i) + ":")
        print(personalization)
        anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

        anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)
        print(anomaly_score)

#    return anomaly_graph
    return anomaly_score
