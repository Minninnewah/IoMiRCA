
import glob
import json
import os


def read_MUD_files(mud_folder):
    json_files = glob.glob(mud_folder + "\*.json")
    #print(json_files)
    mud_data = {}
    for json_file in json_files:
        device_name = os.path.splitext(os.path.basename(json_file))[0]
        print(device_name)
        f = open(json_file)
        data = json.load(f)
        acls = data["ietf-access-control-list:acls"]["acl"]
        general = data["ietf-mud:mud"]
        acl_name_from_device = general["from-device-policy"]["access-lists"]["access-list"][0]["name"]
        acl_name_to_device = general["to-device-policy"]["access-lists"]["access-list"][0]["name"]
        
        acl_from_device = [acl for acl in acls if acl["name"] == acl_name_from_device][0]
        acl_to_device = [acl for acl in acls if acl["name"] == acl_name_to_device][0]
        
        communication_from = [dns_name["matches"]["ipv4"]["ietf-acldns:dst-dnsname"] for dns_name in acl_from_device["aces"]["ace"]]
        communication_to = [dns_name["matches"]["ipv4"]["ietf-acldns:src-dnsname"] for dns_name in acl_to_device["aces"]["ace"]]


        #communication_dns_names = set(communication_from).union(set(communication_to))
        mud_data[device_name] = { "from": communication_from, "to": communication_to}
    print(mud_data)
    return mud_data