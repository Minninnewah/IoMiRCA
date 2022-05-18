import argparse
import sys
import time
from turtle import fd
from paramiko import SSHClient, RSAKey, AutoAddPolicy
import requests




def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--fault', type=str, required=False,
                        default='memory',
                        choices=['memory', 'cpu', 'delay'],
                        help='fault type that should be injected')

    parser.add_argument('--pkey', type=str, required=False,
                        default=r'C:\Users\Zeno\SynologyDrive\Master\PA 1\kubernetes-pki.pem',
                        help='Path to the pem file of the VM')

    parser.add_argument('--host', type=str, required=False,
                        default='10.161.2.161',
                        help='IP address or hostname of VM')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    fault_type = args.fault

#    if fault_type == 'mud':
#        response = requests.get("http://" + args.host + ":3002/enable")
#        if response.status_code == 200:
#            print("Evil iot-handler is enabled")
#            time.sleep(60)
#            response = requests.get("http://" + args.host + ":3002/disable")
#            if response.status_code == 200:
#                print("Evil iot-handler is disabled")
#        sys.exit(0)

    client = SSHClient()
    key = RSAKey.from_private_key_file(args.pkey)
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(hostname=args.host, username='ubuntu', pkey=key)
    ftp = client.open_sftp()
    experiment_config = 'delay_experiment.yaml'

    if fault_type == 'memory':
        experiment_config = 'memory_experiment.yaml'
    elif fault_type == 'cpu':
        experiment_config = 'cpu_experiment.yaml'

    stdin, stdout, sterr = client.exec_command('kubectl delete -f ' + experiment_config)
    channel = stdout.channel
    status = channel.recv_exit_status()
    ftp.put('chaos_experiments/' + experiment_config, experiment_config)
    stdin, stdout, sterr = client.exec_command('kubectl apply -f ' + experiment_config)
    #print(stdout.readlines())
    #print(sterr.readlines())
    channel = stdout.channel
    status = channel.recv_exit_status()
    print('Configuration: ' + experiment_config + "is deployed")

    ftp.close()
    client.close()
