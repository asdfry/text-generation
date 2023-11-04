import json
import time
import argparse
import requests
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--instance", required=True)
parser.add_argument("-p", "--prometheus_url", required=True)
parser.add_argument("-s", "--suffix", required=True)
args = parser.parse_args()


def prometheus_query(query):
    params = {"query": query}

    response = requests.get(args.prometheus_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to query Prometheus. Status code: {response.status_code}")
        return None


query_nw = f"rate(node_network_receive_bytes_total{{instance='{args.instance}', device!~'^v.*'}}[10s])"
query_ib = f"rate(node_infiniband_port_data_received_bytes_total{{instance='{args.instance}', device=~'mlx5_[0-9]'}}[10s])"

while True:
    start_time = time.time()
    data_nw = prometheus_query(query_nw)
    data_ib = prometheus_query(query_ib)

    with open(f"mnt/network.jsonl", "a+") as f:
        json.dump(data_nw, f)
        f.write("\n")
        json.dump(data_ib, f)
        f.write("\n")

    print(f"{datetime.datetime.today()} Write json")
    time.sleep(10 - (time.time() - start_time))
