import json
import time
import argparse
import requests
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--instance", required=True)
parser.add_argument("-p", "--prometheus_url", required=True)
parser.add_argument("-s", "--save_path", required=True)
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


queries = []
queries.append(f"rate(node_network_receive_bytes_total{{instance='{args.instance}', device!~'^v.*'}}[10s])")
queries.append(f"rate(node_network_transmit_bytes_total{{instance='{args.instance}', device!~'^v.*'}}[10s])")
queries.append(f"rate(node_infiniband_port_data_received_bytes_total{{instance='{args.instance}', device=~'mlx5_[0-9]'}}[10s])")
queries.append(f"rate(node_infiniband_port_data_transmitted_bytes_total{{instance='{args.instance}', device=~'mlx5_[0-9]'}}[10s])")


while True:
    start_time = time.time()

    with open(f"mnt/output/{args.save_path}/network.jsonl", "a+") as f:
        for query in queries:
            json.dump(prometheus_query(query), f)
            f.write("\n")

    print(f"{datetime.datetime.today()} Write json")
    time.sleep(10 - (time.time() - start_time))
