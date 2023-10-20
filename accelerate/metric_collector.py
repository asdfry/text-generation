import time
import pandas as pd
import requests

from typing import Dict
from datetime import datetime
from threading import Thread


class MetricCollector(Thread):
    def __init__(
        self,
        prometheus_ip: str,
        node_address: str,
        hostname: str,
        dirpath: str,
        logger,
    ) -> None:
        super().__init__()
        self.stop_flag = False
        self.prometheus_url = f"{prometheus_ip}/api/v1/query"
        self.node_address = node_address
        self.hostname = hostname
        self.dirpath = dirpath
        self.logger = logger
        self.logger.info("Init metric collector")

    def prometheus_query(self, query) -> dict | None:
        params = {"query": query}
        response = requests.get(self.prometheus_url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            self.logger.error(f"Failed to query Prometheus. Status code: {response.status_code}")
            return None

    def get_cpu_metrics(self, node_address) -> Dict:
        metrics = {}
        queries = {
            "cpu_util": f'100 - avg(rate(node_cpu_seconds_total{{mode="idle", instance="{node_address}"}}[10s])) * 100',
            "cpu_mem": f'node_memory_MemTotal_bytes{{instance="{node_address}"}} - node_memory_MemAvailable_bytes{{instance="{node_address}"}}',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            try:
                metrics[metric] = result["data"]["result"][0]["value"][1]
            except:
                metrics[metric] = None

        return metrics

    def get_gpu_metrics(self, hostname) -> Dict:
        metrics = {}
        queries = {
            "util": f'DCGM_FI_DEV_GPU_UTIL{{Hostname="{hostname}"}}',
            "mem": f'DCGM_FI_DEV_MEM_COPY_UTIL{{Hostname="{hostname}"}}',
            "power": f'DCGM_FI_DEV_POWER_USAGE{{Hostname="{hostname}"}}',
            "temp": f'DCGM_FI_DEV_GPU_TEMP{{Hostname="{hostname}"}}',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            for i in result["data"]["result"]:
                try:
                    metrics[f"gpu_{i['metric']['gpu']}_{metric}"] = i["value"][1]
                except:
                    metrics[f"gpu_{i['metric']['gpu']}_{metric}"] = None

        return metrics

    def get_disk_metrics(self, node_address) -> Dict:
        metrics = {}
        queries = {
            "disk_read": f'rate(node_disk_read_bytes_total{{instance="{node_address}",device="dm-0"}}[10s])',
            "disk_write": f'rate(node_disk_written_bytes_total{{instance="{node_address}",device="dm-0"}}[10s])',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            try:
                metrics[metric] = result["data"]["result"][0]["value"][1]
            except:
                metrics[metric] = None

        return metrics

    def run(self) -> None:
        self.logger.info("Start metric collector")
        rows = []

        while not self.stop_flag:
            row = {"time": datetime.now()}
            start_time = time.time()
            row.update(self.get_cpu_metrics(self.node_address))
            row.update(self.get_gpu_metrics(self.hostname))
            row.update(self.get_disk_metrics(self.node_address))
            rows.append(row)

            secs = 5 - (time.time() - start_time)
            time.sleep(secs)

        self.logger.info("Stop metric collector")

        df = pd.DataFrame(rows)
        csv_path = f"{self.dirpath}/metrics.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Save metric.csv (path: {csv_path})")

        df = pd.read_csv(csv_path)
        for col in df.columns:
            if not col == "time":
                self.logger.info(
                    f"[{col}] min: {df[col].min():.2f}, "
                    f"max: {df[col].max():.2f}, "
                    f"avg: {df[col].mean():.2f}"
                )

    def stop(self) -> None:
        self.stop_flag = True
