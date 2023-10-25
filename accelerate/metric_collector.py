import time
import pandas as pd
import requests

from typing import Dict
from datetime import datetime
from threading import Thread
from logger_main import logger


class MetricCollector(Thread):
    def __init__(self, prometheus_ip: str, target_node_ip: str, dirpath: str) -> None:
        super().__init__()
        self.stop_flag = False
        self.prometheus_url = f"http://{prometheus_ip}:30003/api/v1/query"
        self.target_node = f"{target_node_ip}"
        self.dirpath = dirpath
        logger.info("Init metric collector")

    def prometheus_query(self, query) -> dict | None:
        params = {"query": query}
        response = requests.get(self.prometheus_url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Failed to query Prometheus. Status code: {response.status_code}")
            return None

    def get_cpu_metrics(self) -> Dict:
        instance = f"{self.target_node}:9100"
        metrics = {}
        queries = {
            "cpu_util": f'100 - avg(rate(node_cpu_seconds_total{{mode="idle", instance="{instance}"}}[10s])) * 100',
            "cpu_mem": f'node_memory_MemTotal_bytes{{instance="{instance}"}} - node_memory_MemAvailable_bytes{{instance="{instance}"}}',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            try:
                metrics[metric] = result["data"]["result"][0]["value"][1]
            except:
                metrics[metric] = None

        return metrics

    def get_gpu_metrics(self) -> Dict:
        instance = f"{self.target_node}:9400"
        metrics = {}
        queries = {
            "util": f'DCGM_FI_DEV_GPU_UTIL{{instance="{instance}"}}',
            "mem": f'DCGM_FI_DEV_MEM_COPY_UTIL{{instance="{instance}"}}',
            "power": f'DCGM_FI_DEV_POWER_USAGE{{instance="{instance}"}}',
            "temp": f'DCGM_FI_DEV_GPU_TEMP{{instance="{instance}"}}',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            for i in result["data"]["result"]:
                try:
                    metrics[f"gpu_{i['metric']['gpu']}_{metric}"] = i["value"][1]
                except:
                    metrics[f"gpu_{i['metric']['gpu']}_{metric}"] = None

        return metrics

    def get_disk_metrics(self) -> Dict:
        instance = f"{self.target_node}:9100"
        metrics = {}
        queries = {
            "disk_read": f'rate(node_disk_read_bytes_total{{instance="{instance}",device="dm-0"}}[10s])',
            "disk_write": f'rate(node_disk_written_bytes_total{{instance="{instance}",device="dm-0"}}[10s])',
        }

        for metric, query in queries.items():
            result = self.prometheus_query(query)
            try:
                metrics[metric] = result["data"]["result"][0]["value"][1]
            except:
                metrics[metric] = None

        return metrics

    def run(self) -> None:
        logger.info("Start metric collector")
        rows = []

        while not self.stop_flag:
            start_time = time.time()
            row = {"time": datetime.now()}
            row.update(self.get_cpu_metrics())
            row.update(self.get_gpu_metrics())
            row.update(self.get_disk_metrics())
            rows.append(row)
            logger.info("Get metrics")

            secs = 5 - (time.time() - start_time)
            time.sleep(secs)

        logger.info("Stop metric collector")

        csv_path = f"{self.dirpath}/metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Save metric.csv (path: {csv_path})")

        df = pd.read_csv(csv_path)
        for col in df.columns:
            if not col == "time":
                logger.info(
                    f"[{col}] min: {df[col].min():.2f}, "
                    f"max: {df[col].max():.2f}, "
                    f"avg: {df[col].mean():.2f}"
                )

    def stop(self) -> None:
        self.stop_flag = True
