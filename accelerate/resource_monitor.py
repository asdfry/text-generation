import re
import json
import time
import psutil
import socket
import GPUtil

from glob import glob
from datetime import datetime, timedelta, timezone
from threading import Thread


class ResourceMonitor(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.KST = timezone(timedelta(hours=9))
        self.paths = [i for i in glob("/sys/class/infiniband/mlx*") if re.search(r"mlx\d_\d+", i)]
        self.cnt_names = ["port_xmit_data", "port_xmit_packets", "port_rcv_data", "port_rcv_packets"]
        self.latest = {i.split("/")[-1]: {} for i in self.paths}

        for path in self.paths:
            hca = path.split("/")[-1]
            for cnt_name in self.cnt_names:
                with open(f"{path}/ports/1/counters/{cnt_name}", "r") as f:
                    cnt = int(f.read().strip())
                    self.latest[hca][cnt_name] = cnt

    def run(self):
        while True:
            start_time = time.perf_counter()

            resource = {
                "time": datetime.now(self.KST).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                "cpu": {},
                "gpu": {},
                "memory": {},
                "infiniband": {},
            }

            resource["cpu"]["percent"] = psutil.cpu_percent()

            for idx, gpu in enumerate(GPUtil.getGPUs()):
                resource["gpu"][idx] = {"total": gpu.memoryTotal, "used": gpu.memoryUsed, "percent": gpu.load}

            memory_info = psutil.virtual_memory()
            resource["memory"] = {
                "total": memory_info.total,
                "used": memory_info.used,
                "percent": memory_info.percent,
            }

            for path in self.paths:
                hca = path.split("/")[-1]
                resource["infiniband"][hca] = {}
                for cnt_name in self.cnt_names:
                    with open(f"{path}/ports/1/counters/{cnt_name}", "r") as f:
                        cnt = int(f.read().strip())
                        resource["infiniband"][hca][cnt_name] = cnt - self.latest[hca][cnt_name]
                        self.latest[hca][cnt_name] = cnt

            time.sleep(1 - (time.perf_counter() - start_time))

            with open(f"logs/resource.{socket.gethostname()}.jsonl", "a+") as f:
                f.write(json.dumps(resource) + "\n")
