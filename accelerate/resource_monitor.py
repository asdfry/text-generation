import os
import re
import json
import time
# import psutil
import socket
import GPUtil

from glob import glob
from datetime import datetime, timedelta, timezone
from threading import Thread


class ResourceMonitor(Thread):
    def __init__(self, dirpath) -> None:
        super().__init__()
        self.stop_flag = False
        self.KST = timezone(timedelta(hours=9))
        self.paths = [i for i in glob("/sys/class/infiniband/mlx*") if re.search(r"mlx\d_\d+", i)]
        self.cnt_names = ["port_xmit_data", "port_xmit_packets", "port_rcv_data", "port_rcv_packets"]
        self.last_cnt = {i.split("/")[-1]: {} for i in self.paths}

        for path in self.paths:
            hca = path.split("/")[-1]
            for cnt_name in self.cnt_names:
                with open(f"{path}/ports/1/counters/{cnt_name}", "r") as f:
                    cnt = int(f.read().strip())
                    self.last_cnt[hca][cnt_name] = cnt

        self.jsonl = f"{dirpath}/resource.{socket.gethostname()}.jsonl"
        if not os.path.isdir(dirpath):  # only required in local
            os.makedirs(dirpath, exist_ok=True)
        if os.path.isfile(self.jsonl):
            os.remove(self.jsonl)

    def run(self):
        while not self.stop_flag:
            start_time = time.perf_counter()

            resource = {
                "time": datetime.now(self.KST).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                # "cpu": {},
                "gpu": {},
                # "memory": {},
                "infiniband": {},
            }

            # resource["cpu"]["percent"] = psutil.cpu_percent()

            for idx, gpu in enumerate(GPUtil.getGPUs()):
                resource["gpu"][idx] = {"total": gpu.memoryTotal, "used": gpu.memoryUsed, "percent": gpu.load}

            # memory_info = psutil.virtual_memory()
            # resource["memory"] = {
            #     "total": memory_info.total,
            #     "used": memory_info.used,
            #     "percent": memory_info.percent,
            # }

            for path in self.paths:
                hca = path.split("/")[-1]
                resource["infiniband"][hca] = {}
                for cnt_name in self.cnt_names:
                    with open(f"{path}/ports/1/counters/{cnt_name}", "r") as f:
                        cnt = int(f.read().strip())
                        resource["infiniband"][hca][cnt_name] = cnt - self.last_cnt[hca][cnt_name]
                        self.last_cnt[hca][cnt_name] = cnt

            sec = 1 - (time.perf_counter() - start_time)
            if sec < 0:
                time.sleep(1)
            else:
                time.sleep(sec)

            with open(self.jsonl, "a+") as f:
                f.write(json.dumps(resource) + "\n")

    def stop(self):
        self.stop_flag = True
