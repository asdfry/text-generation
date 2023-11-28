import os
import sys
import shutil
import psutil

from loguru import logger


def update_logger_config(dirpath):
    filepath = f"{dirpath}/torch.log"
    if os.path.exists(filepath):
        os.remove(filepath)
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}<white>:</white>{line}</cyan> | {message}",
                "colorize": "true",
            },
            {
                "sink": filepath,
                "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{line} | {message}",
            },
        ]
    }
    logger.configure(**config)


def move_nccl_outputs(dirpath):
    try:
        shutil.move(os.getenv("NCCL_DEBUG_FILE"), f"{dirpath}/nccl-debug.log")
        shutil.move(os.getenv("NCCL_TOPO_DUMP_FILE"), f"{dirpath}/nccl-topo.xml")
    except:
        pass


def get_io(last_io, eth, rdma):
    if eth:
        cur_io = get_ethernet_io(eth)
        real_io = cur_io.copy()
        real_io["rcv"] -= last_io["rcv"]
        real_io["xmit"] -= last_io["xmit"]
        return real_io, cur_io
    elif rdma:
        cur_io = get_rdma_io(rdma)
        real_io = cur_io.copy()
        real_io["rcv"] -= last_io["rcv"]
        real_io["xmit"] -= last_io["xmit"]
        return real_io, cur_io
    else:
        return {"rcv": -1, "xmit": -1}, {"rcv": -1, "xmit": -1}


def get_ethernet_io(if_name):
    io_stats = psutil.net_io_counters(pernic=True)
    interface_stats = io_stats.get(if_name)
    return {"rcv": interface_stats.bytes_recv, "xmit": interface_stats.bytes_sent}


def get_rdma_io(hca_name):
    with open(f"/sys/class/infiniband/{hca_name}/ports/1/counters/port_rcv_data", "r") as f:
        cnt_rcv = int(f.read().strip())
    with open(f"/sys/class/infiniband/{hca_name}/ports/1/counters/port_xmit_data", "r") as f:
        cnt_xmit = int(f.read().strip())
    return {"rcv": cnt_rcv, "xmit": cnt_xmit}
