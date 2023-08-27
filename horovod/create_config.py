import argparse


def write_master_config(network_addr: str, slot_size: int):
    global host_addr
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host master\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port 1041\n")
        f.write(f"    User root\n")
        f.write(f"    IdentityFile /root/.ssh/key.pem\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("hostfile", "a") as f:
        f.write(f"master slots={slot_size}\n")
    print(f"NODE (name: master, addr: {hostname}, slot: {slot_size})")
    host_addr += 1


def write_worker_config(network_addr: str, slot_size: int):
    global worker_num, host_addr
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host worker-{worker_num}\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port 1041\n")
        f.write(f"    User root\n")
        f.write(f"    IdentityFile /root/.ssh/key.pem\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("hostfile", "a") as f:
        f.write(f"worker-{worker_num} slots={slot_size}\n")
    print(f"NODE (name: worker-{worker_num}, addr: {hostname}, slot: {slot_size})")
    host_addr += 1
    worker_num += 1


if __name__ == "__main__":
    worker_num = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-t", "--total_node", type=int, required=True)
    parser.add_argument("-m", "--master_addr", type=str, required=True)
    args = parser.parse_args()

    addr = args.master_addr
    network_addr = addr[: addr.rfind(".")]
    host_addr = int(addr.split(".")[-1])

    write_master_config(network_addr, args.slot_size)

    for _ in range(1, args.total_node):
        write_worker_config(network_addr, args.slot_size)
