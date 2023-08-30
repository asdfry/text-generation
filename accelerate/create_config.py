import os
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


def write_accelerate_config(master_addr: str, slot_size: int, total_node: int):
    os.makedirs(".cache/huggingface/accelerate", exist_ok=True)
    with open(".cache/huggingface/accelerate/default_config.yaml", "w+") as f:
        f.write(
            f"compute_environment: LOCAL_MACHINE\n"
            f"deepspeed_config:\n"
            f"  deepspeed_hostfile: /root/hostfile\n"
            f"  deepspeed_multinode_launcher: pdsh\n"
            f"  gradient_accumulation_steps: 1\n"
            f"  zero3_init_flag: false\n"
            f"  zero_stage: 0\n"
            f"distributed_type: DEEPSPEED\n"
            f"downcast_bf16: 'no'\n"
            f"machine_rank: 0\n"
            f"main_process_ip: {master_addr}\n"
            f"main_process_port: 1040\n"
            f"main_training_function: main\n"
            f"mixed_precision: 'no'\n"
            f"num_machines: {total_node}\n"
            f"num_processes: {slot_size * total_node}\n"
            f"rdzv_backend: static\n"
            f"same_network: true\n"
            f"tpu_env: []\n"
            f"tpu_use_cluster: false\n"
            f"tpu_use_sudo: false\n"
            f"use_cpu: false"
        )


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

    write_accelerate_config(addr, args.slot_size, args.total_node)
