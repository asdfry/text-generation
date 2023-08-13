def write_master_config(network_addr: str, host_addr: int, port: int):
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host master\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port {port}\n")
        f.write(f"    User {user}\n")
        f.write(f"    IdentityFile {identity_file}\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("hostfile", "a") as f:
        f.write(f"master slots=1\n")


def write_worker_config(network_addr: str, host_addr: int, port: int):
    global worker_num
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host worker-{worker_num}\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port {port}\n")
        f.write(f"    User {user}\n")
        f.write(f"    IdentityFile {identity_file}\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("hostfile", "a") as f:
        f.write(f"worker-{worker_num} slots=1\n")
    worker_num += 1


def write_accelerate_config(network_addr: str, host_addr: int, worker_num: int):
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
            f"main_process_ip: {network_addr}.{host_addr}\n"
            f"main_process_port: 1040\n"
            f"main_training_function: main\n"
            f"mixed_precision: 'no'\n"
            f"num_machines: {worker_num}\n"
            f"num_processes: {worker_num}\n"
            f"rdzv_backend: static\n"
            f"same_network: true\n"
            f"tpu_env: []\n"
            f"tpu_use_cluster: false\n"
            f"tpu_use_sudo: false\n"
            f"use_cpu: false"
        )


if __name__ == "__main__":
    worker_num = 1
    user = input("Input User: ")
    identity_file = input("Input Identity File: ")
    slot_count = int(input("Input Slot Count: "))

    network_addr = input("Input Network Address: ")
    host_addr_master = int(input("Input Master's Host Address: "))

    write_master_config(network_addr, host_addr_master, 1041)
    for i in range(1, slot_count):
        write_worker_config(network_addr, host_addr_master + i, 1041 + i)

    while True:
        host_addr_worker = int(input("Input Worker's Host Address: "))
        if host_addr_worker == -1:
            print("Exit")
            break
        for i in range(0, slot_count):
            write_worker_config(network_addr, host_addr_worker + i, 1041 + i)

    write_accelerate_config(network_addr, host_addr_master, worker_num)
