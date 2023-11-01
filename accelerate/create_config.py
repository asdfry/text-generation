import os
import argparse


def write_worker_config(num: int, network_addr: str, host_addr: int, slot_size: int):
    hostname = f"{network_addr}.{host_addr}"
    with open(".ssh/config", "a") as f:
        f.write(f"Host trainer-{num}\n")
        f.write(f"    HostName {hostname}\n")
        f.write(f"    Port 1041\n")
        f.write(f"    User root\n")
        f.write(f"    IdentityFile /root/.ssh/key.pem\n")
        f.write(f"    StrictHostKeyChecking no\n\n")
    with open("hostfile", "a") as f:
        f.write(f"trainer-{num} slots={slot_size}\n")
    print(f"NODE (name: trainer-{num}, addr: {hostname}, slot: {slot_size})")


def write_accelerate_config(master_addr: str, slot_size: int, total_node: int):
    os.makedirs(".cache/huggingface/accelerate", exist_ok=True)
    with open(".cache/huggingface/accelerate/default_config.yaml", "w+") as f:
        f.write(
            f"compute_environment: LOCAL_MACHINE\n"
            f"deepspeed_config:\n"
            f"  deepspeed_hostfile: /root/hostfile\n"
            f"  deepspeed_multinode_launcher: pdsh\n"
            f"  gradient_accumulation_steps: 1\n"
            f"  gradient_clipping: 1.0\n"
            f"  offload_optimizer_device: cpu\n"
            f"  offload_param_device: cpu\n"
            f"  zero3_init_flag: true\n"
            f"  zero3_save_16bit_model: true"
            f"  zero_stage: 3\n"
            f"distributed_type: DEEPSPEED\n"
            f"downcast_bf16: 'no'\n"
            f"machine_rank: 0\n"
            f"main_process_ip: {master_addr}\n"
            f"main_process_port: 1040\n"
            f"main_training_function: main\n"
            f"mixed_precision: 'fp16'\n"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-t", "--total_node", type=int, required=True)
    parser.add_argument("-m", "--master_addr", type=str, required=True)
    args = parser.parse_args()

    addr = args.master_addr
    network_addr = addr[: addr.rfind(".")]
    host_addr = int(addr.split(".")[-1])

    for i in range(args.total_node):
        write_worker_config(i + 1, network_addr, host_addr + i, args.slot_size)

    write_accelerate_config(addr, args.slot_size, args.total_node)
