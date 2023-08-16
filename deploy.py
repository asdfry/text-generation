import argparse

from kubernetes import client, config


def create_master(node, port, gpu):
    global host_addr
    ip = f"{network_addr}.{host_addr}"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "master",
            "namespace": namespace,
            "labels": {"ten1010.io/creator-name": "jsh"},
        },
        "spec": {
            "hostNetwork": True,
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "app",
                    "image": f"{args.image_name}-master-{args.version}",
                    "env": [{"name": "NCCL_SOCKET_IFNAME", "value": f"ib{port-1040}"}],
                    # "imagePullPolicy": "Always",
                    "volumeMounts": [{"name": "pretrained-models", "mountPath": "/root/pretrained-models"}],
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    "ports": [{"containerPort": port, "hostIP": ip, "hostPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
            "volumes": [{"name": "pretrained-models", "persistentVolumeClaim": {"claimName": "jsh-pvc"}}],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    print(f"POD (name: {pod_manifest['metadata']['name']}, node: {node}, ip: {ip}, port: {port})")
    host_addr += 1


def create_worker(node, port, gpu):
    global worker_num, host_addr
    ip = f"{network_addr}.{host_addr}"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"worker-{worker_num}",
            "namespace": namespace,
            "labels": {"ten1010.io/creator-name": "jsh"},
        },
        "spec": {
            "hostNetwork": True,
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "app",
                    "image": f"{args.image_name}-worker-{args.version}",
                    "env": [{"name": "NCCL_SOCKET_IFNAME", "value": f"ib{port-1040}"}],
                    # "imagePullPolicy": "Always",
                    "volumeMounts": [{"name": "pretrained-models", "mountPath": "/root/pretrained-models"}],
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    "ports": [{"containerPort": port, "hostIP": ip, "hostPort": port}],
                    "resources": {
                        "limits": {
                            gpu: "1",
                        }
                    },
                },
            ],
            "volumes": [{"name": "pretrained-models", "persistentVolumeClaim": {"claimName": "jsh-pvc"}}],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    print(f"POD (name: {pod_manifest['metadata']['name']}, node: {node}, ip: {ip}, port: {port})")
    host_addr += 1
    worker_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_name", type=str, required=True)
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-t", "--total_node", type=int, required=True)
    parser.add_argument("-v", "--version", type=str, required=True)
    parser.add_argument("-ma", "--master_addr", type=str, required=True)
    parser.add_argument("-mn", "--master_node_num", type=int, required=True)
    parser.add_argument("-gm", "--gpu_master", type=str, required=True)
    parser.add_argument("-gw", "--gpu_worker", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "common"
    worker_num = 1
    addr = args.master_addr
    network_addr = addr[: addr.rfind(".")]
    host_addr = int(addr.split(".")[-1])

    node = f"k8s-node-{args.master_node_num}"
    create_master(node, 1041, args.gpu_master)
    for i in range(1, args.slot_size):
        create_worker(node, 1041 + i, args.gpu_master)

    for i in range(1, args.total_node):
        node = f"k8s-node-{args.master_node_num + i}"
        for i in range(0, args.slot_size):
            create_worker(node, 1041 + i, args.gpu_worker)
