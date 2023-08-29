import argparse

from kubernetes import client, config


def create_master(node, port, gpu, slot, model_dir_path):
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
                    "image": f"{args.image_name}-{args.version}",
                    # "env": [{"name": "NCCL_SOCKET_IFNAME", "value": f"ib{port-1040}"}],
                    # "imagePullPolicy": "Always",
                    "volumeMounts": [
                        {
                            "name": "pretrained-models",
                            "mountPath": "/root/pretrained-models",
                        }
                    ],
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    # "ports": [{"containerPort": port, "hostIP": ip, "hostPort": port}],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: str(slot),
                        }
                    },
                },
            ],
            # "volumes": [{"name": "pretrained-models", "persistentVolumeClaim": {"claimName": "jsh-pvc"}}],
            "volumes": [
                {
                    "name": "pretrained-models",
                    "hostPath": {"path": f"{model_dir_path}", "type": "Directory"},
                }
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    print(f"POD (name: {pod_manifest['metadata']['name']}, node: {node}, port: {port})")


def create_worker(node, port, gpu, slot, model_dir_path):
    global worker_num
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
                    "image": f"{args.image_name}-{args.version}",
                    # "env": [{"name": "NCCL_SOCKET_IFNAME", "value": f"ib{port-1040}"}],
                    # "imagePullPolicy": "Always",
                    "volumeMounts": [
                        {
                            "name": "pretrained-models",
                            "mountPath": "/root/pretrained-models",
                        }
                    ],
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p {port} && sleep infinity",
                    ],
                    # "ports": [{"containerPort": port, "hostIP": ip, "hostPort": port}],
                    "ports": [{"containerPort": port}],
                    "resources": {
                        "limits": {
                            gpu: str(slot),
                        }
                    },
                },
            ],
            # "volumes": [{"name": "pretrained-models", "persistentVolumeClaim": {"claimName": "jsh-pvc"}}],
            "volumes": [
                {
                    "name": "pretrained-models",
                    "hostPath": {"path": f"{model_dir_path}", "type": "Directory"},
                }
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    print(f"POD (name: {pod_manifest['metadata']['name']}, node: {node}, port: {port})")
    worker_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_name", type=str, required=True)
    parser.add_argument("-n", "--node_prefix", type=str, required=True)
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-t", "--total_node", type=int, required=True)
    parser.add_argument("-v", "--version", type=str, required=True)
    parser.add_argument("-z", "--zero_fill", type=int, required=True)
    # parser.add_argument("-ma", "--master_addr", type=str, required=True)
    parser.add_argument("-mn", "--master_node_num", type=int, required=True)
    parser.add_argument("-mp", "--model_dir_path", type=str, required=True)
    parser.add_argument("-gm", "--gpu_master", type=str, required=True)
    parser.add_argument("-gw", "--gpu_worker", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "common"
    worker_num = 1

    node_num = str(args.master_node_num).zfill(args.zero_fill)
    node = f"{args.node_prefix}{node_num}"
    create_master(node, 1041, args.gpu_master, args.slot_size, args.model_dir_path)

    for i in range(1, args.total_node):
        node_num = str(args.master_node_num + i).zfill(args.zero_fill)
        node = f"{args.node_prefix}{node_num}"
        create_worker(node, 1041, args.gpu_worker, args.slot_size, args.model_dir_path)
