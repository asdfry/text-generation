import argparse

from kubernetes import client, config


def create_pod(name: str, hostname: str, gpu: str):
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"ten1010.io/creator-name": "jsh"},
        },
        "spec": {
            "hostNetwork": True,
            "nodeSelector": {"kubernetes.io/hostname": hostname},
            "containers": [
                {
                    "name": "app",
                    "image": f"{args.image_name}",
                    # "imagePullPolicy": "Always",
                    "volumeMounts": [
                        {"name": "volume", "mountPath": "/root/mnt"},
                        {"name": "shmdir", "mountPath": "/dev/shm"},
                    ],
                    "command": [
                        "/bin/bash",
                        "-c",
                        f"/usr/sbin/sshd -p 1041 && sleep infinity",
                    ],
                    "ports": [{"containerPort": 1041, "hostPort": 1041, "protocol": "TCP"}],
                    "resources": {"limits": {gpu: str(args.slot_size)}},
                    "securityContext": {"privileged": True},
                },
            ],
            # "volumes": [
            #     {"name": "volume", "persistentVolumeClaim": {"claimName": "jsh-pvc"}},
            #     {"name": "shmdir", "emptyDir": {"medium": "Memory", "sizeLimit": "256M"}},
            # ],
            "volumes": [
                {"name": "volume", "hostPath": {"path": f"{args.volume_path}", "type": "Directory"}},
                {"name": "shmdir", "emptyDir": {"medium": "Memory", "sizeLimit": "256M"}},
            ],
        },
    }
    v1.create_namespaced_pod(namespace, pod_manifest)
    print(f"POD (name: {name}, node: {hostname}, gpu: {gpu} x {args.slot_size})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_name", type=str, required=True)
    parser.add_argument("-s", "--slot_size", type=int, required=True)
    parser.add_argument("-v", "--volume_path", type=str, required=True)
    parser.add_argument("-gm", "--gpu_master", type=str, required=True)
    parser.add_argument("-gw", "--gpu_worker", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()
    namespace = "common"

    with open("pods", "r") as f:
        lines = [i.strip() for i in f.readlines() if i]

    for idx, hostname in enumerate(lines):
        if idx == 0:
            create_pod(f"trainer-{idx+1}", hostname, args.gpu_master)
        else:
            create_pod(f"trainer-{idx+1}", hostname, args.gpu_worker)
