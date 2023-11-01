import argparse

from kubernetes import client, config


def create_persistent_volume(name: str, access_mode: str, storage_size: int, host_path: str):
    pv_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": f"{name}-pv"},
        "spec": {
            "storageClassName": "",
            "accessModes": [access_mode],
            "capacity": {"storage": f"{storage_size}Gi"},
            "nfs": {"path": host_path, "server": "pnode1.idc1.ten1010.io"},
        },
    }
    v1.create_persistent_volume(pv_manifest)
    print(f"PV (name: {name}-pv, capacity: {storage_size}Gi)")


def create_persistent_volume_claim(name: str, access_mode: str, storage_size: int):
    pvc_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": f"{name}-pvc"},
        "spec": {
            "storageClassName": "",
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": f"{storage_size}Gi"}},
            "volumeName": f"{name}-pv",
        },
    }
    v1.create_namespaced_persistent_volume_claim(namespace, pvc_manifest)
    print(f"PVC (name: {name}-pvc, request: {storage_size}Gi)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--storage_size", type=str, default=500)
    parser.add_argument("-v", "--volume_path", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "common"
    name = "jsh"
    access_mode = "ReadOnlyMany"

    create_persistent_volume(
        name=name,
        access_mode=access_mode,
        storage_size=args.storage_size,
        host_path=args.volume_path,
    )
    create_persistent_volume_claim(
        name=name,
        access_mode=access_mode,
        storage_size=args.storage_size,
    )
