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
            "nfs": {"path": host_path, "server": "k8s-node-1.idc-1.ten1010.io"},
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
    parser.add_argument("-l", "--log_dir_path", type=str, required=True)
    parser.add_argument("-m", "--model_dir_path", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "common"

    # create pv, pvc for pretrained-models
    name = "jsh-pretrained-models"
    access_mode = "ReadOnlyMany"
    create_persistent_volume(name=name, access_mode=access_mode, storage_size=500, host_path= args.model_dir_path)
    create_persistent_volume_claim(name=name, access_mode=access_mode, storage_size=500)

    # create pv, pvc for logs
    name = "jsh-logs"
    access_mode = "ReadWriteMany"
    create_persistent_volume(name=name, access_mode=access_mode, storage_size=50, host_path= args.log_dir_path)
    create_persistent_volume_claim(name=name, access_mode=access_mode, storage_size=50)
