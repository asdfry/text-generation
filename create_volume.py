import argparse

from kubernetes import client, config


def create_persistent_volume():
    # pv_manifest = client.V1PersistentVolume(
    #     metadata=client.V1ObjectMeta(name="jsh-pv"),
    #     spec=client.V1PersistentVolumeSpec(
    #         storage_class_name="standard",
    #         capacity={"storage": f"{args.storage_size}Gi"},
    #         access_modes=[access_mode],
    #         host_path=client.V1HostPathVolumeSource(path="/data/my-pv")
    #     )
    # )
    name = "jsh-pv"
    pv_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": name},
        "spec": {
            "storageClassName": "pretrained-models",
            "capacity": {"storage": f"{args.storage_size}Gi"},
            "accessModes": [access_mode],
            "nfs": {"path": args.host_path, "server": "k8s-node-1.idc-1.ten1010.io"},
        },
    }
    v1.create_persistent_volume(pv_manifest)
    print(f"PV (name: {name}, capacity: {args.storage_size}Gi)")


def create_persistent_volume_claim():
    # pvc_manifest = client.V1PersistentVolumeClaim(
    #     metadata=client.V1ObjectMeta(name="jsh-pvc"),
    #     spec=client.V1PersistentVolumeClaimSpec(
    #         storage_class_name="standard",
    #         access_modes=[access_mode],
    #         resources=client.V1ResourceRequirements(requests={"storage": f"{args.storage_size}Gi"}),
    #     ),
    # )
    name = "jsh-pvc"
    pvc_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": name},
        "spec": {
            "storageClassName": "pretrained-models",
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": f"{args.storage_size}Gi"}},
            "volumeName": "jsh-pv",
        },
    }
    v1.create_namespaced_persistent_volume_claim(namespace, pvc_manifest)
    print(f"PVC (name: {name}, request: {args.storage_size}Gi)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--storage_size", type=int, default=100)
    parser.add_argument("-p", "--host_path", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "common"
    access_mode = "ReadOnlyMany"

    create_persistent_volume()
    create_persistent_volume_claim()
