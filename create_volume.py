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
    pv_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": "jsh-pv"},
        "spec": {
            "storageClassName": "llama2",
            "capacity": {"storage": f"{args.storage_size}Gi"},
            "accessModes": [access_mode],
            "nfs": {"path": args.host_path, "server": "k8s-node-1.idc-1.ten1010.io"},
        },
    }
    v1.create_persistent_volume(pv_manifest)


def create_persistent_volume_claim():
    # pvc_manifest = client.V1PersistentVolumeClaim(
    #     metadata=client.V1ObjectMeta(name="jsh-pvc"),
    #     spec=client.V1PersistentVolumeClaimSpec(
    #         storage_class_name="standard",
    #         access_modes=[access_mode],
    #         resources=client.V1ResourceRequirements(requests={"storage": f"{args.storage_size}Gi"}),
    #     ),
    # )
    pvc_manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "jsh-pvc"},
        "spec": {
            "storageClassName": "llama2",
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": f"{args.storage_size}Gi"}},
            "volumeName": "jsh-pv",
        },
    }
    v1.create_namespaced_persistent_volume_claim(namespace, pvc_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--storage_size", type=int, default=60)
    parser.add_argument("-p", "--host_path", type=str, required=True)
    args = parser.parse_args()

    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace = "jsh"
    access_mode = "ReadOnlyMany"

    create_persistent_volume()
    create_persistent_volume_claim()
