#!/bin/bash

docker rm -f deploy-pod
docker build -t asdfry/python:3.10.6-slim-kube .
yes | docker prune iamge
sh ~/workspaces/gdrive/add-deploy/deploy_cont.sh
docker exec -it deploy-pod /bin/bash
