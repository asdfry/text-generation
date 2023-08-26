import grequests
import requests

from glob import glob
from tqdm import tqdm
from loguru import logger


logger.add("logs/log_{time}.log")

model_names = sorted(glob("/gpfs/user/jsh/*"))
model_names = [i.split("/")[-1] for i in model_names]
model_names.remove("LLaMA-2-7B-32K")

large_model_names = ["falcon-40b", "Llama-2-70b-chat-hf"]
for large_model_name in large_model_names:
    model_names.remove(large_model_name)

for model_name in model_names:
    logger.info(f"Model: {model_name}")
    headers = {"Content-Type": "application/json"}

    urls = [f"http://127.0.0.1:{1041 + i}/load-model" for i in range(8)]
    rs = (grequests.post(url, json={"name": model_name}, headers=headers) for url in urls)
    results = [i.json() for i in grequests.map(rs)]
    ets = [i["elapsed_time"] for i in results]
    logger.info(
        f"[Load] elapsed time: {round(sum(ets)/len(ets), 2)} sec, "
        f"used memory: {round(results[0]['used_memory'], 2)} GiB"
    )

    urls = [f"http://127.0.0.1:{1041 + i}/infer" for i in range(8)]
    for max_token in [1024, 2048, 4096]:
        etc = []
        ums = []
        for _ in tqdm(range(10)):
            rs = (
                grequests.post(
                    url,
                    json={"sentences": ["What is deep learning?"], "max_token": max_token},
                    headers=headers,
                )
                for url in urls
            )
            results = [i.json() for i in grequests.map(rs)]
            ets += [i["elapsed_time"] for i in results]
            ums += [i["used_memory"] for i in results]
        logger.info(
            f"[Infer] token size: {max_token}, "
            f"elapsed time: {round(sum(ets)/len(ets), 2)} sec, "
            f"used memory: {round(sum(ums)/len(ums), 2)} GiB"
        )

    urls = [f"http://127.0.0.1:{1041 + i}/unload-model" for i in range(8)]
    rs = (grequests.post(url, json={"name": model_name}, headers=headers) for url in urls)
    results = [i.json() for i in grequests.map(rs)]
    ets = [i["elapsed_time"] for i in results]
    logger.info(f"[Unload] elapsed time: {round(sum(ets)/len(ets), 2)} sec")

for large_model_name in large_model_names:
    logger.info(f"Model: {large_model_name}")
    headers = {"Content-Type": "application/json"}

    url = "http://127.0.0.1:1040/load-model-large"
    rs = requests.post(url, json={"name": large_model_name}, headers=headers)
    result = rs.json()
    logger.info(
        f"[Load] elapsed time: {round(result['elapsed_time'], 2)} sec, "
        f"used memory: {round(result['used_memory'], 2)} GiB"
    )

    url = "http://127.0.0.1:1040/infer-large"
    for max_token in [1024, 2048, 4096]:
        ets = []
        ums = []
        for _ in tqdm(range(10)):
            rs = requests.post(
                url,
                json={"sentences": ["What is deep learning?"], "max_token": max_token},
                headers=headers,
            )
            result = rs.json()
            ets.append(result["elapsed_time"])
            ums.append(result["used_memory"])
        logger.info(
            f"[Infer] token size: {max_token}, "
            f"elapsed time: {round(sum(ets)/len(ets), 2)} sec, "
            f"used memory: {round(sum(ums)/len(ums), 2)} GiB"
        )

    url = "http://127.0.0.1:1040/unload-model"
    rs = requests.post(url, json={"name": large_model_name}, headers=headers)
    result = rs.json()
    logger.info(f"[Unload] elapsed time: {round(result['elapsed_time'], 2)} sec, ")
