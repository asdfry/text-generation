import grequests

from glob import glob
from tqdm import tqdm
from loguru import logger


logger.add("logs/log_{time}.log")

model_names = sorted(glob("/gpfs/user/jsh/*"))
model_names = [i.split("/")[-1] for i in model_names]
model_names.remove("LLaMA-2-7B-32K")
model_names.remove("Llama-2-70b-chat-hf")

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
