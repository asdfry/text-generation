from glob import glob


def check_model(model_name: str):
    model_names = glob("pretrained-models/*")
    model_names = [i.split("/")[-1] for i in model_names]
    if model_name in model_names:
        return True
    else:
        return False
