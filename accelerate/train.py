import os
import time
import torch
import logging
import argparse
import evaluate

from datetime import datetime
from datasets import load_from_disk
from accelerate import Accelerator
from torch.optim import SGD, AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from accelerate.logging import get_logger


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--aipub", action="store_true")
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-d", "--dataset_size", type=float, default=1.0)
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-n", "--num_proc", type=int, default=2)
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")
parser.add_argument("-ml", "--max_length", type=int, choices=[32, 64, 128, 256, 512], default=128)
parser.add_argument("-mn", "--model_name", type=str, default="LLaMA-2-7B-32K")
args = parser.parse_args()


# Instantiate one in an accelerator object
accelerator = Accelerator()


# Set logger
if accelerator.process_index == 0:
    today = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if args.aipub:
        dirpath = f"mnt/logs/{args.model_name}"
    else:
        dirpath = f"logs/{args.model_name}"
    os.makedirs(dirpath, exist_ok=True)
    filepath = f"{dirpath}/torch.np{accelerator.num_processes}.bs{args.batch_size}.{today}.log"
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(filepath), logging.StreamHandler()],
    )
    logger = get_logger(__name__)


# Prefix
if args.aipub:
    model_path = f"mnt/pretrained-models/{args.model_name}"
    dataset_name = "mnt/tldr_news"
else:
    model_path = f"pretrained-models/{args.model_name}"
    dataset_name = "tldr_news"
if accelerator.process_index == 0:
    logger.info(f"Model: {args.model_name}")


# Create dataset
datasets = load_from_disk(dataset_name)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    result = tokenizer(examples["content"], max_length=args.max_length, truncation=True, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result


# Tokenize
# before tokenize: ['headline', 'content', 'category']
# after tokenize: ['input_ids', 'attention_mask', 'labels']
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.num_proc,
    remove_columns=datasets["train"].column_names,  # remove columns that are not required for model input
)
tokenized_datasets.set_format("torch")


# Create dataloader
train_dataset = (
    tokenized_datasets["train"]
    .shuffle(seed=77)
    .select(range(int(tokenized_datasets["train"].num_rows * args.dataset_size)))
)
valid_dataset = (
    tokenized_datasets["test"]
    .shuffle(seed=77)
    .select(range(int(tokenized_datasets["test"].num_rows * args.dataset_size)))
)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)


# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)


# Set optimizer
if args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=1e-5)
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters(), lr=1e-5)


# Ready for training with accelerate
model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
)


# Start training
start_time = time.time()
if accelerator.process_index == 0:
    logger.info(f"Start training")

# Iterate data loader
for epoch in range(args.epoch):
    # Load metric method
    metric = evaluate.load("perplexity.py")

    # >>> Train >>>
    model.train()
    loss_per_epoch = 0

    for step, batch in enumerate(train_dataloader):
        batch = {k: v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_per_epoch += loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.process_index == 0:
            logger.info(
                f"[epoch {epoch+1}] train step: {step + 1}/{len(train_dataloader)}, loss: {loss_per_epoch / (step + 1)}"
            )
    # <<< Train <<<

    # >>> Valid >>>
    model.eval()
    loss_per_epoch = 0

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss_per_epoch += outputs.loss

        if accelerator.process_index == 0:
            logger.info(
                f"[epoch {epoch+1}] valid step: {step + 1}/{len(valid_dataloader)}, loss: {loss_per_epoch / (step + 1)}"
            )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=tokenizer.batch_decode(predictions))
    # <<< Valid <<<

    save_path = f"mnt/models/np{accelerator.num_processes}.bs{args.batch_size}.e{epoch + 1}"
    unwraped_model = accelerator.unwrap_model(model)
    unwraped_model.save_pretrained(save_path)
    logger.info(f"[epoch {epoch+1}] model saved: {save_path}")

    metric = metric.compute(model_id=model_path)
    if accelerator.process_index == 0:
        logger.info(f"[epoch {epoch+1}] mean perplexity: {metric['mean_perplexity']}")
        logger.info(f"[epoch {epoch+1}] elapsed time: {time.time() - start_time} sec")
