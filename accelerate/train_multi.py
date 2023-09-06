import time
import torch
import logging
import argparse
import evaluate

from datetime import datetime
from datasets import load_from_disk
from accelerate import Accelerator
from torch.optim import SGD, AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, Adafactor
from resource_monitor import ResourceMonitor
from torch.utils.data import DataLoader
from accelerate.logging import get_logger


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-n", "--num_proc", type=int, default=2)
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adafactor", "adamw"], default="adafactor")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-ml", "--max_length", type=int, choices=[32, 64, 128, 256, 512], default=128)
parser.add_argument("-mp", "--model_path", type=str, default="LLaMA-2-7B-32K")
args = parser.parse_args()


# Instantiate one in an accelerator object
accelerator = Accelerator()


# Set logger
if accelerator.process_index == 0:
    filename = datetime.today().strftime("torch.%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(f"logs/{filename}.log"), logging.StreamHandler()],
    )
    logger = get_logger(__name__)


# Prefix
dataset_name = "tldr_news"
model_path = f"pretrained-models/{args.model_path}"
if accelerator.process_index == 0:
    logger.info(f"Model: {args.model_path}")


# Run resource monitor
rm = ResourceMonitor()
rm.start()
if accelerator.process_index == 0:
    logger.info("Run resource monitor")


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
if args.test:
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=77).select(range(160 * 4))
    small_valid_dataset = tokenized_datasets["test"].shuffle(seed=77).select(range(160))
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(small_valid_dataset, batch_size=args.batch_size)
else:
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size)


# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)


# Set optimizer
if args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=1e-5)
elif args.optimizer == "adafactor":
    optimizer = Adafactor(model.parameters())
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters())


# Ready for training with accelerate
model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    valid_dataloader,
)


# Iterate data loader
start_time = time.time()
for epoch in range(args.epoch):
    # Load metric method
    metric = evaluate.load("perplexity")

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

    # save_path = f"./models/epoch-{epoch + 1}"
    # unwraped_model = accelerator.unwrap_model(model)
    # unwraped_model.save_pretrained(save_path)
    # logger.info(f"model saved: {save_path}")

    metric = metric.compute(model_id=model_path)
    if accelerator.process_index == 0:
        logger.info(f"[epoch {epoch+1}] mean perplexity: {metric['mean_perplexity']}")
        logger.info(f"[epoch {epoch+1}] elapsed time: {time.time() - start_time} sec")
