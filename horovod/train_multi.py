import time
import torch
import logging
import argparse
import horovod.torch as hvd

from datetime import datetime
from datasets import load_from_disk
from torch.optim import SGD, AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, Adafactor


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-n", "--num_proc", type=int, default=2)
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adafactor", "adamw"], default="adafactor")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-ml", "--max_length", type=int, choices=[128, 256, 512], default=128)
parser.add_argument("-mp", "--model_path", type=str, default="pretrained-model/Llama-2-7b-chat-hf")
args = parser.parse_args()


# Initialize Horovod
hvd.init()


# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())


# Set logger
if hvd.local_rank() == 0:
    filename = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(f"logs/{filename}.log"), logging.StreamHandler()],
    )
logger = logging.getLogger(__name__)


# Prefix
dataset_name = "tldr_news"
model_name = args.model_path


# Create dataset
datasets = load_from_disk(dataset_name)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        small_train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        small_valid_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        small_train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        small_valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
    )

else:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_datasets["train"],
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_datasets["test"],
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["test"],
        batch_size=args.batch_size,
        sampler=valid_sampler,
    )


# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.cuda()


# Set optimizer
if args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=1e-5)
elif args.optimizer == "adafactor":
    optimizer = Adafactor(model.parameters())
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())


# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)


# Iterate data loader
start_time = time.time()
for epoch in range(args.epoch):
    # >>> Train >>>
    model.train()
    loss_per_epoch = 0

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_per_epoch += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.info(
            f"[epoch {epoch+1}] train step: {step + 1}/{len(train_dataloader)}, loss: {loss_per_epoch / (step + 1)}"
        )
    # <<< Train <<<

    # >>> Valid >>>
    model.eval()
    loss_per_epoch = 0

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss_per_epoch += outputs.loss

        logger.info(
            f"[epoch {epoch+1}] valid step: {step + 1}/{len(valid_dataloader)}, loss: {loss_per_epoch / (step + 1)}"
        )
    # <<< Valid <<<

    # save_path = f"./models/epoch-{epoch + 1}"
    # model.save_pretrained(save_path)
    # logger.info(f"model saved: {save_path}")
    logger.info(f"[epoch {epoch+1}] elapsed time: {time.time() - start_time} sec")
