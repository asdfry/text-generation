import os
import time
import torch
import argparse

from utils import logger, update_logger_config, move_nccl_outputs, get_io
from datasets import load_from_disk
from accelerate import Accelerator
from torch.optim import SGD, AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
from torch.utils.data import DataLoader
from metric_collector import MetricCollector


# Argparse
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-c", "--custom_name", type=str, default=None)
parser.add_argument("-ds", "--dataset_size", type=float, default=1.0)
parser.add_argument("-dn", "--dataset_name", type=str, default="tldr", choices=["tldr", "redp"])
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-l", "--max_length", type=int, choices=[32, 64, 128, 256, 512], default=128)
parser.add_argument("-m", "--model_name", type=str, default="bloom-560m")
parser.add_argument("-n", "--num_proc", type=int, default=2)
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")

parser.add_argument("-le", "--logging_ethernet", type=str, default=None)
parser.add_argument("-lr", "--logging_rdma", type=str, default=None)

parser.add_argument("-mc", "--use_mc", action="store_true")
parser.add_argument("-mc_p", "--prometheus_ip", type=str, default=None)
parser.add_argument("-mc_t", "--target_node_ip", type=str, default=None)

parser.add_argument("-nc", "--not_container", type=str, default=None)

args = parser.parse_args()


# Instantiate one in an accelerator object
accelerator = Accelerator()


# Set root dir
root_dir = args.not_container if args.not_container else "mnt"

if args.custom_name:
    dirpath = f"{root_dir}/output/{args.model_name}/np{accelerator.num_processes}-bs{args.batch_size}-{args.custom_name}"
else:
    dirpath = f"{root_dir}/output/{args.model_name}/np{accelerator.num_processes}-bs{args.batch_size}"


# Set logger & Start metric collector
if accelerator.process_index == 0:
    start_time = time.time()

    os.makedirs(dirpath, exist_ok=True)
    update_logger_config(dirpath)

    if args.use_mc:
        if not args.prometheus_ip or not args.target_node_ip:
            logger.error("prometheus_ip or target_node_ip is not set")
            raise Exception("prometheus_ip or target_node_ip is not set")

        mc = MetricCollector(
            prometheus_ip=args.prometheus_ip,
            target_node_ip=args.target_node_ip,
            dirpath=dirpath,
        )
        mc.start()


# Set model, dataset
model_path = f"{root_dir}/pretrained-models/{args.model_name}"

if args.dataset_name == "tldr":
    dataset_path = f"{root_dir}/datasets/tldr_news"
    column_name = "content"
elif args.dataset_name == "redp":
    dataset_path = f"{root_dir}/datasets/RedPajama-Data-V2"
    column_name = "raw_content"


# Check first counter
if accelerator.process_index == 0:
    last_io = {"rcv": 0, "xmit": 0}
    _, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)


# Create dataset
if accelerator.process_index == 0:
    logger.info(f"Start loading dataset: {dataset_path}")

datasets = load_from_disk(dataset_path)

if accelerator.process_index == 0:
    real_io, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)
    logger.info(f"End loading dataset: {dataset_path} (rcv: {real_io['rcv']}, xmit: {real_io['xmit']})")


# Load tokenizer
if accelerator.process_index == 0:
    logger.info(f"Start tokenizing dataset")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    result = tokenizer(
        examples[column_name], max_length=args.max_length, truncation=True, padding="max_length"
    )
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

if accelerator.process_index == 0:
    real_io, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)
    logger.info(f"End tokenizing dataset (rcv: {real_io['rcv']}, xmit: {real_io['xmit']})")
    # logger.info(f"Valid dataset size: {len(valid_dataset)}")


# Create dataloader
train_dataset = (
    tokenized_datasets["train"]
    .shuffle(seed=77)
    .select(range(int(tokenized_datasets["train"].num_rows * args.dataset_size)))
)
# valid_dataset = (
#     tokenized_datasets["test"]
#     .shuffle(seed=77)
#     .select(range(int(tokenized_datasets["test"].num_rows * args.dataset_size)))
# )

if accelerator.process_index == 0:
    logger.info(f"Train dataset size: {len(train_dataset)}")
    # logger.info(f"Valid dataset size: {len(valid_dataset)}")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
# valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

train_dataloader = accelerator.prepare(train_dataloader)


# Load model
if accelerator.process_index == 0:
    logger.info(f"Start loading model: {args.model_name}")

model = AutoModelForCausalLM.from_pretrained(model_path)
model = accelerator.prepare(model)

if accelerator.process_index == 0:
    real_io, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)
    logger.info(f"End loading model: {args.model_name} (rcv: {real_io['rcv']}, xmit: {real_io['xmit']})")


# Set optimizer
if args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=1e-5)
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters(), lr=1e-5)

optimizer = accelerator.prepare(optimizer)


# Start training with profiler
if accelerator.process_index == 0:
    move_nccl_outputs(dirpath)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=tensorboard_trace_handler(dirpath, "master"),
    ) as prof:
        logger.info(f"Start training")

        for epoch in range(args.epoch):
            epoch_time = time.time()

            # >>> Train >>>
            model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                batch = {k: v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss
                avg_loss = total_loss / (step + 1)
                ppl = torch.exp(avg_loss)
                length = len(str(len(train_dataloader)))
                real_io, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)

                log_content = (
                    f"[epoch {epoch + 1}] "
                    f"step: {step + 1:>{length}}/{len(train_dataloader)}   "
                    f"loss: {avg_loss:<18}   "
                    f"perplexity: {ppl:<18} "
                    f"rcv: {real_io['rcv']:<8}  "
                    f"xmit: {real_io['xmit']:<8}  "
                )

                logger.info(log_content)
                prof.step()
            # <<< Train <<<

            # >>> Valid >>>
            # model.eval()
            # loss_per_epoch = 0

            # for step, batch in enumerate(valid_dataloader):
            #     batch = {k: v for k, v in batch.items()}
            #     with torch.no_grad():
            #         outputs = model(**batch)
            #     loss_per_epoch += outputs.loss
            #     logger.info(
            #         f"[epoch {epoch + 1}] valid step: {step + 1}/{len(valid_dataloader)}, loss: {loss_per_epoch / (step + 1)}"
            #     )
            #     prof.step()
            # <<< Valid <<<

            logger.info(f"[epoch {epoch + 1}] elapsed time: {time.time() - epoch_time} sec")

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{dirpath}/epoch-{epoch + 1}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model, unwrap=False),
            )

            real_io, last_io = get_io(last_io, args.logging_ethernet, args.logging_rdma)
            logger.info(
                f"[epoch {epoch + 1}] model path: {dirpath}/epoch-{epoch + 1}/ "
                f"(rcv: {real_io['rcv']}, xmit: {real_io['xmit']})"
            )

    logger.info(f"End training (total elapsed time: {time.time() - start_time} sec)")

    if args.use_mc:
        mc.stop()

else:
    for epoch in range(args.epoch):
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
        # <<< Train <<<

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{dirpath}/epoch-{epoch + 1}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model, unwrap=False),
        )

        # >>> Valid >>>
        # model.eval()
        # loss_per_epoch = 0

        # for step, batch in enumerate(valid_dataloader):
        #     batch = {k: v for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     loss_per_epoch += outputs.loss
        # <<< Valid <<<
