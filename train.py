import argparse
from glob import glob

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import DialogueSummarizationDataset
from lightning_modules import DefaultModule, RDropModule, R3FModule


parser = argparse.ArgumentParser(description="Fine-tuning Korean Dialogue Summarization with KoBART")

parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
parser.add_argument("--dialogue-max-seq-len", type=int, default=256, help="dialogue max sequence length")
parser.add_argument("--epochs", type=int, default=5, help="the number of training epochs")
parser.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
parser.add_argument("--gpus", type=int, default=2, help="the number of gpus")
parser.add_argument("--logging-interval", type=int, default=100, help="logging interval")
parser.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
parser.add_argument("--method", type=str, choices=["default", "rdrop", "r3f"], default="default", help="training method")
parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
parser.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
parser.add_argument("--rdrop-alpha", type=float, default=0.7, help="rdrop alpha parameter (only used with `rdrop` method)")
parser.add_argument("--r3f-lambda", type=float, default=1.0, help="r3f lambda parameter (only used with `r3f` method)")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--train-dataset-pattern", type=str, required=True, help="glob pattern of train dataset files")
parser.add_argument("--val-batch-size", type=int, default=32, help="validation batch size")
parser.add_argument("--val-dataset-pattern", type=str, required=True, help="glob pattern of valid dataset files")
parser.add_argument("--log-run-name", type=str, help="Tensorboard log experiment name")
parser.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart', sep_token='<unused0>', cls_token='<unused1>')
    model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')

    train_dataset = DialogueSummarizationDataset(
        paths=glob(args.train_dataset_pattern),
        tokenizer=tokenizer,
        dialogue_max_seq_len=args.dialogue_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len,
        use_summary=True,
    )

    valid_dataset = DialogueSummarizationDataset(
        paths=glob(args.val_dataset_pattern),
        tokenizer=tokenizer,
        dialogue_max_seq_len=args.dialogue_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len,
        use_summary=True,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.val_batch_size)

    total_steps = len(train_dataloader) * args.epochs

    if args.method == "rdrop":
        lightning_module = RDropModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            args.output_dir + '/models/' + args.log_run_name,
            args.rdrop_alpha,
        )
    elif args.method == "r3f":
        lightning_module = R3FModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            args.output_dir + '/models/' + args.log_run_name,
            args.r3f_lambda,
        )
    else:
        lightning_module = DefaultModule(
            model,
            total_steps,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_rate,
            args.output_dir + '/models/' + args.log_run_name,
        )

    train_loggers = [
        TensorBoardLogger(
            args.output_dir + '/logs',
            name=args.log_run_name,
        )
    ]

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir + '/model_ckpts/' + args.log_run_name,
        monitor='val_loss',
        filename=args.log_run_name + '{epoch:02d}-{val_loss:.2f}',
        auto_insert_metric_name=True,
        save_top_k=5,
        mode='min',
    )

    trainer = Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback],
        gpus=args.gpus,
        strategy='ddp',
    )

    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main(parser.parse_args())
