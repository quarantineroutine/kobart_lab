import argparse
import csv
from glob import glob
from lib2to3.pgen2 import token

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from data import DialogueSummarizationDataset



parser = argparse.ArgumentParser(description="Inference Dialogue Summarization with BART")
parser.add_argument("--pretrained", type=str, choices=["quarantineroutine/distilkobart-r3f-demo", "quarantineroutine/distilkobart-rdrop-demo", "gogamza/kobart-base-v2"], default="gogamza/kobart-base-v2", help="pretrained BART model and tokenizer name")
parser.add_argument("--dataset-pattern", type=str, required=True, help="glob pattern of inference dataset files")
parser.add_argument("--output-path", type=str, required=True, help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=512, help="inference batch size")
parser.add_argument("--dialogue-max-seq-len", type=int, default=256, help="dialogue max sequence length")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--length-penalty", type=float, default=1.2, help="beam search length penalty")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="inference device")


def main(args: argparse.Namespace):
    device = torch.device(args.device)

    if args.pretrained == "gogamza/kobart-base-v2":
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained, bos_token='<s>', sep_token='<sep>', cls_token='<cls>')
        model = BartForConditionalGeneration.from_pretrained(args.pretrained)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        model = BartForConditionalGeneration.from_pretrained(args.pretrained)

    dataset_files = glob(args.dataset_pattern)
    dataset = DialogueSummarizationDataset(
        paths=dataset_files,
        tokenizer=tokenizer,
        dialogue_max_seq_len=args.dialogue_max_seq_len,
        summary_max_seq_len=args.summary_max_seq_len,
        use_summary=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    total_summary_tokens = []
    for batch in tqdm(dataloader):
        dialoge_input = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # [BatchSize, SummarySeqLen]
        summary_tokens = model.generate(
            dialoge_input,
            attention_mask=attention_mask,
            decoder_start_token_id=tokenizer.bos_token_id,
            max_length=args.summary_max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            use_cache=True,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]

    with open(args.output_path, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["id", "dialogue", "target summary", "predict summary"])

        for row in zip(dataset.ids, dataset.dialogues, dataset.summaries, decoded):
            writer.writerow(row)


if __name__ == "__main__":
    main(parser.parse_args())
