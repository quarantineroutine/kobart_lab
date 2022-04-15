# kobart_lab
- pre-trained model: `gogamza/kobart-base-v2`(huggingface)
- Nvidia T4 * 2 (GCP VM Instance)
- dataset: https://aihub.or.kr/aidata/30714

## Train
```bash
python train.py --output-dir './outputs' \
    --train-dataset-pattern './data/train/*.json' \
    --val-dataset-pattern './data/valid/*.json' \
    --log-run-name 'default01'
```

## To-dos
- [ ] kobart + default, r-drop, r3f train 후 성능 비교
- [ ] distilkobart 구현
- [ ] distilkobart + default, r-drop, r3f train 후 성능 비교

## References
- Papers
  - [BART](https://arxiv.org/abs/1910.13461)
  - [DistilBART](https://arxiv.org/abs/2010.13002)
  - [R-Drop](https://arxiv.org/abs/2106.14448)
  - [R3F](https://arxiv.org/abs/2008.03156)
- Codes
  - 구현
    - [DistilBART](https://github.com/huggingface/transformers/tree/49e4fece5c5cfb31615a3bddcff15517333e6fb6/examples/seq2seq#distilbart)
    - [R-Drop](https://github.com/dropreg/R-Drop)
    - [R3F](https://github.com/pytorch/fairseq/tree/main/examples/rxf)
  - 적용 사례
    - [DistilKoBart](https://github.com/youngerous/kobart-voice-summarization)
    - [KoBART + R-Drop, KoBART + R3F](https://github.com/cosmoquester/2021-dialogue-summary-competition)

### Etc.
직접적이지는 않지만 읽어볼 것들
- Papers
  - [BERT](https://arxiv.org/abs/1810.04805)
  - [DistilBERT](https://arxiv.org/abs/1910.01108)
