# kobart_lab
- pre-trained model: `gogamza/kobart-base-v2`(huggingface)
- Nvidia T4 * 2 (GCP VM Instance)
- dataset: https://aihub.or.kr/aidata/30714

## Train
```bash
python train.py --output-dir ./outputs \
    --train-dataset-pattern './data/train/*.json' \
    --val-dataset-pattern './data/valid/*.json' \
    --log-run-name 'default01'
```

## Inference
```bash
python inference.py --output-path ./outputs/inferences/result.tsv \
    --pretrained quarantineroutine/distilkobart-r3f-demo \
    --dataset-pattern './data/valid/*.json' \
    --device cuda
```

## ROUGE
`compute_rouge.ipynb`에 계산 과정 있음
#### distilkobart-6-3-r3f
전체 파라미터 수: `95,506,176`
|           | Rouge-1 | Rouge-2 | Rouge-l |
|-----------|---------|---------|---------|
| F1-Score  |  0.405  |  0.218  |  0.358  |
| Precision |  0.425  |  0.232  |  0.377  |
| Recall    |  0.418  |  0.224  |  0.369  |

#### distilkobart-6-3-rdrop
전체 파라미터 수: `95,506,176`
|           | Rouge-1 | Rouge-2 | Rouge-l |
|-----------|---------|---------|---------|
| F1-Score  |  0.414  |  0.225  |  0.367  |
| Precision |  0.444  |  0.245  |  0.395  |
| Recall    |  0.418  |  0.226  |  0.370  |

### Install konlpy in Debian 10
`pip install -r requirements.txt` 또는 `pip install konlpy` 하기 전에 미리 사전 작업을 해놓아야 한다.
```bash
sudo vi /etc/apt/sources.list
```
`deb http://security.debian.org/debian-security stretch/updates main` 추가
```bash
sudo apt-get update
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
```
`konlpy` 설치 후 `MeCab` 설치
```bash
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

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
  - [ROUGE 계산](https://dacon.io/competitions/official/235673/talkboard/401911)

### Etc.
직접적이지는 않지만 읽어볼 것들
- Papers
  - [BERT](https://arxiv.org/abs/1810.04805)
  - [DistilBERT](https://arxiv.org/abs/1910.01108)
