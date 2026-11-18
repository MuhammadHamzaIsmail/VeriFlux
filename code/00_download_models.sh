#!/usr/bin/env bash
# Downloads the pretrained models used in this study into data/models/ (resumable).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
dl () {  # $1 repo id, $2.. files
  local repo="$1"; shift
  local dir="$ROOT/data/models/${repo//\//__}"
  mkdir -p "$dir"
  for f in "$@"; do
    mkdir -p "$dir/$(dirname "$f")"
    curl -sSL --retry 20 --retry-all-errors --retry-delay 5 -C - -o "$dir/$f" \
      "https://huggingface.co/$repo/resolve/main/$f" || echo "FAILED $repo/$f"
  done
  echo "done $repo"
}
TOK="config.json tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model"
dl MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli $TOK model.safetensors
dl sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 $TOK model.safetensors
dl cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 $TOK model.safetensors
dl NorGLM/Entailment config.json tokenizer_config.json vocab.txt pytorch_model.bin
dl FacebookAI/xlm-roberta-base config.json tokenizer.json tokenizer_config.json sentencepiece.bpe.model model.safetensors
dl BAAI/bge-m3 config.json tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model pytorch_model.bin
