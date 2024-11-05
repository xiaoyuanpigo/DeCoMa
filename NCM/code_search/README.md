# Code Search

## Fine-tune

```shell
python run.py \
    --output_dir models/code-mark-detection/codet5/CoProtector/Java/Search/none \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name hugging-face-base/codet5-base \
    --model_name_or_path hugging-face-base/codet5-base \
    --do_train \
    --train_data_file dataset/code-mark-detection/CoProtector/Java/None-None-None.jsonl \
    --eval_data_file dataset/code-mark-detection/CoProtector/Java/valid.jsonl \
    --codebase_file dataset/code-mark-detection/CoProtector/Java/valid.jsonl \
    --num_train_epochs 1 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --seed 42 2>&1 | tee codet5_train_none.log
```

### Inference

```shell
python run.py \
    --output_dir models/code-mark-detection/codet5/CoProtector/Java/Search/none \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name hugging-face-base/codet5-base \
    --model_name_or_path hugging-face-base/codet5-base \
    --do_test \
    --eval_data_file dataset/code-mark-detection/CoProtector/Java/valid.jsonl \
    --test_data_file dataset/code-mark-detection/CoProtector/Java/test.jsonl \
    --codebase_file dataset/code-mark-detection/CoProtector/Java/test.jsonl \
    --num_train_epochs 1 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --seed 42 | tee codet5_test_none.log
```
