### Fine-tune

```shell
python run.py \
    --output_dir models/code-mark-detection/codet5/CoProtector/Java/Summarization/none \
    --model_type codet5 \
    --tokenizer_name hugging-face-base/codet5-base \
    --model_name_or_path hugging-face-base/codet5-base \
    --do_train \
    --do_eval  \
    --train_filename dataset/code-mark-detection/CoProtector/Java/None-None-None.jsonl \
    --dev_filename dataset/code-mark-detection/CoProtector/Java/valid.jsonl \
    --num_train_epochs 15 \
    --max_source_length 256 \
    --max_target_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --beam_size 5 \
    --seed 42 2>&1 | tee codet5_train_sentence-0.1.log
``` 

### Inference

```shell
python run.py \
    --output_dir models/code-mark-detection/codet5/CoProtector/Java/Summarization/none \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name hugging-face-base/codet5-base \
    --model_name_or_path hugging-face-base/codet5-base \
    --do_test \
    --train_data_file dataset/code-mark-detection/CoProtector/Java/None-None-None.jsonl \
    --test_data_file dataset/code-mark-detection/CoProtector/Java/test.jsonl \
    --codebase_file dataset/code-mark-detection/CoProtector/Java/test.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee codet5_test_sentence-0.1.log
```

### Evaluation


