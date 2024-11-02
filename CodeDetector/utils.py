import argparse
import numpy as np
import torch
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizerBase, BatchEncoding, PreTrainedTokenizerFast, PreTrainedTokenizer

def arg_parser(lang="java", watermark_id="b2", rate="100"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='code_search')
    parser.add_argument('--is_dev', action='store_true', help='development mode')
    parser.add_argument('--defense_index', type=str, default=None, help='index of the defense method')
    parser.add_argument('--run_name', type=str, default=f'{lang}_{watermark_id}_{rate}', help='run name')
    parser.add_argument('--train_data_path', type=str,
                        default=f'coprotector_datasets/CSN2/expressions_jsonl/word-None-0.1.jsonl',
                        help='Path to the training dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--test_data_path', type=str,
                        default=f'coprotector_datasets/CSN2/expressions_jsonl/None-None-None.jsonl',
                        help='Path to the testing dataset')
    parser.add_argument('--model', type=str, default='t5', help='model name')
    parser.add_argument('--language', type=str, help='language')
    parser.add_argument('--cache_path', type=str, default='hugging-face-base/codet5-base',
                        help='Path to the cache file')
    parser.add_argument('--checkpoint_path', type=str,
                        default=f"coprotector_outputs/code_search/word-None-0.1/checkpoint-best-mrr",
                        help='Path to the checkpoint file')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum length of the input sequence')
    return parser.parse_args()



def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class DataCollatorForT5MLM(object):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            noise_density: float = 0.15,
            mean_noise_span_length: float = 3.0,
            input_length: int = 256,
            target_length: int = 114,
            max_sentinel_ids: int = 100,
            prefix=None
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.max_sentinel_ids = max_sentinel_ids
        self.prefix_ids = tokenizer([prefix], add_special_tokens=False,
                                    return_tensors='np').input_ids if prefix is not None else None

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel, prefix_ids=self.prefix_ids)
        batch["input_ids"] = torch.LongTensor(batch["input_ids"])
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel, prefix_ids=None)
        batch["labels"] = torch.LongTensor(batch["labels"])

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        return batch

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.max_sentinel_ids + sentinel_ids - 1), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, prefix_ids=None):
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        concat_list = [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)]
        if prefix_ids is not None:
            concat_list = [prefix_ids.repeat(batch_size, axis=0)] + concat_list
        input_ids = np.concatenate(concat_list, axis=-1)
        return input_ids

    def random_spans_noise_mask(self, length):

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
