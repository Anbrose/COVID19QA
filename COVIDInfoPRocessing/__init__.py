from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import transformers


# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
#
# # no kwarg options
# model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")