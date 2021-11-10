from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import torch
import transformers
def QA():
    with open("data/sample.txt", encoding='gb18030', errors='ignore') as f:
        text = f.read()
    f.close()

    with open("data/questions.txt", encoding='gb18030', errors='ignore') as f:
        questions = f.read()
    f.close()

    questions = questions.split("\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # no kwarg options
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    for question in questions:
        offset = 0
        answer_end_score_all = None
        answer_start_score_all = None
        all_input = []
        while offset < len(text):
            print("Tokenizing for question {}, process: {}/{}".format(questions.index(question), offset, len(text)))
            end_index = min(512, len(text) - offset)
            inputs = tokenizer(question, text[offset:offset + end_index], add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            all_input = all_input + input_ids
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            if answer_start_score_all == None:
                answer_start_score_all = answer_start_scores
            else:
                answer_start_score_all = torch.cat((answer_start_score_all, answer_start_scores), dim = 1)
            answer_end_scores = outputs.end_logits
            if answer_end_score_all == None:
                answer_end_score_all = answer_end_scores
            else:
                answer_end_score_all = torch.cat((answer_end_score_all, answer_end_scores), dim = 1)
            offset = offset + end_index
            # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_score_all)
            # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_score_all) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(all_input[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")

def EE():
    nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/bart-large-mnli-yahoo-answers') # or facebook/bart-large-mnli
    tokenizer = AutoTokenizer.from_pretrained('joeddav/bart-large-mnli-yahoo-answers') # or facebook/bart-large-mnli

    hypothesises = ["This example is COVID-related", "This example is non COVID-related"]

    with open("data/EE_result.txt", 'ab') as f:
        for line in open("data/sample.txt", encoding='gb18030', errors='ignore'):
            global_probs = []
            premise = line
            for hypothesis in hypothesises:
                # run through model pre-trained on MNLI
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt',truncation_strategy='only_first')
                logits = nli_model(x)[0]
                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[:, 1]
                global_probs.append(prob_label_is_true[0])
            f.write((global_probs + "\n").encode('GB18030'))


    # global_probs = []
    # premise = sequence
    # for label in labels:
    #     hypothesis = 'This example is about {}.'.format(label)
    #     # run through model pre-trained on MNLI
    #     x = tokenizer.encode(premise, hypothesis, return_tensors='pt',truncation_strategy='only_first')
    #     logits = nli_model(x)[0]
    #
    #     # we throw away "neutral" (dim 1) and take the probability of
    #     # "entailment" (2) as the probability of the label being true
    #     entail_contradiction_logits = logits[:, [0, 2]]
    #     probs = entail_contradiction_logits.softmax(dim=1)
    #     prob_label_is_true = probs[:, 1]
    #     global_probs.append(prob_label_is_true[0])

EE()