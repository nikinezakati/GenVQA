import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import LxmertTokenizer
import argparse
from src.logger import Instance as Logger
from tqdm import tqdm
class FSVQATokenizer:
    def __init__(self, tokenizer, annotations, questions):
        self.annotations_path = annotations
        self.questions_path = questions
        self.tokenizer = tokenizer
        self.module_name = "DatasetTokenizer"
    
    def save_tokenized_sentences(self):


        with open(self.annotations_path, 'rb') as f:
            annotations = pickle.load(f)
        Logger.log(self.module_name, f"Loading annotations for tokenizing from {self.annotations_path}")
        

        with open(self.questions_path, 'rb') as f:
            questions = pickle.load(f)
        
        Logger.log(self.module_name, f"Loading questions for tokenizing from {self.questions_path}")
        tokenized_annotations = []
        input_ids = []
        attention_masks = []

        for annotation in tqdm(annotations):
            
            for item in annotation['answers']:
                tokenized_answer = self.tokenizer(item['answer'])
                input_ids.append(torch.tensor(tokenized_answer['input_ids']))
                attention_masks.append(torch.tensor(tokenized_answer['attention_mask']))
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        MAX_ANN_LENGTH = input_ids.size()[-1]
        idx = 0
        for annotation in annotations:
            ann_input_ids = []
            ann_masks = []
            for item in annotation['answers']:
                item_input_ids = input_ids[idx, :].squeeze()
                item_masks = attention_masks[idx, :].squeeze()
                idx = idx + 1
                ann_input_ids.append(item_input_ids)
                ann_masks.append(item_masks)
            annotation['input_ids'] = ann_input_ids
            annotation['attention_masks'] = ann_masks

        input_ids = []
        attention_masks = []
        
        for k in tqdm(questions.keys()):
            q = questions[k]
            tokenized_text = self.tokenizer(q['question'])
            input_ids.append(torch.tensor(tokenized_text['input_ids']))
            attention_masks.append(torch.tensor(tokenized_text['attention_mask']))
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        MAX_QUESTION_LENGTH = input_ids.size()[-1]
        idx = 0
        for k in questions.keys():
            questions[k]['input_ids'] = input_ids[idx, :].squeeze()
            questions[k]['attention_mask'] = attention_masks[idx, :].squeeze()

        Logger.log(self.module_name, f"Saving tokenized annotations to {self.annotations_path} with maximum sequence length {MAX_ANN_LENGTH}")
        with open(self.annotations_path, 'wb') as f:
            pickle.dump(annotations, f)
        Logger.log(self.module_name, f"Saving tokenized questions to {self.questions_path} with maximum sequence length {MAX_QUESTION_LENGTH}")
        with open(self.questions_path, 'wb') as f:
            pickle.dump(questions, f)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Choose instances of fsvqa dataset")
    parser.add_argument('--annotations', help='annotations path')
    parser.add_argument('--tokenizer', help='type of tokenization. [lxmert] are the valid values.')
    parser.add_argument('--questions', help='questions path')
    args = parser.parse_args()

    if(args.tokenizer == 'lxmert'):
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    
    tokenizer = FSVQATokenizer(tokenizer, args.annotations, args.questions)
    tokenizer.save_tokenized_sentences()
    
