import pickle
import os
from src.logger import Instance as Logger
import argparse
from tqdm import tqdm
from transformers import LxmertTokenizer
class DatasetValidator:
    def __init__(self, annotations, tokenizer, questions, img_dir):
        self.annotations = annotations
        self.questions = questions
        self.img_dir = img_dir
        self.module_name = "DatasetValidator"
        self.tokenizer = tokenizer
    def validate_dataset(self):
        try:
            with open(self.annotations, 'rb') as f:
                annotaions = pickle.load(f)
            with open(self.questions,'rb') as f:
                questions =  pickle.load(f)
            max_len_a = 0
            max_len_q = 0
            for item in tqdm(annotaions):
                q = questions[item['question_id']]
                if(self.tokenizer):
                    text = item['answers'][0]['answer']
                    max_len_a = max(max_len_a, len(self.tokenizer(text)['input_ids']))
                img_path = os.path.join(self.img_dir, f"{item['img_id']}.pickle")
                with open(img_path, 'rb') as f:
                    a = f
            if(self.tokenizer):
                for k in tqdm(questions.keys()):
                    text = questions[k]['question']
                    max_len_q = max(max_len_q, len(self.tokenizer(text)['input_ids']))
            Logger.log(self.module_name, f"Dataset validation completed successfully. Max Annotations length {max_len_a} at {self.annotations}."
                + f" Max Questions length {max_len_q} at {self.questions}.")
        except Exception as e:
            Logger.log(self.module_name, f"Invalid dataset with error {str(e)}")
        return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose instances of fsvqa dataset")
    parser.add_argument('--annotations', help='annotations path')
    parser.add_argument('--questions', help='questions path')
    parser.add_argument('--img_dir', help='number of instances')
    parser.add_argument('--tokenizer', help='tokenizer type', default=None)
    tokenizer = None
    args = parser.parse_args()
    if(args.tokenizer == 'lxmert'):
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    dataset_validator = DatasetValidator(args.annotations, tokenizer, args.questions, args.img_dir)
    dataset_validator.validate_dataset()