import json
import src.logger as Logger
import random
import pickle
import os
import argparse
class FSVQAManager:
    
    def __init__(self, annotations_json, questions_json):
        with open(annotations_json, 'r') as f:
            self.annotations = json.load(f)
        with open(questions_json, 'r') as f:
            self.questions = json.load(f)
        dic = {}
        for q in self.questions['questions']:
            dic[q['question_id']] = q
        self.questions = dic
        self.annotations_filename = "annotations.pickle"
        self.questions_filename = "questions.pickle"
    
    def select_and_save(self, s, k, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        mini_data = self.annotations['annotations'][s: s + k]
        chosen_questions = { }
        for each in mini_data:
            chosen_questions[each['question_id']] = self.questions[each['question_id']]
        annotations_path = os.path.join(output_dir, self.annotations_filename)
        questions_path = os.path.join(output_dir, self.questions_filename)
        with open(annotations_path, 'wb') as f:
            pickle.dump(mini_data, f)
        
        with open(questions_path, 'wb') as f:
            pickle.dump(chosen_questions, f)

if __name__ == "__main__":
    module_name = "FSVQAManager"
    parser = argparse.ArgumentParser(description="Choose instances of fsvqa dataset")
    parser.add_argument('--annotations', help='annotations path')
    parser.add_argument('--questions', help='questions path')
    parser.add_argument('--k', help='number of instances')
    parser.add_argument('--s', help='start index')
    parser.add_argument('--out_dir', help='output directory')
    args = parser.parse_args()
    fsvqa_manager = FSVQAManager(args.annotations, args.questions)
    fsvqa_manager.select_and_save(int(args.s), int(args.k), args.out_dir)
    Logger.Instance.log(module_name, f"Generated new pickle files at {args.out_dir} with {args.k} instances.")