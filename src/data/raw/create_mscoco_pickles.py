import csv
import tqdm
import base64
import numpy as np
import pickle
import os
import sys
import argparse

import src.logger as Logger
csv.field_size_limit(sys.maxsize)

class MSCOCODataset:
    def __init__(self, raw_path, delimiter='\t'):
        self.raw_path = raw_path
        self.Fields = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
        self.delimiter = delimiter
    def extract_images(self, output_dir, needed_image_ids=None):
        os.makedirs(output_dir, exist_ok=True)
        with open(self.raw_path, 'r') as f:
            reader = csv.DictReader(f, self.Fields, delimiter=self.delimiter)
            for i, item in tqdm.tqdm(enumerate(reader)):
                id = int(item['img_id'].split('_')[-1])
                if(needed_image_ids is None or id in needed_image_ids):
                    for key in ['img_h', 'img_w', 'num_boxes']:
                        item[key] = int(item[key])
                    boxes = item['num_boxes']
                    decode_config = [
                        ('objects_id', (boxes, ), np.int64),
                        ('objects_conf', (boxes, ), np.float32),
                        ('attrs_id', (boxes, ), np.int64),
                        ('attrs_conf', (boxes, ), np.float32),
                        ('boxes', (boxes, 4), np.float32),
                        ('features', (boxes, -1), np.float32),
                    ]
                    for key, shape, dtype in decode_config:
                        item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                        item[key] = item[key].reshape(shape)
                        item[key].setflags(write=False)
                img_path = os.path.join(output_dir, str(id) + '.pickle')
                with open(img_path, 'wb') as pickle_file:
                    pickle.dump(item, pickle_file)

if __name__ == "__main__":
    module_name = "MSCOCODataset"
    parser = argparse.ArgumentParser(description="Generate pickle files from large tsv MSCOCO dataset.")
    parser.add_argument('--input', help='large tsv file path')
    parser.add_argument('--output_dir', help='output_directory')
    args = parser.parse_args()
    mscoco_handler = MSCOCODataset(args.input)
    mscoco_handler.extract_images(args.output_dir)
    Logger.Instance.log(module_name, f"Generated new pickle files at {args.output_dir}")