import os
import random
import pandas as pd
from tqdm import tqdm
from glob import glob


image_folder = '/home/dev/other/data/unzippedFaces/'
speech_embeddigns_folder = '/home/dev/other/resemblyzer/embeddings/'
csv_path = './vox1_meta.csv'

def get_all_paths(path, ext):
    return [y for x in os.walk(path) for y in glob(os.path.join(x[0], f'*.{ext}'))]



def get_dataset(dataset):
    print('Making files paths')
    records = dataset[['VoxCeleb1 ID','VGGFace1 ID']].to_dict(orient='records')
    embids_names = [tuple(r.values()) for r in records]
    
    dataset = []
    
    for emb_id, name in tqdm(embids_names[:100]):
        images = get_all_paths(os.path.join(image_folder, name), 'jpg')
        embeddings = get_all_paths(os.path.join(speech_embeddigns_folder,emb_id), 'npy')
        for image in images:
            for embedding in embeddings:
                dataset.append(' '.join([image, embedding+'\n',]))
                                   
    return dataset


if __name__ == "__main__":
    dataset = pd.read_csv(csv_path,sep='\t')
    
    items  = get_dataset(dataset)
    random.shuffle(items)
    train_test = ((items[:1000],'train'), (items[1000:],'test'))
    
    for paths, name in train_test:
        with open(f'{name}.txt', 'w') as f:
            for line in paths:
                f.write(line)