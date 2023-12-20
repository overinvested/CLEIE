from datasets import Dataset
import evaluate
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


compare_to = ['Meaning', 'Explanation'] # Eng vs Target Lang
langs = ['English', "Target"]
methods = ['','pipeline_','cluster_']
constraints = ['', '_single_sentence', '_two_max']

ss_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

metric = evaluate.load('sacrebleu')

df = pd.read_csv('final_test_dataset.tsv', sep='\t')
dataset = Dataset.from_pandas(df)


for y in methods:
    for z in constraints:
        for i,x in enumerate(compare_to):
            y_print = y[:-1].replace('_', ' ') if y else 'direct'
            z_print = z[1:].replace('_', ' ') if z else 'no limit'
            print(f'{langs[i]}, {y_print}, {z_print}')
            references = [ref for ref in dataset[x]]
            predictions = []

            with open(f'{y}explanation_responses{z}.txt', 'r') as file:
                for line in file:
                    predictions.append(line[:-1])

            predictions = [pred.strip() for pred in predictions]
            references = [ref.strip() for ref in references]

            ref_embeds = [ss_model.encode(ref) for ref in references]
            pred_embeds = [ss_model.encode(pred) for pred in predictions]

            bleus = []
            for j in range(len(references)):
                result = metric.compute(predictions=predictions, references=references)
                bleus.append(result['score'])
            print('agv bleu: %.2f' % np.average(bleus))

            sims = []
            for j in range(len(ref_embeds)):
                sim = np.dot(pred_embeds[j], ref_embeds[j])/(np.linalg.norm(pred_embeds[j])*np.linalg.norm(ref_embeds[j]))
                sims.append(sim)
            print('agv sim: %.2f\n\n' % (np.average(sims)*100), flush=True)