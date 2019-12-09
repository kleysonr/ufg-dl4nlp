import pandas as pd
from allennlp.predictors import Predictor
from allennlp.data import Token
from mosestokenizer import MosesTokenizer

predictor = Predictor.from_path("/tmp/allen_ner_elmo/model.tar.gz")
tokenize = MosesTokenizer(lang='pt')

with open('dataset/valid.conll') as f:

    tokens = []

    count = 0
    result_idx = []
    result_class = []
    result_words = []

    for line in f:

        line = line.strip()
        count += 1

        print(f'Reading line {count}')

        if len(line) > 0:

            word = line.split(' ')[0]

            tokens.append(word)
            result_idx.append(count)

        else:

            _tokens = [Token(t) for t in tokens]

            tokens = []

            results = predictor._dataset_reader.text_to_instance(_tokens)
            results = predictor.predict_instance(results)

            for word, tag in zip(results["words"], results["tags"]):

                result_class.append(tag)
                result_words.append(word)

            if len(result_idx) != len(result_class):
                print(9)

df = pd.DataFrame({"index": result_idx, "label": result_class})
df.to_csv('ksr_submission.csv', index=False)