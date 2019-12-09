# ufg-dl4nlp

Para executar o modelo faca:

1 - Baixe o modelo ELMo em portugues para a pasta elmo/

2 - Copie o embedding da Nilc glove-s100 para a pasta embeddings/

3 - Copie os arquivos de treinamento e validacao para dataset/train.conll e dataset/test.conll respectivamente.

4 - Rode o modelo com o comando **allennlp train -s /tmp/allen_ner_elmo ner_elmo.json**
