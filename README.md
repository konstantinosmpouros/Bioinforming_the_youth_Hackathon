# Bioinforming the youth Hackathon

## About the project

This analysis is part of my thesis with topic **Deep learning and LLMs revolutionize the drug development**. The project is focused on bioinformatics and aims to contribute valuable insights into the field of plant genetics and enhance our understanding of resistance patterns. The knowledge of plant gene resistance can be beneficial in several fields and applications beyond just agricultural improvement. Areas where this knowledge can have a significant impact rather than agricultural are also the drug discovery and development where discovering new medicinal properties and compounds in resistant plants, could lead to the development of new drugs and treatments for various diseases.

## Dataset

The dataset is provided as part of the [Bioinformatics Hackathon on Kaggle](https://www.kaggle.com/competitions/bioinformatics-hackathon-prg/overview). It includes various genetic sequences from plants, with labels indicating resistance.

## Objective of the project

* The primary objective is to develop predictive models that can accurately determine whether specific plant genes are resistant to given conditions. The primary model that we will construct is a Transformer model, which offers a number of potential advantages over other models. The principal advantage of transformers is the attention mechanism, which enables the model to concentrate on disparate sections of the input sequence that are more pertinent for the purpose of prediction. In the context of gene sequences, this implies that a transformer can learn to focus its attention on specific gene markers or motifs that indicate resistance. Other models, such as LSTMs, will also be developed for comparison with transformer models.

* The secondary objective is to compare the ability of LLM systems, such as chatGPT, to classify protein sequences. This will be achieved by creating embeddings for each sequence and utilising these embeddings with a transformer model. The hypothesis is that the modern LLM embeddings can create useful embeddings for the classification of proteins.