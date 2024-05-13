# Bioinforming the youth Hackathon

## About the project

This analysis is part of my thesis with topic **Deep learning and LLMs revolutionize the drug development**. The project is focused on bioinformatics and aims to contribute valuable insights into the field of plant genetics and enhance our understanding of resistance patterns. The knowledge of plant gene resistance can be beneficial in several fields and applications beyond just agricultural improvement. Areas where this knowledge can have a significant impact rather than agricultural are also the drug discovery and development where discovering new medicinal properties and compounds in resistant plants, could lead to the development of new drugs and treatments for various diseases.

## Dataset

The dataset is provided as part of the [Bioinformatics Hackathon on Kaggle](https://www.kaggle.com/competitions/bioinformatics-hackathon-prg/overview). It includes various genetic sequences from plants, with labels indicating resistance.

## Objective of the project

* Main objective is to develop predictive models that can accurately determine whether certain plant genes are resistant to specific conditions or not. The model that we will build is a Transformer model that can have several advantages rather than other mondel. The core advantage of transformers is the attention mechanism, which allows the model to focus on different parts of the input sequence that are more relevant for making predictions. In the context of gene sequences, this means a transformer can learn to pay more attention to specific gene markers or motifs that indicate resistance

* Secondary objective is the comparison of LLMs like chatGPT on their ability to classify protein sequences with the creation of embedding for each sequence and the useage of these embeddings for the transformer model to see if the modern LLMs's embeddings can create usefull embeddings for the classification of proteins.