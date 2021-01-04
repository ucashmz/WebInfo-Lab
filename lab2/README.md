# Lab2

Two main methods are provided in this project, the Bayes and the RNN. RNN is the one we used to produced the final result.

To run any of them, you need to download `Stanford_core_nlp` from web and install `stanfordcorenlp` in pip. 

For RNN, you may also need to have `tensorflow(2.2.0)` and pretrained words embedding  `GloVe` (could be downloaded from web).

Set the path in `config.py` and run `main.py`. 

For the first time running RNN, you need to set `need_process` to `True`, this will produced embedding_matrix and other basic files.

