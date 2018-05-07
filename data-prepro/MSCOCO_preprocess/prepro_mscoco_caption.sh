# download and preprocess captions
# wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
# unzip captions_train-val2014.zip
# rm captions_train-val2014.zip

# output k_annotations_%s2014.pkl
# output id2name, name2id, id2caption

python preprocess_entity.py train
python preprocess_entity.py test
python preprocess_entity.py val

# output tokenized_phase_caption.pkl
python preprocess_token.py train
python preprocess_token.py val
python preprocess_token.py test
