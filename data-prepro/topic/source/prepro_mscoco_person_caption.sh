echo 'preprocess_entity'
echo 'train'
python preprocess_entity.py train
echo 'test'
python preprocess_entity.py test
echo 'val'
python preprocess_entity.py val

echo 'preprocess_token'
echo 'train'
python preprocess_token.py train
echo 'val'
python preprocess_token.py val
echo 'test'
python preprocess_token.py test

