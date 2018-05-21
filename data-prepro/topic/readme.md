
#### Captions Tokenization
1. Clone the [NeuralTalk2](https://github.com/karpathy/neuraltalk2/tree/bd8c9d879f957e1218a8f9e1f9b663ac70375866) repository and head over to the coco/ folder and run the IPython notebook to generate a json file for Karpathy split: `coco_raw.json`.
2. Run the following script:
` python extract_topic.py filter` to filter "person" topic from downloaded caption
`./prepro_mscoco_caption.sh` for tokenizing captions.
3. Run `python prepro_coco_annotation.py` to generate annotation json file for testing. 
4. Run `python extract_topic.py extract` to add themes into json. Each caption's verb and noun is considered as themes

#### Target Captions Tokenization
1. Download the [description data](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE).
2. Run `python get_split.py` to generate dataset split following the ECCV16 paper "Generating Visual Explanations".
3. Run `python prepro_cub_annotation.py` to generate annotation json file for testing. 
4. Run `./cub_preprocess.sh` for tokenization.

#### AI Challenge Tokenization
1. Run `python get_split.py` to generate `splits.pkl` and so on.
2. Run `python preprocess_ai_entity.py phase.`
3. Run `python preprocess_ai_token.py`
4. Run `python prepro_ai_annotation.py`
