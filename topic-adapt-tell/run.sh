# conda info --envs
# conda environments:
#
#                *  /home/smelly/miniconda2/envs/tensorflow
#                   /home/smelly/miniconda2/envs/weibo
# base              /home/smelly/miniconda3
# test              /home/smelly/miniconda3/envs/test
# source activate2 tensorflow

# for training
# CUDA_VISIBLE_DEVICES=1 python2 -u main.py

# for test
date
CUDA_VISIBLE_DEVICES=0 python2 -u main_test.py \
    --topic 美食 \
    --tendency 正 \
    --themes 划算_好吃_粥_便宜_不贵_菜_质量_味道 \
    --caption_nums 5 \
    --batch_size 5
date
