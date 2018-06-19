# encoding: utf-8
import numpy as np
import json

for i in [str(x) for x in range(0,1000,200)]:
    cub = np.load('./SeqGAN_samples_sample/cub_no_scheduled/cub_' + i + '.npz')
    with open('weibo_' + i + '.txt', 'w') as f:
        for j in range(len(cub['string'])):
            i = cub['string'].item(j)
            f.write(i + '\n')

test = json.load(open('./cub/K_test_annotation.json'))
with open('weibo_gt.txt', 'w') as f:
    for i in cub['id']:
        topic = test[i][0]['topic']
        caption = test[i][0]['caption']
        for j in topic:
            f.write(j.encode('utf-8') + ' ')
        f.write('#' + caption.encode('utf-8') + '\n')


