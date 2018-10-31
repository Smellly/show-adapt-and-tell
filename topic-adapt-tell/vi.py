# encoding: utf-8
import numpy as np
import json

for i in [str(x) for x in range(0,5000,200)]:
    cub = np.load('./SeqGAN_samples_sample/cub_no_scheduled/cub_' + i + '.npz')
    with open('results_0713/weibo_' + i + '.txt', 'w') as f:
        for j in range(len(cub['string'])):
            k = cub['string'].item(j)
            try:
                f.write(k + '\n')
            except:
                print(k)

test = json.load(open('./cub/K_test_annotation.json'))
with open('weibo_gt.txt', 'w') as f:
    for i in cub['id']:
        topic = test[i][0]['topic']
        caption = test[i][0]['caption']
        for j in topic:
            f.write(j.encode('utf-8') + ' ')
        f.write('#' + caption.encode('utf-8') + '\n')


