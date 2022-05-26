import numpy as np

exp_num = 9

submission = ['id,pred\n']
f2 = './nmt_simple.pred.csv'

pred = np.load('./result/pred_ar_attn_nllloss_0.npy')

with open(f2, 'rb') as f:
    file = f.read().decode('utf-8')
    content = file.split('\n')[:-1] # column name

    for idx, line in enumerate(content):
        if idx == 0: # first line is id, pred so just skip it
            continue
        tmp1 = line.split(',') # split the id and prediction result by ,
        # res = final_pred[idx-1].item() # get the final prediction result of this id
        res = pred[idx-1]
        tmp2 = tmp1[0] + ',' + str(res) + '\n'
        submission.append(tmp2)

with open(f'./submissions/20214047_{exp_num}.csv', 'w') as f:
    f.write(''.join(submission)) # store the submission file
