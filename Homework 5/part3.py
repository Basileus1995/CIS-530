# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:49:41 2018

@author: shash
"""

'''
order=0
k = 0.01
lms=[]
lm = train_char_lm("./cities_train/train/af.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/cn.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/de.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/fi.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/fr.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/in.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/ir.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/pk.txt", order=order, add_k=k)
lms.append(lm)
lm = train_char_lm("./cities_train/train/za.txt", order=order, add_k=k)
lms.append(lm)
'''


lms=[]
order = 3
add_k = 1
lm1=[]
for i in range(order,-1,-1):    
    lm = train_char_lm("./cities_train/train/af.txt", order=i, add_k=add_k)
    lm1.append(lm)
lms.append(lm1)

lm2=[]
for i in range(order,-1,-1):    
    lm = train_char_lm("./cities_train/train/cn.txt", order=i, add_k=add_k)
    lm2.append(lm)
lms.append(lm2)

lm3=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/de.txt", order=i, add_k=add_k)
    lm3.append(lm)
lms.append(lm3)

lm4=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/fi.txt", order=i, add_k=add_k)
    lm4.append(lm)
lms.append(lm4)

lm5=[]
for i in range(order,-1,-1):    
    lm = train_char_lm("./cities_train/train/fr.txt", order=i, add_k=add_k)
    lm5.append(lm)
lms.append(lm5)

lm6=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/in.txt", order=i, add_k=add_k)
    lm6.append(lm)
lms.append(lm6)

lm7=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/ir.txt", order=i, add_k=add_k)
    lm7.append(lm)
lms.append(lm7)

lm8=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/pk.txt", order=i, add_k=add_k)
    lm8.append(lm)
lms.append(lm8)

lm9=[]
for i in range(order,-1,-1):
    lm = train_char_lm("./cities_train/train/za.txt", order=i, add_k=add_k)
    lm9.append(lm)
lms.append(lm9)



lambda_list = []
lambda1 = set_lambdas(lms[0], "./cities_val/val/af.txt")
lambda_list.append(lambda1)
lambda2 = set_lambdas(lms[1], "./cities_val/val/cn.txt")
lambda_list.append(lambda2)
lambda3 = set_lambdas(lms[2], "./cities_val/val/de.txt")
lambda_list.append(lambda3)
lambda4 = set_lambdas(lms[3], "./cities_val/val/fi.txt")
lambda_list.append(lambda4)
lambda5 = set_lambdas(lms[4], "./cities_val/val/fr.txt")
lambda_list.append(lambda5)
lambda6 = set_lambdas(lms[5], "./cities_val/val/in.txt")
lambda_list.append(lambda6)
lambda7 = set_lambdas(lms[6], "./cities_val/val/ir.txt")
lambda_list.append(lambda7)
lambda8 = set_lambdas(lms[7], "./cities_val/val/pk.txt")
lambda_list.append(lambda8)
lambda9 = set_lambdas(lms[8], "./cities_val/val/za.txt")
lambda_list.append(lambda9)


i=0
countries=['af','cn','de','fi','fr','in','ir','pk','za']
writes=[]
with open('cities_test.txt') as f:
    for line in f:
        good_p=np.inf
        to_write=None
        for i in range(len(countries)):
            with open('te.txt','w') as g:
                g.write(line) 
            p=calculate_perplexity_with_backoff(lms[i],'te.txt',lambda_list[i])     
            #p=perplexity('te.txt',lms[i],order)
            # print(p)
            if p<good_p:
                good_p=p
                to_write=countries[i]
        writes.append(to_write)

with open('labels.txt','w') as f:
    f.write('\n'.join(writes))
    
