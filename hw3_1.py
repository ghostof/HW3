###COGS109
### HW3
### Gustav Sto. Tomas
### A15358078


import pandas as pd
import pandas.tseries
from pandas.core import datetools #pandas keeps bugging me about needing update...
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('hw3_divseq_data.csv')
df.head()

### a)

#for df.Lars2

lars = []
lars0 = []
for i in range(len(df.Lars2)):
    if df.mature[i] == 1:
        lars.append([df.Lars2[i]])
    else: lars0.append([df.Lars2[i]])
        
#for df.Malat1   
       
malat = []
malat0 = []
for i in range(len(df.Malat1)):
    if df.mature[i] == 1:
        malat.append([df.Malat1[i]])
    else: malat0.append([df.Malat1[i]])     


plt.figure(1)
plt.boxplot([lars,lars0],labels = ['mature','immature'])
plt.ylabel('expression level (logTPM)')
plt.title('Lars2')

plt.figure(2)
plt.boxplot([malat,malat0],labels = ['mature','immature'])
plt.ylabel('expression level (logTPM)')
plt.title('Malat1')

plt.show()

### there are a few extreme outliers in the boxplots...


"""
#matrix = []
#matrix0 = []
#for i in range(len(df.Lars2)):
#    if df.mature[i] == 1:
#        matrix.append([df.Lars2[i]])#, df.mature[i]])
#    else: matrix0.append([df.Lars2[i]])#, df.mature[i]])

#plt.figure(1)
#plt.boxplot([matrix,matrix0],labels = ['mature','immature'])
#plt.ylabel('expression level (logTPM)')
#plt.show()"""



### b) Lars2 displays NO overlap in spread inside the interquartile range, so there IS a difference between mature and immature (cells?). We could almost use Lars2 to make a perfect classifier. However, there are also a few extreme outliers of the immature (cells?) that fall inside and above the median of the mature expression levels, meaning that these would be falsely classified as mature, and therefore the classifier would not be perfect.
#4b) Lars2 displays NO overlap in spread inside the interquartal range, so there IS a difference between mature and immature neurons. We could almost use Lars2 to make a perfect classifier. However, there are also a few extreme outliers of the immature neurons that fall inside and above the median of the mature expression levels, meaning that these would be falsely classified as mature, and therefore the classifier would not be perfect. Furthermore, in the plot for Malat1, the expression levels for mature show similar values as the immature ones in Lars2, which is why using Lars2 as training set would not classify correctly for Malat1 as a test set. Conclusion: No.


#c) 

#4c)

logreg = smf.logit(formula = 'mature ~ Lars2', data=df).fit()
print(logreg.summary())
print(logreg.pvalues)
print('p-value for Lars2 coeffecient is 3.455778e-34; my conclusion is that there is a significantly positive correlation between Lars2 neurons\'s expression values and maturity==1.')


#4d)

p = logreg.predict(df.Lars2)
#Pdf = []
#plt.figure(3)
fig, ax = plt.subplots()
ax.scatter(df.Lars2, p)
plt.xlabel('Lars2 expression level (log TPM)')
plt.ylabel('p of maturity')
plt.title('Prediction for Lars2')
ax.grid(linewidth=0.5)
plt.show()


p8 = []
for i in range(len(df.Lars2)):
    if df.Lars2[i] == 8.0:
        p8.append(p[i])
    if df.Lars2[i] == 7.9:
        p8.append(p[i])
    if df.Lars2[i] == 8.1:
        p8.append(p[i])
print(p8)

# as there is no true value 8.0 in Lars2, our closest value is 8.1, for which the predicted probability is 0.9: a very strong probability that the neuron is mature.


#4e)

#pm = []
#for i in range(len(p)):
#    if p[i] > 0.5:
#        pm.append(p[i])


df['pred']=p
df['pred_Mature'] = 1*(df.pred) > 0.5
df.head(10)

correct = []
for i in range(len(df.pred_Mature)):
    if df.pred_Mature[i] == True:
        correct.append(df.pred_Mature[i])

print('correctly predicted:', len(correct))
print('total amount of predictions:', len(df.pred_Mature))

sensitivity = len(correct)/len(df.pred_Mature)
print('sensitivity:', sensitivity)


#4f
df['pred_Immature'] = 1 *(df.pred) < 0.5

correct_im = []
for i in range(len(df.pred_Immature)):
    if df.pred_Immature[i] == True:
        correct_im.append(df.pred_Immature[i])

print('correctly predicted immature neurons:', len(correct_im))
print('total anount of predictions:', len(df.pred))
specificity = len(correct_im)/len(df.pred)
print('specificity:', specificity)


#4g

df['pred_not_immature'] = 1 *(df.pred) > 0.2

correct_nim = []
for i in range(len(df.pred_not_immature)):
    if df.pred_not_immature[i] == True:
        correct_nim.append(df.pred_not_immature[i])

correct_nim2 = []
for i in range(len(df.pred_not_immature)):
    if df.pred_not_immature[i] == True and df.pred_Immature[i] == True:
        correct_nim2.append(df.pred_not_immature[i])

#print(correct_nim2)        
        
print('correctly predicted \"not immature\" neurons:', len(correct_nim))
print('total anount of predictions:', len(df.pred))
sensitivity_nim = len(correct_nim)/len(df.pred)

specificity_nim = len(correct_nim2)/len(df.pred) 
print('sensitivity:', sensitivity_nim)
print('specificity:', specificity_nim)


#df.head(20)
# sensitivity increase because we get fewer false negatives
# specificity decrease because we get more false positives
#we may want to set the classification threshold to 20% for non-binary classifiers,
# in which case we would maybe want an 'almost mature neuron' class.
# furthermore, if the data does not contain any values above 0.5,
# we may still want to discriminate between the data points that are closer to 0.5 than the ones cloaser to 0.

#4h

#4h
    
fig, ax = plt.subplots()
untrues = [i for i,val in enumerate(df.mature) if val != True]
trues = [i for i, val in enumerate(df.mature) if val==True]
ax.scatter(df.Lars2[untrues],df.Malat1[untrues],color='b',marker='x',label='Immature neurons')
ax.scatter(df.Lars2[trues], df.Malat1[trues],color='r', marker='^',label='Mature neurons')
plt.xlabel('Lars2 expression level (log TPM)')
plt.ylabel('Malat1 expression level (log TPM)')
plt.legend()
plt.show()


#4i)

logreg2 = smf.logit(formula = 'mature ~ Lars2 + Malat1', data=df).fit()
print(logreg2.summary())
print(logreg2.pvalues)
print(logreg2.tvalues)
print('p-values are < 0.01 for Lars2 and Malat (and intercept), and t-statistics are > 2 for both Lars and Malat1 (but not the intercept)')


#4j

pred_both = df[['Lars2','Malat1']]

#df2 = pd.DataFrame([df.Lars2,df.Malat1])
df['pred_both'] = logreg2.predict(pred_both)
df['pred_both_mature'] = 1*(df.pred_both > 0.5)
df['pred_both_immature'] = 1 *(df.pred_both < 0.5)
#df.head()


correct_mb = []
for i in range(len(df.pred_both_mature)):
    if df.pred_both_mature[i] == 1:
        correct_mb.append(df.pred_both_mature[i])
        
correct_imb = []
for i in range(len(df.pred_both_mature)):
    if df.pred_both_immature[i] == 1:
        correct_imb.append(df.pred_both_immature[i])

    

print('correctly predicted mature neurons:', len(correct_mb))
print('total abount of predictions:', len(df.pred_both_mature))

sensitivity = len(correct_mb)/len(df.pred_both_mature)
print('sensitivity:', sensitivity)

specificity = len(correct_imb)/len(df.pred_both_mature)
print('specificity:', specificity)
#both sensitivity and specifiity are similar to the values in 43 and 4f, though sensitivity is slightly larer and specificity slightly smaller




