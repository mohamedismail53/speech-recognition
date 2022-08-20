#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import walk
import numpy as np
import librosa
from dtw import *
import matplotlib.pyplot as plt
from scipy.spatial import distance


# ## script for loading .wav file and convert it into mf MFCC

# In[2]:


# load the data into Test data and Reference data 2-D arrays
Test = [[[] for h in range(124)] for g in range(300)]
Reference = [[[] for h in range(124)] for g in range(100)]
# counter for Test and Reference data
test_cnt = 0
ref_cnt = 0


# In[3]:


teams = []
#Change this to the path of the Recording Files on your machine
path=r'C:\Users\hp\OneDrive\Desktop\ECE\DSP\Project_dtw\Recordings'
for (dirpath, dirnames, filenames) in walk(path):
    teams.extend(dirnames)
    for team in teams:
        # print(team)
        students = []
        # access all students in the group
        for (dirpath, dirnames, filenames) in walk(path+'\\'+str(team)):
            students.extend(dirnames)
            for student in students:
                # print(student)
                genders = []
                # access different reconds for each student male/femal/child(directory)
                for (dirpath, dirnames, filenames) in walk(path+'\\'+str(team)+'\\'+str(student)):
                    genders.extend(dirnames)
                    for gender in genders:
                        # print(gender)
                        audios = []
                        #print("Loading "+str(team)+' ---> '+str(student)+' ----> '+str(gender))
                        for (dirpath, dirnames, filenames) in walk(path+'\\'+str(team)+'\\'+str(student)+'\\'+str(gender)):
                            audios.extend(filenames)
                            # Test or Ref flag if 1 Test, if 0 Ref
                            TR_flag = 0
                            # word counter for 123 words
                            word_cnt = 0
                            for audio in audios:
                                record, samplerate = librosa.load(path+'\\'+str(team)+'\\'+str(student)+'\\'+str(gender)+'\\'+str(audio),sr=16000,mono=False)
                                #print(record.shape)
                                if record.ndim == 2:
                                    print(str(audio))
                                    raise "Stereo"
                                record_mfcc = librosa.feature.mfcc(y=record, sr=samplerate,n_mfcc=39)
                                
                                if str(audio)[-5]=='T':
                                    #print(test_cnt, word_cnt)
                                    Test[test_cnt][word_cnt].append(record_mfcc)
                                    
                                    # Test[test_cnt][123].append(str(audio)[5])
                                    TR_flag = 1
                                if str(audio)[-5]=='R':
                                    #print(ref_cnt, word_cnt)
                                    print(str(audio),ref_cnt)
                                    Reference[ref_cnt][word_cnt].append(record_mfcc)
                                    TR_flag = -1
                                    # Reference[ref_cnt][123].append(str(audio)[5])
                                word_cnt = word_cnt+1
                            if TR_flag == 1:
                                Test[test_cnt][123].append(str(audio)[5])
                                test_cnt = test_cnt+1
                            if TR_flag == -1:
                                Reference[ref_cnt][123].append(str(audio)[5])
                                ref_cnt = ref_cnt+1
                            if word_cnt<123 and  word_cnt!= 0:
                                raise "Files are less than 123"
                            if word_cnt>123:
                                raise "Files are more than 123"    
                            break
                    break
            break
    break


# In[4]:


print(ref_cnt)


# In[5]:


x = np.array(Test[100][0]).reshape((np.array(Test[100][0]).shape[1],np.array(Test[100][0]).shape[2]))
#x = x[0:20,:]
#x.shape


# ## Extracting threshold array

# In[6]:


def extract_threshold(Test,reference,word_num):
    dtw_W1=[]
    dtw_W2=[]
    for x in range(len(Test)):
        try:
            if Test[x][0]==[]:
                break
            word_1=np.array(Test[x][word_num]).reshape((np.array(Test[x][word_num]).shape[1],np.array(Test[x][word_num]).shape[2]))
            word_2=np.array(Test[x][word_num+1 if word_num%2==0 else word_num-1]).reshape((np.array(Test[x][word_num+1 if word_num%2==0 else word_num-1]).shape[1],np.array(Test[x][word_num+1 if word_num%2==0 else word_num-1]).shape[2]))
            ref_1=np.array(reference[word_num]).reshape((np.array(reference[word_num]).shape[1],np.array(reference[word_num]).shape[2]))
            #ref_2=np.array(reference[2*pair+1]).reshape((np.array(reference[2*pair+1]).shape[1],np.array(reference[2*pair+1]).shape[2]))
            #print(word_1.shape,word_2.shape,ref_1.shape)
            dtw_W1.append(dtw(word_1.T, ref_1.T,keep_internals=True).distance)    
            dtw_W2.append(dtw(word_2.T, ref_1.T,keep_internals=True).distance)
        except:
            if np.array(dtw_W1).shape[0] > np.array(dtw_W2).shape[0]:
                dtw_W1.pop()
            if np.array(dtw_W1).shape[0] < np.array(dtw_W2).shape[0]:
                dtw_W2.pop()
            pass
    #Indecies_W1 = [dtw_W1.index(x) for x in sorted(dtw_W1)]    
    dtw_W1.sort() 
    dtw_W2.sort(reverse=True)
    plt.figure()
    plt.hist(dtw_W1[0:50],color='orange',alpha=0.8)
    #plt.hist(np.array(dtw_W2)[Indecies_W1[0:20]],color='g',alpha=0.8)
    plt.hist(dtw_W2[0:50],color='g',alpha=0.8)
    plt.xlabel(str(word_num))
    return [max(dtw_W1[0:50]),max(dtw_W2[0:50])]
    


# ## Looping through all words to find the thresholds
# #####     Remove the comment to see the graphs that we used to get these thresholds

# In[7]:


Thresholds = []
for x in range(122):
    Thresholds.append(extract_threshold(Test,Reference[1],x))
Thresholds_numpy=numpy.array(Thresholds)    


# In[8]:


#Reference_numpy=numpy.array(Reference[0][0])
#print(Reference_numpy.shape)
#counter = 0
#for person in range(11):
    #for word in range(123):
       # np.save(str(counter),np.array(Reference[person][word]).reshape(np.array(Reference[person][word]).shape[1],np.array(Reference[person][word]).shape[2]))
       # counter = counter + 1
        
    


# In[9]:


#np.save('Thresholds',Thresholds_numpy)
#Reference_mfcc=numpy.array(Reference)
#np.save('References_mfcc',Reference_mfcc)


# ## Constructing the Table 
# 

# In[10]:


def closest_reference(Speaker):
    distance_i=[]
    word_1=np.array(Speaker[-2]).reshape((np.array(Speaker[-2]).shape[1],np.array(Speaker[-2]).shape[2]))
    for i in range(ref_cnt):
       # try:
            ref_i=np.array(Reference[i][-2]).reshape((np.array(Reference[i][-2]).shape[1],np.array(Reference[i][-2]).shape[2]))
            distance_i.append(dtw(word_1.T, ref_i.T,keep_internals=True).distance)
       # except:
          #  pass
    return np.argmin(np.array(distance_i))

def closest_reference_byword(el_word):
    distance_i=[]
    word_1=np.array(el_word).reshape((np.array(el_word).shape[1],np.array(el_word).shape[2]))
    for i in range(ref_cnt):
        try:
            ref_i=np.array(Reference[i][-2]).reshape((np.array(Reference[i][-2]).shape[1],np.array(Reference[i][-2]).shape[2]))
            distance_i.append(dtw(word_1.T, ref_i.T,keep_internals=True,step_pattern=asymmetric).distance)
        except:
            pass
    return np.argmin(np.array(distance_i))


# In[11]:


Males=np.zeros((122,3))
Females=np.zeros((122,3))
Children=np.zeros((122,3))
mismatched=[]
for p in range(test_cnt):
    closest_index=closest_reference(Test[p])
    for w in range(122):
           word_w =np.array(Test[p][w]).reshape((np.array(Test[p][w]).shape[1],np.array(Test[p][w]).shape[2])) 
           ref_w= np.array(Reference[closest_index][w]).reshape((np.array(Reference[closest_index][w]).shape[1],np.array(Reference[closest_index][w]).shape[2])) 
           distance_w = dtw(word_w.T, ref_w.T,keep_internals=True).distance
           if Test[p][-1] == ['M']:
                if distance_w < Thresholds[w][0]:
                    Males[w][0]=Males[w][0]+1
                elif distance_w > Thresholds[w][0]  and distance_w < Thresholds[w][1]:
                    Males[w][1]=Males[w][1]+1
                else:
                    Males[w][2]=Males[w][2]+1
                    mismatched.append([p,w])
           elif Test[p][-1] == ['F']:
                if distance_w < Thresholds[w][0]:
                    Females[w][0]=Females[w][0]+1
                elif distance_w > Thresholds[w][0]  and distance_w < Thresholds[w][1]:
                    Females[w][1]=Females[w][1]+1
                else:
                    Females[w][2]=Females[w][2]+1                    
           elif Test[p][-1] == ['C']:
                if distance_w < Thresholds[w][0]:
                    Children[w][0]=Children[w][0]+1
                elif distance_w > Thresholds[w][0]  and distance_w < Thresholds[w][1]:
                    Children[w][1]=Children[w][1]+1
                else:
                    Children[w][2]=Children[w][2]+1  
           else:
            print('Something is worng')


# In[12]:


right = 0
total=0
for x in range(122):
    right=right+Males[x][0]
    total=total+Males[x][0]+Males[x][1]+Males[x][2]
right/total    


# In[13]:


import pandas as pd
## convert your array into a dataframe
df = pd.DataFrame (Children)
## save to xlsx file
filepath = 'my_excel_file.xlsx'
df.to_excel(filepath, index=False)


# ## Drawing the error of 5 mismatched Words

# In[31]:


word_m=np.array(Test[48][62]).reshape((np.array(Test[48][62]).shape[1],np.array(Test[48][62]).shape[2]))
ref_m=np.array(Reference[closest_reference(Test[48])][62]).reshape((np.array(Reference[closest_reference(Test[48])][62]).shape[1],np.array(Reference[closest_reference(Test[48])][62]).shape[2]))
def dist(word_dist , ref_dist):
    alignment = dtw(word_dist.T, ref_dist.T,keep_internals=True)
    wq=warp(alignment,index_reference=False)
    warped= word_dist.T[wq]
    print(warped.shape,ref_dist.T.shape)
    ret=[]
    for n in range(len(warped)):
        ret.append(distance.euclidean(warped[n],ref_dist.T[n]))
    return ret


# In[32]:


dist(word_m,ref_m)


# In[34]:


plt.plot(dist(word_m,ref_m))


# In[ ]:




