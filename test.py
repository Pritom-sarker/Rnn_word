import tensorflow as tf
import numpy as np
import pandas as pd
import re
import time


txt=""
#Train or Test
is_train='Train'

# ------------------------------ DATA PREPARE ----------------------


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower().replace("       ", " ")
    string = string.lower().replace("     ", " ")
    string = string.lower().replace("    ", " ")
    string = string.lower().replace("   ", " ")
    string = string.lower().replace("  ", " ")
    string = string.lower().replace("   ", " ")
    return re.sub(strip_special_chars, "", string.lower())


# In[22]:


allText = ""
with open("data.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    numWords = 0
    for line in lines:

        allText += (cleanSentences(line))
        numWords += len(line.split())

    chars = sorted(list(set(allText)))
    print('Num of words', numWords)
    print('num of char', len(chars))

# In[24]:


nChars = len(allText)
nVocab = len(chars)
seqLength = 10



#----------------------------- ENd of data prepare ----------------------------
# In[26]:

#------------------------------ create a dict where every char have a unique number --> ['a':5]
charToInt = dict((c, i) for i, c in enumerate(chars))


#according to sequence word x arry te r next char arakta arrey te
# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, nChars - seqLength,1):
    # input r output alada kore nichi akane ... seq onujai 1st 3 char "Seq_in" r last char SEQ_OUT e
    seq_in = allText[i:i + seqLength]
    seq_out = allText[i + seqLength]
    dataX.append([charToInt[char] for char in seq_in])
    dataY.append(charToInt[seq_out])

nExamples = len(dataX)
print("Total Examples: ", nExamples)


# reshape X to be [samples--> Num of example , time steps---> Num of sequence , features----> for word its 1]
X = np.reshape(dataX, (nExamples, seqLength, 1))
#print('X-> shape ', X.shape)
# normalize
X = X / float(nVocab)

# one hot encode the output variable
y = np.zeros([nExamples, nVocab])
for i, example in enumerate(dataY):
    lis = np.zeros(nVocab)
    lis[example] = 1
    y[i] = lis

#print('Y-> shape', y.shape)

#-------------------------- Data shape and all things  done --------------------


#-----------------------------------start model with peramiter --------------------

lstmUnits = 48
numDimensions = 1  # shape is [ [ [1][1][1] ] [...] [...]] --> so feature is 1
numClasses = nVocab  # Num of char s --> 36

# In[46]:
#-------------------------------- create Model -----------------------------

import tensorflow as tf

tf.reset_default_graph()

#input and output
labels = tf.placeholder(tf.float32, [None, numClasses])
input_data = tf.placeholder(tf.float32, [None, seqLength, numDimensions])

# RNN with LSTM
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.85)
value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

#For last layer
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

#calculate loss
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# In[49]:

# write sammary and graph ------------------------
import datetime
#to open graph -->> tensorboard --logdir=tensorboard/ --host localhost --port 8088
sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# In[48]:


from random import randint

#create Input as batch size

def getTrainBatch(b):
    num = b*batchSize
    labels = y[num:num + batchSize]
    arr = X[num:num + batchSize]
    return arr, labels

def getInput(str):
    data=[]
    for i in range(0, len(str) - seqLength, 1):
        # input r output alada kore nichi akane ... seq onujai 1st 3 char "Seq_in" r last char SEQ_OUT e
        seq_in = allText[i:i + seqLength]
        data.append([charToInt[char] for char in seq_in])
    return data;


def getKeysByValue( valueToFind):
    listOfKeys = list()
    listOfItems = charToInt.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    x="".join(listOfKeys)
    return x

if '__main__' == __name__:

    file = open("sammary.txt","w")
    txt = 'Total Num of words : {} \nNum of Char:: {} \nTotal Sentence: {} \n'.format(numWords, len(chars), nExamples)
    file.write(txt)
    file.close()

    batchSize = 1000000
    iterations = 1000


    if is_train=='Train':

            file = open("sammary.txt", "a")
            txt = '\n\n\n******************** Loop Start*******************\n\n\n'
            file.write(txt)
            file.close()

            Final_s=time.time()

            for i in range(iterations+1):
                # Next Batch of reviews

                start1 = time.time()
                for bat in range(1,int(nExamples/batchSize) ):

                    los=0
                    nextBatch, nextBatchLabels = getTrainBatch(bat)
                    los=+ np.sum( sess.run(loss,{input_data: nextBatch, labels: nextBatchLabels}))
                    sess.run(optimizer,{input_data: nextBatch, labels: nextBatchLabels})

                print("{}-> Loss is {} ------------------------------------------------->>>>".format(i, los))
                end1 = time.time()
                print("For {}th loop Time:: {} sec ".format(i,end1 - start1))

                # Write summary to Tensorboard
                if (i % 2 == 0):
                    summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                    writer.add_summary(summary, i)
                if (i%100==0):
                    sv_time=time.time()
                    ti=sv_time-Final_s

                    file = open("sammary.txt", "a")
                    txt = "\n\n******************************** For {}th Saving Time:: {} sec **************************************".format(i,ti)
                    file.write(txt)
                    file.close()
                    print('******************************Save  for {} steps ********************************************************************'.format(i))
                    saver.save(sess, "Data/{}/For_{}_steps.ckpt".format(i,i))

                file = open("sammary.txt", "a")
                txt = "\n\nFor {}th loop **************** \nTime:: {} sec \nLoss is:: {}".format(i,end1 - start1,los)
                file.write(txt)
                file.close()

            writer.close()

            Final_c=time.time()
            print("Total Time :: {} sec".format( Final_c-Final_s))


    elif is_train=='Test':

        with tf.Session() as ses:
            saver.restore(ses, "Data\90000\For_90000_steps.ckpt")

            num_of_char=200
            input = "hello my name is bisal"
            for num in range(num_of_char):
                in_size = len(input)-10-1
                str=input[in_size:]
                str=getInput(str)
                In = np.reshape(str, (1, seqLength, 1))
                # normalize
                In = In / float(nVocab)
                res=ses.run(prediction,feed_dict={input_data:In})

                input=input+getKeysByValue(np.argmax(res))

            print(input)






