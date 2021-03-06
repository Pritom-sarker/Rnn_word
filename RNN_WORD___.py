import tensorflow as tf
import numpy as np
import pandas as pd
import re

#Train or Test
is_train='Test'


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
with open("kafka.txt", "r", encoding='utf-8') as f:
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
print(charToInt)

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
print('X-> shape ', X.shape)
# normalize
X = X / float(nVocab)

# one hot encode the output variable
y = np.zeros([nExamples, nVocab])
for i, example in enumerate(dataY):
    lis = np.zeros(nVocab)
    lis[example] = 1
    y[i] = lis

print('Y-> shape', y.shape)

#-------------------------- Data shape and all things  done --------------------


#-----------------------------------start model with peramiter --------------------
batchSize = 24
iterations = 100000
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

def getTrainBatch():
    num = randint(0, nExamples - batchSize - 1)
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


if '__main__' == __name__:


    batchSize = 24
    iterations = 100000

    if is_train=='Train':

            for i in range(iterations):
                # Next Batch of reviews
                nextBatch, nextBatchLabels = getTrainBatch();
                sess.run(optimizer,{input_data: nextBatch, labels: nextBatchLabels})


                # Write summary to Tensorboard
                if (i % 25 == 0):

                    summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                    writer.add_summary(summary, i)
                if (i%10000==0):

                    print('Save  for {} steps'.format(i))
                    saver.save(sess, "Data/{}/For_{}_steps.ckpt".format(i,i))

            writer.close()

    elif is_train=='Test':

        with tf.Session() as ses:
            saver.restore(ses, "Data\90000\For_90000_steps.ckpt")

            num_of_char=200
           # for num in range(num_of_char):
            input="           s"
            in_size=len(input)
            str=getInput(input)

            In = np.reshape(str, (in_size-seqLength, seqLength, 1))
            print('X-> shape ', In.shape)
            # normalize
            In = In / float(nVocab)
            res=ses.run(prediction,feed_dict={input_data:In})
            print(np.argmax(res))





