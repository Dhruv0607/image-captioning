## Importing the required libraries
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.compat.v1.keras.backend import set_session
#from keras.backend.tensorflow_backend import set_session
import keras
import sys, time, os, warnings 
import numpy as np
import pandas as pd 
from collections import Counter 
warnings.filterwarnings("ignore")
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))




def checking_model():
    return 'This is to just check whether the model is working or not'



r = checking_model()
print(r)

def train_model():

    ## Importing the required libraries
    import matplotlib.pyplot as plt
    import tensorflow as tf
    #from tensorflow.compat.v1.keras.backend import set_session
    #from keras.backend.tensorflow_backend import set_session
    #from keras import preprocessing
    import sys, time, os, warnings 
    import numpy as np
    import pandas as pd 
    from collections import Counter 

    print('Train model called')

    '''
    Importing the image dataset and its respective captions
    '''

    ## The location of the Flickr8K_ images
    dir_Flickr_jpg = "Data/Flickr8k_Dataset/"
    ## The location of the caption file
    dir_Flickr_text = "Data/Flickr8k.token.txt"

    jpgs = os.listdir(dir_Flickr_jpg)
    print("The number of jpg flies in Flicker8k: {}".format(len(jpgs)))


    #Finding the captions for each image.
    file = open(dir_Flickr_text,'r', encoding='utf8')
    text = file.read()
    file.close()


    datatxt = []
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue
        w = col[0].split("#") # Splitting the caption dataset at the required position
        datatxt.append(w + [col[1].lower()])

    df_txt = pd.DataFrame(datatxt,columns=["filename","index","caption"])


    uni_filenames = np.unique(df_txt.filename.values)
    print("The number of unique file names : {}".format(len(uni_filenames)))
    print("The distribution of the number of captions for each image:")
    Counter(Counter(df_txt.filename.values).values())
    print(df_txt[:5])



    # Cleaning caption for further analysi
    # Defining a function to calculate the top 3 words in all the captions available for the images
    def df_word(df_txt):
        vocabulary = []
        for txt in df_txt.caption.values:
            vocabulary.extend(txt.split())
        print('Vocabulary Size: %d' % len(set(vocabulary)))
        ct = Counter(vocabulary)
        dfword = pd.DataFrame({"word":list(ct.keys()),"count":list(ct.values())})
        dfword = dfword.sort_values("count",ascending=False)
        dfword = dfword.reset_index()[["word","count"]]
        return(dfword)
    dfword = df_word(df_txt)
    dfword.head(3)



    # Cleaning captions for further processing
    import string
    text_original = "I ate 1000 apples and a banana. I have python v2.7. It's 2:30 pm. Could you buy me iphone7?"

    print(text_original)
    print("\nRemove punctuations..")
    def remove_punctuation(text_original):
        text_no_punctuation = text_original.translate(str.maketrans('','',string.punctuation))
        return(text_no_punctuation)
    text_no_punctuation = remove_punctuation(text_original)
    print(text_no_punctuation)


    print("\nRemove a single character word..")
    def remove_single_character(text):
        text_len_more_than1 = ""
        for word in text.split():
            if len(word) > 1:
                text_len_more_than1 += " " + word
        return(text_len_more_than1)
    text_len_more_than1 = remove_single_character(text_no_punctuation)
    print(text_len_more_than1)

    print("\nRemove words with numeric values..")
    def remove_numeric(text,printTF=False):
        text_no_numeric = ""
        for word in text.split():
            isalpha = word.isalpha()
            if printTF:
                print("    {:10} : {:}".format(word,isalpha))
            if isalpha:
                text_no_numeric += " " + word
        return(text_no_numeric)
    text_no_numeric = remove_numeric(text_len_more_than1,printTF=True)
    print(text_no_numeric)
    # to remove the punctuations
    def text_clean(text_original):
        text = remove_punctuation(text_original)
        text = remove_single_character(text)
        text = remove_numeric(text)
        return(text)


    for i, caption in enumerate(df_txt.caption.values):
        newcaption = text_clean(caption)
        df_txt["caption"].iloc[i] = newcaption


        
        
        
        '''
        Adding start and end sequence tokens for each captions
        '''
    from copy import copy
    def add_start_end_seq_token(captions):
        caps = []
        for txt in captions:
            txt = 'startseq ' + txt + ' endseq'
            caps.append(txt)
        return(caps)
    df_txt0 = copy(df_txt)
    df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])
    df_txt0.head(5)
    del df_txt


    '''
    Loading VGG16 model and wights to extract features fromt the images
    '''


    from tensorflow.keras.applications import VGG16

    modelvgg = VGG16(include_top=True,weights=None)
    ## load the locally saved weights 
    modelvgg.load_weights("Data/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    modelvgg.summary()


    ##
    # Deleting the last layer of the model
    ##

    from keras import models
    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    ## show the deep learning model
    modelvgg.summary()

    '''
    Feature extraction
    '''
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import preprocess_input
    from collections import OrderedDict

    images = OrderedDict()
    npix = 224 #image size is fixed at 224 because VGG16 model has been pre-trained to take that size.
    target_size = (npix,npix,3)
    data = np.zeros((len(jpgs),npix,npix,3))
    for i,name in enumerate(jpgs):
        # load an image from file
        filename = dir_Flickr_jpg + '/' + name
        image = load_img(filename, target_size=target_size)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        nimage = preprocess_input(image)
        
        y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
        images[name] = y_pred.flatten()



    '''
    Mergin the images and the captions for training
    '''

    dimages, keepindex = [],[]
    # Creating a datframe where only first caption is taken for processing
    df_txt0 = df_txt0.loc[df_txt0["index"].values == "0",: ]
    for i, fnm in enumerate(df_txt0.filename):
        if fnm in images.keys():
            dimages.append(images[fnm])
            keepindex.append(i)

    #fnames are the names of the image files        
    fnames = df_txt0["filename"].iloc[keepindex].values
    #dcaptions are the captions of the images 
    dcaptions = df_txt0["caption"].iloc[keepindex].values
    #dimages are the actual features of the images
    dimages = np.array(dimages)


    '''
    Tokenizing the captions for further processing
    '''

    from keras.preprocessing.text import Tokenizer
    ## the maximum number of words in dictionary
    nb_words = 6000
    tokenizer = Tokenizer(nb_words=nb_words)
    tokenizer.fit_on_texts(dcaptions)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocabulary size : {}".format(vocab_size))
    dtexts = tokenizer.texts_to_sequences(dcaptions)
    print(dtexts[:5])



    '''
    Splitting the training and test data
    '''


    prop_test, prop_val = 0.2, 0.2 

    N = len(dtexts)
    Ntest, Nval = int(N*prop_test), int(N*prop_val)

    def split_test_val_train(dtexts,Ntest,Nval):
        return(dtexts[:Ntest], 
            dtexts[Ntest:Ntest+Nval],  
            dtexts[Ntest+Nval:])

    dt_test,  dt_val, dt_train   = split_test_val_train(dtexts,Ntest,Nval)
    di_test,  di_val, di_train   = split_test_val_train(dimages,Ntest,Nval)
    fnm_test,fnm_val, fnm_train  = split_test_val_train(fnames,Ntest,Nval)


    '''
    Finding the max length of the caption
    '''
    maxlen = np.max([len(text) for text in dtexts])
    print(maxlen)

    '''
    Processing the captions and images as per the required shape by the model
    '''


    from keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    def preprocessing(dtexts,dimages):
        N = len(dtexts)
        print("# captions/images = {}".format(N))

        assert(N==len(dimages)) # using assert to make sure that length of images and captions are always similar
        Xtext, Ximage, ytext = [],[],[]
        for text,image in zip(dtexts,dimages):
            # zip() is used to create a tuple of iteratable items
            for i in range(1,len(text)):
                in_text, out_text = text[:i], text[i]
                in_text = pad_sequences([in_text],maxlen=maxlen).flatten()# using pad sequence to make the length of all captions equal
                out_text = to_categorical(out_text,num_classes = vocab_size) # using to_categorical to 

                
                Xtext.append(in_text)
                Ximage.append(image)
                ytext.append(out_text)

        Xtext  = np.array(Xtext)
        Ximage = np.array(Ximage)
        ytext  = np.array(ytext)
        print(" {} {} {}".format(Xtext.shape,Ximage.shape,ytext.shape))
        return(Xtext,Ximage,ytext)


    Xtext_train, Ximage_train, ytext_train = preprocessing(dt_train,di_train)
    Xtext_val,   Ximage_val,   ytext_val   = preprocessing(dt_val,di_val)
    # pre-processing is not necessary for testing data
    #Xtext_test,  Ximage_test,  ytext_test  = preprocessing(dt_test,di_test)


    print('Building the LSTM Model')

    '''
    Building the LSTM Model
    '''

    # RELU - Reactified Linear Activation Function
    from keras import layers
    from keras.layers import Input, Flatten, Dropout, Activation
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    print(vocab_size)
    ## image feature

    dim_embedding = 64

    input_image = layers.Input(shape=(Ximage_train.shape[1],))
    fimage = layers.Dense(256,activation='relu',name="ImageFeature")(input_image)
    ## sequence model
    input_txt = layers.Input(shape=(maxlen,))
    ftxt = layers.Embedding(vocab_size,dim_embedding, mask_zero=True)(input_txt)
    ftxt = layers.LSTM(256,name="CaptionFeature",return_sequences=True)(ftxt)
    #,return_sequences=True
    #,activation='relu'
    se2 = Dropout(0.04)(ftxt)
    ftxt = layers.LSTM(256,name="CaptionFeature2")(se2)
    ## combined model for decoder
    decoder = layers.add([ftxt,fimage])
    decoder = layers.Dense(256,activation='relu')(decoder)
    output = layers.Dense(vocab_size,activation='softmax')(decoder)
    model = models.Model(inputs=[input_image, input_txt],outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())


    print("Training the LSTM Model")


    # fit model
    from time import time
    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    #start = time.time()
    hist = model.fit([Ximage_train, Xtext_train], ytext_train, 
                    epochs=6, verbose=2, 
                    batch_size=32,
                    validation_data=([Ximage_val, Xtext_val], ytext_val),callbacks=[tensorboard])
    #end = time.time()
    #print("TIME TOOK {:3.2f}MIN".format((end - start )/60))



