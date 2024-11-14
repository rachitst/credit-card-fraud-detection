from flask import Flask ,render_template,request,jsonify,session
from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import sqlite3 as sql
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from flask_bootstrap import Bootstrap
import numpy as np
from sklearn.utils import shuffle
import os
from flask import Flask, render_template, request, url_for,send_from_directory
import os
import tensorflow as tf
#from flask import Flask,render_template, flash, redirect,url_for,request
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Markup
import numpy as np
from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import pandas as pd
import os 
#import josn
app=Flask(__name__)
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPool1D,Conv2D

# Data processing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
print(tf.__version__)


# In[3]:


data = pd.read_csv("./creditcard.csv")


# In[4]:


data


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


data["Class"].value_counts()


# In[9]:


non_fraud=data[data["Class"]==0]


# In[10]:


non_fraud.shape


# In[11]:


fraud=data[data["Class"]==1]


# In[12]:


fraud.shape


# In[13]:


non_fraud_sample=non_fraud.sample(fraud.shape[0])


# In[14]:


non_fraud_sample.shape


# In[15]:


bal_data=fraud.append(non_fraud_sample,ignore_index=True)


# In[16]:


bal_data


# In[17]:


bal_data["Class"].value_counts()


# In[18]:


features = bal_data.drop("Class",axis=1)
Labels=bal_data["Class"]


# In[19]:


X_train,X_test,Y_train,Y_test = train_test_split(features,Labels,test_size=0.25,random_state=41,stratify = Labels)


# In[20]:


X_train


# In[21]:


Y_train.shape


# In[22]:


Y_train.value_counts()


# In[23]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[24]:


X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)


# In[25]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


# In[26]:



X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

X_train.shape, X_test.shape


# In[27]:


epochs = 45
model = Sequential()


# In[28]:
model.add(Conv1D(32,2,activation = 'relu',input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2)) # prevents over-fitting (randomly remove some neurons)
# SECOND LAYER
model.add(Conv1D(64,2,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Flattening the layer ( multidimentional data into vector)
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))

# FINAL LAYER
model.add(Dense(1,activation='sigmoid')) # binary classification 


# In[29]:
model.summary()


# In[30]:
model.compile(optimizer = Adam(lr=0.0001),loss = 'binary_crossentropy',metrics=['accuracy'])


# In[31]:
history = model.fit(X_train, Y_train, epochs = epochs,
                    validation_data = (X_test,Y_test),verbose = 1)


# In[32]:
def plot_learning_curve(history,epochs):
    
    # plot training and validation accuracy 
    epoch_range = range(1,epochs+1)
    plt.plot(epoch_range,history.history['accuracy'])
    plt.plot(epoch_range,history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()
        # plot training and validation loss
    plt.plot(epoch_range,history.history['loss'])
    plt.plot(epoch_range,history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()


# In[33]:
#plot_learning_curve(history,epochs)
# In[34]:
model = Sequential()


# In[35]:
# FIRST LAYER
model.add(Conv1D(64,2,activation = 'relu',input_shape = X_train[0].shape))
model.add(BatchNormalization())
'''Batch normalization is a technique for training very deep neural networks 
   that standardizes the inputs to a layer for each mini-batch. This 
   has the effect of stabilizing the learning process and dramatically
   reducing the number of training epochs required to train deep networks'''
model.add(MaxPool1D(2))
'''Max pooling is done to in part to help over-fitting by providing an abstracted form of the
   representation. As well, it reduces the computational cost by reducing the 
   number of parameters to learn and provides basic translation invariance to 
   the internal representation.'''
model.add(Dropout(0.2)) # prevents over-fitting (randomly remove some neurons)

# SECOND LAYER
model.add(Conv1D(128,2,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

# Flattening the layer ( multidimentional data into vector)
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))

# FINAL LAYER
model.add(Dense(1,activation='sigmoid')) # binary classification 


# In[36]:


model.summary()

#from geo import getTweetLocation


app = Flask(__name__)
app.secret_key = 'any random string'
PEOPLE_FOLDER = os.path.join('static', 'people_photo')
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('main.html')

   
def validate(username,password):
    con = sql.connect('static/chat.db')
    completion = False
    with con:
        cur = con.cursor()
        cur.execute('SELECT * FROM persons')
        rows = cur.fetchall()
        for row in rows:
            dbuser = row[1]
            dbpass = row[2]
            if dbuser == username:
                completion = (dbpass == password)
    return completion


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        completion = validate(username,password)
        if completion == False:
            error = 'invalid Credentials.please try again.'
        else:
            session['username'] = request.form['username']
            return render_template('main.html')
    return render_template('credit.html', error=error)



    
@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            username = request.form['username']
            password = request.form['password']
            with sql.connect("static/chat.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO persons(name,username,password) VALUES (?,?,?)",(name,username,password))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"
        finally:
            return render_template("main.html",msg = msg)
            con.close()
    return render_template('register.html')


@app.route('/first',methods = ['POST'])
def first():
	return render_template('credit.html')

#app=Flask(__name__)
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'txt'}
REGISTER_PATH="./Register/"
TEMP_PATH="./Temp"
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#model = joblib.load(open('./Model/ranfor_model.pkl','rb'))

@app.route('/register', methods=['GET', 'POST'])
def upload_register():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return json.dumps({"status": "Error", "msg": "Image cannot be empty "})
        name = request.form.get('name')
        email = request.form.get('email')
        
       
        if(name ==''):
            return json.dumps({"status": "Error", "msg": "Name cannot be empty "})

        file = request.files['file']
      
       
       
        print(file)
        
       

        if file.filename == '':
            return json.dumps({"status": "Error", "msg": "Image cannot be empty "})

        if file and allowed_file(file.filename):
        
            print(file)
            
        else:
            return json.dumps({"status": "Error", "msg": "Image Format not supported <png,jpg,jpeg> "})
prediction=0
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded

        return render_template('credit.html')

            
@app.route('/result', methods=['GET', 'POST'])
def result():
        
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            

        file = request.files['file']
        print(file)
        #file.save(file.filename)

        if file.filename == '':
            return redirect(request.url)
        #convertToBinaryData(uploaded_file.filename)
        #foo = pd.read_excel(file)
        if file and allowed_file(file.filename):
            
            print(file)
            #print(foo)
           
            #foo.to_excel(TEMP_PATH+"/"+file.filename.strip())#,optimize=True)#,quality=85)
            file.save(TEMP_PATH+"/"+file.filename.strip())
            print("############################",TEMP_PATH+"/"+file.filename.strip())
            g=(TEMP_PATH+"/"+file.filename.strip())
            df=pd.read_excel(g,header=None)
           
            #df = df.drop([0], axis=1)
            print("############################")
            print("############################")
            #print(df)
            #df = df.drop(0)
            prediction = model.predict(df)
            prediction=prediction.round()
            
            print(prediction)
            

            #return (tes_ft)
    return render_template("result.html",prediction=prediction)
#@app.route('/form',methods = ['POST'])
#def form():
    #return render_template('form.html')

#prediction function
#def ValuePredictor(to_predict_list):
    #to_predict = np.array(to_predict_list).reshape(1,13)
    #loaded_model = pickle.load(open("model/heart_model.pkl","rb"))
    #result = loaded_model.predict(to_predict)
    #return result[0]
   


#@app.route('/result',methods = ['POST'])
#def result():
    #if request.method == 'POST':
        #to_predict_list = request.form.to_dict()
        #to_predict_list=list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        #result = ValuePredictor(to_predict_list)
        
        #if int(result)==1:
            #prediction='You have been diagnosed Heart Disease'
        #else:
            #prediction='You have been diagnosed with no disease. Congratulations'
        #from keras import backend as K
        #K.clear_session()
            
        #return render_template("result.html",prediction=prediction)


if __name__ == '__main__':
   app.run(debug = True )
