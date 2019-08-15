"""
Created on Mon Jun 24 14:27:49 2019

@author: Pedro Salamoni
"""

#Parameters
data_path = 'data2.csv'
model_path = 'Models/model_20190731_025103_991.yaml'
weight_path = 'Models/tt.h5'
labels = ['Alvara','Habite-se','Verso Alvara','Outros']
test_rate = 1
n_predict_data = 1000


################################################################################################################################

def freememo():
    import gc
    
    gc.collect()
    
def importdata(data_path):
    import pandas as pd
    
    my_data = pd.read_csv('data.csv', sep=',',header=None)
    path = my_data[0]
    print("\n Metadata Imported \n")
    
    return path

def preprocessdata(path):
    import numpy as np

    path = np.array(path)
    
    print("\n Metadata Processed \n")
    
    return path

def getdata(paths,n_data,res):
    from PIL import Image
    import numpy as np
    
    if n_data!=None:
        if len(res)==0:
            if n_data>len(paths):
                n_data = len(paths)
        else:
            if n_data>min(len(paths),len(res)):
                n_data = min(len(paths),len(res))

    im = Image.open(paths[0])
    im = im.resize((205,270))
    x = np.array(im)[None,:,:,:]

    for i,path in enumerate(paths[1:n_data]):
        print(i,path)
        im = Image.open(path)
        im = im.resize((205,270))
        new = np.array(im)[None,:,:,:]
        x = np.append(x,new, axis=0)
        
    x = x / 255.0
    if len(res)==0:
        print("\n Data Imported \n")
        return x
    else:
        y = res[:n_data]
        print("\n Data Imported \n")
        return x,y

def createai(weight_path):
    from tensorflow import keras
    
    model = keras.models.load_model(weight_path)
    print("Loaded model from disk")
    
    return model
    
def predictai(model,x_predict):
    
    result = model.predict(x_predict)
    
    return result

def printerrors(n_data,result,x_predict,y_predict,labels):
    import matplotlib.pyplot as plt
    import numpy as np
    
    for i in range(n_data):
        classe = np.argmax(result[i])
        if classe!=int(y_predict[i]):
            plt.imshow(x_predict[i])
            plt.show()
            print('Predicted:',labels[classe],'('+str(result[i][classe])+') | Trully:',labels[int(y_predict[i])],'('+str(result[i][int(y_predict[i])])+')')
    
def main(model):
    
    
    paths = importdata(data_path)
    
    path_train,path_test,res_train,res_test = preprocessdata(paths,res,test_rate)
    
    x_predict,y_predict = getdata(path_train,n_predict_data,res_train)
    
    result = predictai(model,x_predict)
    
    return x_predict,y_predict,result

model = createai(weight_path)

x_predict,y_predict,result = main(model)

printerrors(n_predict_data,result,x_predict,y_predict,labels)


<<<<<<< Updated upstream
    my_data = pd.read_csv(data_path, sep=',',header=None)
    path, res = my_data[0], my_data[1]
=======
    my_data = pd.read_csv('data.csv', sep=',',header=None)
    path = my_data[0]
>>>>>>> Stashed changes