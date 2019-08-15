"""
Created on Mon Jun 24 14:27:49 2019

@author: Pedro Salamoni
"""

#Parameters
data_pdf_path = '/home/setup/Documents/DevBarn/DocClassAI/Data_Minning/tt.csv'
data_jpg_path = '/home/setup/Documents/DevBarn/DocClassAI/data.csv'
model_path = 'Models/tt.h5'
labels = ['Alvara','Habite-se','Verso Alvara','Outros']
n_data = 1000


################################################################################################################################

def pdftojpg(data_path):
    import os
    from pdf2image import convert_from_path
    import pandas as pd
    
    my_data = pd.read_csv(data_path, sep=',',header=None)
    paths = my_data[0]
    
    print('Converting pdfs', end ="")
    
    for i,path in enumerate(paths):
        slash_position = path.rfind('/')
        folder = '.temp/' + path[slash_position:path.rfind('.')] + '/'
        os.path.normpath(folder)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            pages = convert_from_path(path, 100)
            for n, page in enumerate(pages):
                save_img_path = folder + '/' + str(i) + '_' + str(n) + '.jpg'
                os.path.normpath(save_img_path)
                page.save(save_img_path, 'JPEG')
                print('.', end ="")
    print('.')

def imglookup():
	import glob
	files = glob.glob('.temp/**/*.jpg', recursive=True)
	files.sort()
	return files

def freememo():
    import gc
    
    gc.collect()
    
def importdata(jpg_path):
    import pandas as pd
    
    my_data = pd.read_csv(jpg_path, sep=',',header=None)
    path = my_data[0]
    print("\n Metadata Imported \n")
    
    return path

def preprocessdata(path):
    import numpy as np

    path = np.array(path)
    
    print("\n Metadata Processed \n")
    
    return path

def getdata(paths):
    from PIL import Image
    import numpy as np
    
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
    print("\n Data Imported \n")
    return x

def createai(model_path):
    from tensorflow import keras
    
    model = keras.models.load_model(model_path)
    print("Loaded model from disk")
    
    return model
    
def predictai(model,path,n_data,data_pdf_path):
    import numpy as np
    import shutil
    from pdf2image import convert_from_path
    import os

    page_outros = [-1]

    for i in range(int(len(path)/n_data)+(len(path)%n_data>0)):

        init = i*n_data
        if (len(path)<init+n_data):
            end = len(path)
        else:
            end = init+n_data

        x_predict = getdata(path[init:end])
        result_part = model.predict(x_predict)
        for k in range(len(result_part)):
            classe = np.argmax(result_part[k])
            id_path = int(path[init+k][path[init+k].rfind('/')+1:path[init+k].rfind('_')])
            page = int(path[init+k][path[init+k].rfind('_')+1:path[init+k].rfind('.')])

            if classe == len(labels)-1:
                if page_outros[0]==id_path:
                    page_outros.append(id_path)
                elif page_outros[0]!=-1:
                    #pages = convert_from_path(data_pdf_path[page_outros[0]], 100)
                    #save_img_path = 'Result/' + labels[classe] + '/' + path[init+k-1][path[init+k-1].rfind('/').rfind('/')+1:path[init+k-1].rfind('/')] + '_' + str(page) + '.pdf'
            else:
                pages = convert_from_path(data_pdf_path[id_path], 100)
                save_img_path = 'Result/' + labels[classe] + '/' + path[init+k][path[init+k].rfind('/').rfind('/')+1:path[init+k].rfind('/')] + '_' + str(page) + '.pdf'
                os.path.normpath(save_img_path)
                pages[page].save(save_img_path, 'PDF')
            print('.', end="")
        print('.')
        teste = np.column_stack((path[init:end],result_part))
        if i==0:
            result = teste
        else:
            result = np.append(result,teste, axis=0)
        del x_predict
        freememo()
    
    return result

def createfolders():
    import os
    
    if not os.path.exists('Result'):
        os.mkdir('Result')
    for i in range(len(labels)):
        if not os.path.exists('Result/' + labels[i]):
            os.makedirs('Result/' + labels[i])
    
def main(model):
    
    pdftojpg(data_pdf_path)
    
    jpg_path = imglookup()
    
    #paths = importdata(data_jpg_path)
    
    path = preprocessdata(jpg_path)
    
    createfolders()
    
    result = predictai(model,path,n_data,data_pdf_path)
    
    return result

model = createai(model_path)

result = main(model)

