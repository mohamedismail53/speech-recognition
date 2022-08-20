import sys
import os
from os import path, listdir
import random
from PyQt5.QtGui import QPalette, QColor, QPixmap, QImage, QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
    QPushButton, QVBoxLayout, QFileDialog, QProgressBar, QListWidget,
    QLabel, QMenuBar, QMenu, QListWidgetItem)
import qdarkstyle
from playsound import playsound
from PIL import Image
from PIL.ImageQt import ImageQt
from os import walk
import numpy as np
import librosa
from dtw import *
import matplotlib.pyplot as plt
from scipy.spatial import distance


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() 
        self.resize(600, 700)
        self.setWindowTitle("Words Checker")
        self.setWindowIcon(QIcon('logo.png'))
        self.label1 = QLabel("Click select to select a test speaker and start.")     
        self.btn1 = QPushButton("Select")    
        self.btn2 = QPushButton("Reset")
        self.btn2.setEnabled(False)
        self.progressBar = QProgressBar()
        self.progressBar.setProperty("value", 0)
        self.listWidget = QListWidget()
        default_image = Image.new("RGB", (400, 400), (68, 83, 101))
        imageqt = ImageQt(default_image)
        self.pixmap = QPixmap.fromImage(imageqt)
        self.pixmap = self.pixmap.scaled(600, 280)
        self.label2 = QLabel()
        self.label2.setPixmap(self.pixmap)

        layout = QVBoxLayout()	
        layout.addWidget(self.label1) 		   
        layout.addWidget(self.btn1)       
        layout.addWidget(self.btn2)      
        layout.addWidget(self.progressBar)
        layout.addWidget(self.listWidget)
        layout.addWidget(self.label2)
        widget = QWidget()                        
        widget.setLayout(layout)                
        self.setCentralWidget(widget)

        self.btn1.clicked.connect(self.select)
        self.btn2.clicked.connect(self.reset)
        self.show()

    def dist_calculator(self,word_dist , ref_dist):
            word_np=np.array(word_dist)
            alignment = dtw(word_dist.T, ref_dist.T,keep_internals=True)
            wq=warp(alignment,index_reference=False)
            warped= word_dist.T[wq]
            print(warped.shape,ref_dist.T.shape)
            ret=[]
            for n in range(len(warped)):
                ret.append(distance.euclidean(warped[n],ref_dist.T[n]))
            return ret

    def select(self):
        directory_path = QFileDialog.getExistingDirectory(self)         
        directory_name = path.basename(directory_path)

        file_pathes = [path.join(os.getcwd(), file_path) 
            for file_path in listdir(directory_path)]
        file_names = [file_path
            for file_path in listdir(directory_path)]
        pairs_words = [(file_name[10:12], file_name[13]) for file_name in file_names]

        self.label1.setText(f'The test speaker is {directory_name}, click Reset to try again.')   
        self.btn1.setEnabled(False)
        self.btn2.setEnabled(True)
        


        ###############
        # this area is for the implementation code
        #Loading the 123 Words to be Checked
        self.words=[]
        self.Reference=[]
        for i in range(123):
            print(directory_path+'/'+file_names[i])
            record, samplerate = librosa.load(directory_path+'/'+file_names[i],sr=16000,mono=False)
            record_mfcc = librosa.feature.mfcc(y=record, sr=samplerate,n_mfcc=39)
            self.words.append(record_mfcc)
        #Loading Threshold array

        Threshold=np.load(os.getcwd()+'/Thresholds.npy')   
        print(Threshold.shape)

        #Loading Reference_Data array
        counter = 0
        for person in range(11):
            temp=[]
            for word in range(123):
                loaded_word=np.load(os.getcwd()+'/References_Data/'+str(counter)+'.npy')  
                temp.append(loaded_word)
                counter = counter+1
            self.Reference.append(temp)    
        print(self.Reference[0][0].shape) 
        print(self.words[0].shape)    
        #Checking Closest Reference

        def closest_reference_byword(el_word):
            distance_i=[]
            word_1=el_word
            for i in range(11):
                try:
                    ref_i=self.Reference[i][-1]
                    distance_i.append(dtw(word_1.T, ref_i.T,keep_internals=True,step_pattern=asymmetric).distance)
                except:
                    pass
            return np.argmin(np.array(distance_i))

        self.closest_reference_index=closest_reference_byword(self.words[122])  

        def Right_or_Wrong(Word_array):
           distances=[] 
           for x in range(122):
                word_np=np.array(Word_array[x])
                distance_w = dtw(word_np.T, self.Reference[self.closest_reference_index][x].T,keep_internals=True).distance 
                if distance_w <= Threshold[x][0]:
                    distances.append(1)
                else:
                    distances.append(0)  
           return distances

        Right_Wrong_array=Right_or_Wrong(self.words)    

        distances_word_4=self.dist_calculator(self.words[3],self.Reference[self.closest_reference_index][3])
        print(distances_word_4) 

        # here you have access on those variables:
        # directory_path: string for the directory path that have the 123 segmentied .wav files
        # directory_name: string for the directory name that have the 123 segmentied .wav files
        # file_pathes:    list of strings of the pathes for the 123 segmentied .wav files
        # file_pathes:    list of strings of the names for the 123 segmentied .wav files
        # 
        # you must do the reference and test comparison here and produce a list
        # (named: self.correct) of 123 elements. each element is either a one or
        # a zero indicating whether the word was correct or wrong
        #
        # comment the below line and write your own code
        self.correct = Right_Wrong_array
        ###############

        self.progressBar.setProperty("value", 100)

        for pair_word, c in zip(pairs_words, self.correct):
            if c:
                #item_text = f"Pair {pair_word[0]} word {pair_word[1]}\t\t\tCorrect       \U00002705"
                item_text = f"Pair {pair_word[0]} word {pair_word[1]}\t\t\t\U00002705 Correct"  
            else:
                #item_text = f"Pair {pair_word[0]} word {pair_word[1]}\t\t\tNot correct \U0000274C"
                item_text = f"Pair {pair_word[0]} word {pair_word[1]}\t\t\t\U0000274C Not correct"

            self.listWidget.addItem(QListWidgetItem(item_text))

        self.listWidget.itemClicked.connect(self.item_clicked)

    def item_clicked(self, item):
        print(item.text())
        correct_or_not = item.text()[19]
        if correct_or_not == 'C':   # the word is correct, display a white image
            default_image = Image.new("RGB", (400, 400), (68, 83, 101))
            imageqt = ImageQt(default_image)
        else:   # the word is not correct, display the distance image 
            pair = item.text()[5:7]
            word = item.text()[13]
            print (pair)
            print(word)
            # function that takes pair and word and return image to display
            distance_image = self.getting_distance_image(pair, word)
            imageqt = ImageQt(distance_image) 
 
        self.pixmap = QPixmap.fromImage(imageqt)
        self.pixmap = self.pixmap.scaled(600, 280)
        self.label2.setPixmap(self.pixmap)

    def reset(self):   
        self.close()
        self.__init__()

    def getting_distance_image(self, pair, word):  
        ###############
        # to write by the implementation team
        # this function takes the pair and word of the wrong word
        # and return the distance image as a PIL Image object
        #
        # comment the below line and write your own
        def fig2img(fig):
            import io
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img

        self.list_to_be_poltted = self.dist_calculator(self.words[2*int(pair)-(int(word)%2)-1],self.Reference[self.closest_reference_index][2*int(pair)-(int(word)%2)-1])
        image = Image.new("RGB", (400, 400))
        #image.putdata()
        plt.close()
        plt.plot(self.list_to_be_poltted)
        fig=plt.gcf()
        image=fig2img(fig)
        return image
        ###############

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    sys.exit(app.exec_())
