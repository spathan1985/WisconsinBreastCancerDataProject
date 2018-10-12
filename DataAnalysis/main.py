# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:45:23 2017

@author: Shagufta
"""

#Assignment final project phase1
#This part of the code loads the Breast Cancer data into a pandas dataframe
#Imputes missing values in the dataframe with the mean of the column into the column A7
#Plots 9 histograms for all the 9 attributes 
#And finally computes mean, median, std. deviation and variance for each of these 9 attributes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests

def main():
    #Download the data from the url 
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    s = requests.get(url).content
    s = s.decode('utf-8')
    s_rows = s.split('\n')
    s_rows_cols = [each.split(',') for each in s_rows]
    header_row = ['Scn','A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'CLASS']
    c = pd.DataFrame(s_rows_cols, columns = header_row)
    c.to_csv('Breast_Cancer_Data.csv')

    df = pd.read_csv('Breast_Cancer_Data.csv', index_col=0)
    df['A7'].replace('?', np.nan, inplace=True) #replacing '?' in column A7 with NaN using df.replace() method
    df['A7'] = df['A7'].astype(float)   #convert all values of column A7 into float since NaN is of type float
    df = df.fillna(np.round(df.mean(),1)) #Using 'mean' imputation method to insert the missing values in df
    
    with PdfPages('HistogramResults.pdf') as pdf:
        #For column A2 - Clump Thickness
        clump_thickness = df['A2']
        fig1 = plt.figure(figsize=(9,6))
        plt.hist(clump_thickness, bins=9, color='blue', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,160)
        plt.title('Fig1: Clump thickness')
        plt.xlabel('thickness')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        clump_thickness_mean = np.round(clump_thickness.mean(),2)
        mean = s1 + str(clump_thickness_mean)
        plt.text(8.1,150, mean, fontsize=12)
        s2 = "Median: "
        clump_thickness_median = np.round(clump_thickness.median(),2)
        median = s2 + str(clump_thickness_median)
        plt.text(8.1,140,median,fontsize=12)
        s3 = "Std. Deviation: "
        clump_thickness_SD = np.round(clump_thickness.std(),2)
        sd = s3 + str(clump_thickness_SD)
        plt.text(8.1,130,sd,fontsize=12)
        s4 = "Variance: "
        clump_thickness_variance = np.round(clump_thickness.var(), 2)
        var = s4 + str(clump_thickness_variance)
        plt.text(8.1,120,var,fontsize=12)
        pdf.savefig(fig1)
        plt.show()
        plt.close()
        
         #For column A3 - Cell Size Uniformity
        cell_size_uniformity = df['A3']
        fig2 = plt.figure(figsize=(9,6))
        plt.hist(cell_size_uniformity, bins=9, color='yellow', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,400)
        plt.title('Fig2: Uniformity of Cell Size')
        plt.xlabel('Cell Size')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        cell_size_uniformity_mean = np.round(cell_size_uniformity.mean(),2)
        mean = s1 + str(cell_size_uniformity_mean)
        plt.text(8.1,380, mean, fontsize=12)
        s2 = "Median: "
        cell_size_uniformity_median = np.round(cell_size_uniformity.median(),2)
        median = s2 + str(cell_size_uniformity_median)
        plt.text(8.1,360,median,fontsize=12)
        s3 = "Std. Deviation: "
        cell_size_uniformity_SD = np.round(cell_size_uniformity.std(),2)
        sd = s3 + str(cell_size_uniformity_SD)
        plt.text(8.1,340,sd,fontsize=12)
        s4 = "Variance: "
        cell_size_uniformity_variance = np.round(cell_size_uniformity.var(), 2)
        var = s4 + str(cell_size_uniformity_variance)
        plt.text(8.1,320,var,fontsize=12)
        pdf.savefig(fig2)
        plt.show()
        plt.close()
        
        #For column A4 - Cell Shape Uniformity
        cell_shape_uniformity = df['A4']
        fig3 = plt.figure(figsize=(9,6))
        plt.hist(cell_shape_uniformity, bins=9, color='green', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,400)
        plt.title('Fig3: Uniformity of Cell Shape')
        plt.xlabel('Cell Shape')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        cell_shape_uniformity_mean = np.round(cell_shape_uniformity.mean(),2)
        mean = s1 + str(cell_shape_uniformity_mean)
        plt.text(8.1,380, mean, fontsize=12)
        s2 = "Median: "
        cell_shape_uniformity_median = np.round(cell_shape_uniformity.median(),2)
        median = s2 + str(cell_shape_uniformity_median)
        plt.text(8.1,360,median,fontsize=12)
        s3 = "Std. Deviation: "
        cell_shape_uniformity_SD = np.round(cell_shape_uniformity.std(),2)
        sd = s3 + str(cell_shape_uniformity_SD)
        plt.text(8.1,340,sd,fontsize=12)
        s4 = "Variance: "
        cell_shape_uniformity_variance = np.round(cell_shape_uniformity.var(), 2)
        var = s4 + str(cell_shape_uniformity_variance)
        plt.text(8.1,320,var,fontsize=12)
        pdf.savefig(fig3)
        plt.show()
        plt.close()
        
        #For column A5 - Marginal Adhesion
        marginal_adhesion = df['A5']
        fig4 = plt.figure(figsize=(9,6))
        plt.hist(marginal_adhesion, bins=9, color='red', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,450)
        plt.title('Fig4: Marginal Adhesion')
        plt.xlabel('Size of Marginal Adhesion')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        marginal_adhesion_mean = np.round(marginal_adhesion.mean(),2)
        mean = s1 + str(marginal_adhesion_mean)
        plt.text(8.1,420, mean, fontsize=12)
        s2 = "Median: "
        marginal_adhesion_median = np.round(marginal_adhesion.median(),2)
        median = s2 + str(marginal_adhesion_median)
        plt.text(8.1,400,median,fontsize=12)
        s3 = "Std. Deviation: "
        marginal_adhesion_SD = np.round(marginal_adhesion.std(),2)
        sd = s3 + str(marginal_adhesion_SD)
        plt.text(8.1,380,sd,fontsize=12)
        s4 = "Variance: "
        marginal_adhesion_variance = np.round(marginal_adhesion.var(), 2)
        var = s4 + str(marginal_adhesion_variance)
        plt.text(8.1,360,var,fontsize=12)
        pdf.savefig(fig4)
        plt.show()
        plt.close()
        
        #For column A6 - Single Epithelial Cell Size
        single_epi_cell_size = df['A6']
        fig5 = plt.figure(figsize=(9,6))
        plt.hist(single_epi_cell_size, bins=9, color='royalblue', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,400)
        plt.title('Fig5: Single Epithelial Cell Size')
        plt.xlabel('Size of Single Epithelial Cell')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        single_epi_cell_size_mean = np.round(single_epi_cell_size.mean(),2)
        mean = s1 + str(single_epi_cell_size_mean)
        plt.text(8.1,380, mean, fontsize=12)
        s2 = "Median: "
        single_epi_cell_size_median = np.round(single_epi_cell_size.median(),2)
        median = s2 + str(single_epi_cell_size_median)
        plt.text(8.1,360,median,fontsize=12)
        s3 = "Std. Deviation: "
        single_epi_cell_size_SD = np.round(single_epi_cell_size.std(),2)
        sd = s3 + str(single_epi_cell_size_SD)
        plt.text(8.1,340,sd,fontsize=12)
        s4 = "Variance: "
        single_epi_cell_size_variance = np.round(single_epi_cell_size.var(), 2)
        var = s4 + str(single_epi_cell_size_variance)
        plt.text(8.1,320,var,fontsize=12)
        pdf.savefig(fig5)
        plt.show()
        plt.close()
        
        #For column A7 - Bare Nuclei
        bare_nuclei = df['A7']
        fig6 = plt.figure(figsize=(9,6))
        plt.hist(bare_nuclei, bins=9, color='pink', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,450)
        plt.title('Fig6: Bare Nuclei')
        plt.xlabel('Size of Bare Nuclei')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        bare_nuclei_mean = np.round(bare_nuclei.mean(),2)
        mean = s1 + str(bare_nuclei_mean)
        plt.text(8.1,420, mean, fontsize=12)
        s2 = "Median: "
        bare_nuclei_median = np.round(bare_nuclei.median(),2)
        median = s2 + str(bare_nuclei_median)
        plt.text(8.1,400,median,fontsize=12)
        s3 = "Std. Deviation: "
        bare_nuclei_SD = np.round(bare_nuclei.std(),2)
        sd = s3 + str(bare_nuclei_SD)
        plt.text(8.1,380,sd,fontsize=12)
        s4 = "Variance: "
        bare_nuclei_variance = np.round(bare_nuclei.var(), 2)
        var = s4 + str(bare_nuclei_variance)
        plt.text(8.1,360,var,fontsize=12)
        pdf.savefig(fig6)
        plt.show()
        plt.close()
        
        #For column A8 - Bland Chromatin
        bland_chromatin = df['A8']
        fig7 = plt.figure(figsize=(9,6))
        plt.hist(bland_chromatin, bins=9, color='crimson', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,180)
        plt.title('Fig7: Bland Chromatin')
        plt.xlabel('Bland Chromatin')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        bland_chromatin_mean = np.round(bland_chromatin.mean(),2)
        mean = s1 + str(bland_chromatin_mean)
        plt.text(8.1,170, mean, fontsize=12)
        s2 = "Median: "
        bland_chromatin_median = np.round(bland_chromatin.median(),2)
        median = s2 + str(bland_chromatin_median)
        plt.text(8.1,160,median,fontsize=12)
        s3 = "Std. Deviation: "
        bland_chromatin_SD = np.round(bland_chromatin.std(),2)
        sd = s3 + str(bland_chromatin_SD)
        plt.text(8.1,150,sd,fontsize=12)
        s4 = "Variance: "
        bland_chromatin_variance = np.round(bland_chromatin.var(), 2)
        var = s4 + str(bland_chromatin_variance)
        plt.text(8.1,140,var,fontsize=12)
        pdf.savefig(fig7)
        plt.show()
        plt.close()
        
        #For column A9 - Normal Nuclei
        normal_nucleoli = df['A9']
        fig8 = plt.figure(figsize=(9,6))
        plt.hist(normal_nucleoli, bins=9, color='burlywood', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,450)
        plt.title('Fig8: Normal Nucleoli')
        plt.xlabel('Normal Nucleoli')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        normal_nucleoli_mean = np.round(normal_nucleoli.mean(),2)
        mean = s1 + str(normal_nucleoli_mean)
        plt.text(8.1,420, mean, fontsize=12)
        s2 = "Median: "
        normal_nucleoli_median = np.round(normal_nucleoli.median(),2)
        median = s2 + str(normal_nucleoli_median)
        plt.text(8.1,400,median,fontsize=12)
        s3 = "Std. Deviation: "
        normal_nucleoli_SD = np.round(normal_nucleoli.std(),2)
        sd = s3 + str(normal_nucleoli_SD)
        plt.text(8.1,380,sd,fontsize=12)
        s4 = "Variance: "
        normal_nucleoli_variance = np.round(normal_nucleoli.var(), 2)
        var = s4 + str(normal_nucleoli_variance)
        plt.text(8.1,360,var,fontsize=12)
        pdf.savefig(fig8)
        plt.show()
        plt.close()
        
        #For column A10 - Mitosis
        mitosis = df['A10']
        fig9 = plt.figure(figsize=(9,6))
        plt.hist(mitosis, bins=9, color='purple', alpha=0.5)
        plt.xticks(range(12))
        plt.xlim(0,11)
        plt.ylim(0,600)
        plt.title('Fig9: Mitosis')
        plt.xlabel('Mitosis')
        plt.ylabel('No. of people')
        s1 = "Mean: "
        mitosis_mean = np.round(mitosis.mean(),2)
        mean = s1 + str(mitosis_mean)
        plt.text(8.1,560, mean, fontsize=12)
        s2 = "Median: "
        mitosis_median = np.round(mitosis.median(),2)
        median = s2 + str(mitosis_median)
        plt.text(8.1,530,median,fontsize=12)
        s3 = "Std. Deviation: "
        mitosis_SD = np.round(mitosis.std(),2)
        sd = s3 + str(mitosis_SD)
        plt.text(8.1,500,sd,fontsize=12)
        s4 = "Variance: "
        mitosis_variance = np.round(mitosis.var(), 2)
        var = s4 + str(mitosis_variance)
        plt.text(8.1,470,var,fontsize=12)
        pdf.savefig(fig9)
        plt.show()
        plt.close()
        
main()