# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:44:33 2017

@author: Shagufta
"""

# Module 12: Assignment Final Project Phase 3

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculateEuclDistance(p1,p2):
    ''' This function calculates the distance between 2 points.
    retVal: calculated distance'''
    
    total_Squared = 0
    for i in range(0,9):
        total_Squared = total_Squared + ((p1[i]-p2[i]) ** 2)
    return math.sqrt(total_Squared)
        
def impute_missing_value(df):
    ''' This function replaces the special char with NaN , computes the mean
    and imputes missing value into the dataframe.
    retVal: non NaN df'''
    
    df['A7'].replace('?', np.nan, inplace=True) #replacing '?' in column A7 with NaN using df.replace() method
    df['A7'] = df['A7'].astype(float)   #convert all values of column A7 into float since NaN is of type float
    df = df.fillna(np.round(df.mean(),1)) #using mean() to impute the NaN values
    return df
 
def initialize_mean(df, k):
    '''This function initializes 2 means from 699 datapoints.
    retVal: u2 and u4 as the new means.'''
    
    u2 = []  #initialize the means as lists
    u4 = []

    df_new = df.sample(k)  #pick random datapoints from df
    u2 = df_new.iloc[0]['A2':'A10'] #assign the 9-dim vector to u2 and u4 
    u4 = df_new.iloc[1]['A2':'A10']  #df_new.values also cud work
    return u2,u4
    
def assignment(df, u2, u4):
    '''This function calculate the distance between u2 and all datapoints 
    and u4 with all datapoints and assign it to cluster 2 and 4 respectively.
    retVal: clustered df'''
    
    for i in range(len(df)):
        u = df.iloc[i]['A2':'A10']
        dist_u2 = calculateEuclDistance(list(u),list(u2))
        dist_u4 = calculateEuclDistance(list(u),list(u4))
        if dist_u2 < dist_u4:
            df.loc[i, 'Predicted_Class'] = 2  #create a new column Predicted_Class
        else:
            df.loc[i, 'Predicted_Class'] = 4
    return df
           
def recalculation(df):
    '''This function recalculates new means from the clustered df.
    retVal: updated u2 and u4'''
    
    u2 = df.loc[df['Predicted_Class'] == 2, 'A2':'A10'].mean()
    u4 = df.loc[df['Predicted_Class'] == 4, 'A2':'A10'].mean()
    return u2, u4
        
def main():
    
    ######################### Phase 1 ################################
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    colNames = ['Scn', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                'A10', 'CLASS']
    raw_data = pd.read_csv(url, names=colNames) #download the data from the url
    
    k = 2  #centroid value
    df = impute_missing_value(raw_data) #replace missing values with mean
    
    with PdfPages('C:\\Users\\Shagufta\\Downloads\\MS Application Docs\\Indiana University\\Summer 2017 - Sem3\\Python\\Module12\\spathan\\HistogramResults.pdf') as pdf:
        #this module PdfPages writes the  plots to a pdf file mentioned in the path
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
       
 ############################# Phase 2 ######################################
    u2, u4 = initialize_mean(df,k)  #initialize new means u2, u4
    
    for iteration in range(1500):  
        data = assignment(df, u2, u4)  #assign datapoints to cluster 2 and 4
        u2, u4 = recalculation(data)   #recalculate new means based on assignment step
            
    print('-'*30,'Final Mean','-'*30) #print final means after 1500 iterations
    print('u2: ', list(u2))
    print('\n')
    print('u4: ', list(u4))
    
    print('\n')
    print('-'*30,'Cluster Assignment','-'*30) 
    
    df1 = df[['Scn', 'CLASS', 'Predicted_Class']] #only choosing 3 columns to print
    df1 = df1.rename(columns={'Scn':'ID', 'CLASS': 'Actual_Class'})   #rename Scn to ID
    print(df1.loc[:20].astype(int))  #typecasting all values to int
    #print(df1.loc[:].astype(int))
  
 ############################# Phase 3 ##################################
    print('\n')
    print('-'*30,'Error Rate','-'*30)
    
    erroru2 = 0
    erroru4 = 0
    totalB = 0
    totalM = 0
    
    totalB = df1.loc[df1['Actual_Class']==2, 'Actual_Class'].count() #calculate total error rate for cluster2 of actual class
    totalM = df1.loc[df1['Actual_Class']==4, 'Actual_Class'].count() #calculate total error rate for cluster4 of actual class
    
    for i in range(len(df1)):
        if ((df1.loc[i, 'Predicted_Class']==4) and (df1.loc[i,'Actual_Class']==2)):
            erroru2 += 1
        elif ((df1.loc[i, 'Predicted_Class']==2) and (df1.loc[i,'Actual_Class']==4)):
            erroru4 += 1
            
    errorB = erroru2 / totalB #calculate error for cluster2
    errorM = erroru4 / totalM #calculate error for cluster4
    
    print('Error rate for class 2:', errorB)
    print('Error rate for class 4:', errorM)
    totalerror = errorB + errorM
    print('Total Error of both classes:', totalerror)
        
if __name__ == '__main__':
    main()