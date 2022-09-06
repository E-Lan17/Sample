# Importing the libraries
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import os


# Start timer :
start_time = datetime.now()

# Open all csv files in the current directory, select the corect column/row 
# and add them to a list of dataframe df :
os.chdir("./")
list_df = []
for f, file in enumerate(glob.glob("*.csv")) :  
    list_df.append(pd.read_csv(file).iloc[11: ,1:4].values)
del(f,file)

# Iterate thrue the list of dataframe to clean them :
for dataframe in range(len(list_df)) :
    df = list_df[dataframe]            
    
# Create a list of activity from the dataframe df and iterate thrue it to   
# clean the activity :
    list_activity = df[:,2]
    #letters = str()
    for i in range(len(list_activity)) :
        list_activity[i] = list_activity[i].lower()
        list_activity[i] = list_activity[i].replace(" ", "")
        list_activity[i] = list_activity[i].replace("(", "")        
        list_activity[i] = list_activity[i].replace(")", "")  
        list_activity[i] = list_activity[i].replace(",", "")
        list_activity[i] = list_activity[i].replace('"', "")
        list_activity[i] = list_activity[i].replace('"', "")
    del(i)
# Replace activities in daframe df with the clean list_activity for all 
# the entries :
    df[:,2] = list_activity
    del(list_activity)
    
# Create a list of the time list_hour when a entry occure, split hour, minutes,
# secondes and convert evrything in seconds :
    list_hour = df[:, 0]
    for i in range(len(list_hour)) :          
        if ':' in list_hour[i] and "'" in list_hour[i] and '"' in list_hour[i] :
            list_hour[i] = list_hour[i].replace('"', "")
            list_hour[i] = str(list_hour[i]).split("'")
            s = int(list_hour[i][1])
            list_hour[i] = str(list_hour[i][0]).split(":")
            h = int(list_hour[i][0])
            m = int(list_hour[i][1])
            list_hour[i] = h*3600 + m*60 + s
            del(h,m,s)
            continue
        if "'"in list_hour[i] and '"' in list_hour[i] :
            list_hour[i] = list_hour[i].replace('"', "")
            list_hour[i] = str(list_hour[i]).split("'")
            h = 0
            m = int(list_hour[i][0])
            s = int(list_hour[i][1])
            list_hour[i] = h*3600 + m*60 + s
            del(h,m,s)
            continue 
        if '"' in list_hour[i] :
            list_hour[i] = list_hour[i].replace('"', "")
            h = 0
            m = 0
            s = int(list_hour[i][0])
            list_hour[i] = h*3600 + m*60 + s
            del(h,m,s)
            continue
    del(i)
    
# Take  list_hour  and replace it in df :
    df[:,0] = list_hour
  
# Iterate thrue list_hour and determine if its day (1) or night (0),
# add the value to a new list and delete the variable list_hour after :        
    list_day = []   
    for i in range(len(list_hour)) :
        if int(list_hour[i])<28800 or int(list_hour[i])>=64800 :
              list_day.append(1)
        else :         
              list_day.append(0) 
    del(i,list_hour) 
    
# Add a new column for days from list_day at the end of the dataframe df 
# and delete the variable list_day after :
    list_day = np.array(list_day)
    list_day = list_day.reshape((len(list_day), 1))
    df = np.append(df, list_day, axis=1)
    del(list_day)  
    
# delete first column of df (time in seconds) array and put back df in 
# dataframe panda (probably not the right way to do it)
    df = np.delete(df, 0, 1) 
    df = pd.DataFrame(df, index=None, columns=None)  

# Put back df in list_df. Dont know why but not all variables are updated 
# in the list_df otherwise (like column day) and clean df for an another iteration : 
    list_df[dataframe] = df
    del(df) 
    
# End of the loop thrue list_df
del(dataframe) 

# Create an empty list that will take all windows, decide the time windows :
list_activity = ""
list_activity_final = []
list_day = []
windows = 5
timer = 0
# Iterate thrue the list of dataframe df to get the time windows :
for dataframe in range(len(list_df)) :
    df = list_df[dataframe].iloc[: ,:].values
# Iterate thrue df to get the duration off the activity, add the duration off the 
# activity in the dictionnary with activities to the right activity. Add the dictionary
# in a list of dictionnary/windows. If the total duration of the activities in the 
# dictionnary is biger than the time windows, split the duration of the activity 
# and put the remainning duration in a new dictionary/windows :   
    
# Create a list of unique activities with a total duration the size of the windows
    for i in range(len(df)) :  
        duration = float(df[i,0])
            
        while (timer + duration) > windows :
            list_activity = list_activity + str(df[i,1])
            list_day.append(df[i,2])
            list_activity_final.append(list_activity)
            list_activity = ""
            duration = duration - (windows - timer)
            timer = 0

        if (timer + duration) < windows :
            list_activity = list_activity + str(df[i,1])
            timer = timer + duration
            continue
        
        if (timer + duration) == windows :     
            list_activity = list_activity + str(df[i,1])
            list_activity_final.append(list_activity)
            list_activity = ""
            timer = 0
            list_day.append(df[(i+1),2])
            continue
    
    df_f= pd.DataFrame()   
    df_f[0] = list_activity_final
    df_f[1] = list_day
    list_df[dataframe] = df_f       
        
del(i,timer,windows,dataframe,df,duration,list_activity)

# Create the clean dataframe for analysis with all df combined :
df_words = pd.DataFrame()
for i in list_df :
    df_words =df_words.append(i)
del(i)

# Create new csv cluster file in the Processed_data folder:
df_words.to_csv('./SPLIT_SEQ_SEQ_1000.csv',header = ["Activities","Day"])            
del(df_words) 
# End timer and display it :
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

