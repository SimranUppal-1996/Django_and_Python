#importing SequenceMatcher class from difflib module
from difflib import SequenceMatcher

#importing Path class from pathlib module
from pathlib import Path

#importing os for running commands and re for regular expressions
import os
import re

#creating three empty lists to store all the files as elements in the first,
#their names in the second, and content without new lines in the third
List_Files = []
FN_list=[]
new_list = []

#creating function to open and read files
def OpenRead(n):
    n = open(n , encoding = 'utf-8', errors = 'ignore')
    n = n.read()
    return(n)

#this function removes all the escape sequences from a list of files
def RemEscSeq(L_F):
    for x in L_F:
        filter = ''.join([chr(i) for i in range(1, 32)])
        x = x.translate(str.maketrans('', '', filter))
        new_list.append(x)
    return(new_list)

#this function finds out similarity among files, two at a time
def sim(FC,NL,FN):
    for j in range(len(list(FC.values()))):
        for k in range(len(list(FC.values()))):
            if(j != k):
                s = SequenceMatcher(None, NL[j], NL[k])
                sim = s.ratio()
                print(f"For files {FN[j]} and {FN[k]}")
                print(f"The contents are {sim*100}% common.")
                print("\n")

#function that allows users to upload text files whose plagiarism is to be detected
def Upload():

#searching for all the files with .txt extension in the mentioned path
    FolOfFiles = Path('P:\PlagiDetect').rglob('*.txt')

#for all the paths in above, splitting the file name to read it and
#append its contents to the first list, file names in the second
#list, content without escape sequences in the third, and zipping file names and the content together as
#key value pairs in the dictionary
    for file in FolOfFiles:
        Str_F = str(file)
        F_Name = Str_F[15:]
        if all(F_Name!=h for h in ('drive.txt','r1.txt','r2.txt','r3.txt')):
            Con_File = OpenRead(F_Name)
            List_Files.append(Con_File)
            FN_list.append(F_Name)
            new_list = RemEscSeq(List_Files)
            FC_dict = dict(zip(FN_list,new_list))

#calculating the ratio of matches among sequences by comparing files and
#printing similarity percentage only if the file names are not same
    sim(FC_dict,new_list,FN_list)

#function that allows users to check plagiarism for all the text files in their local computer
def AllLocalFiles():
#to store all the drives in a text file and extract drive names
    os.system('cmd /C " wmic logicaldisk get name > drive.txt"')
    drive = OpenRead('drive.txt')
    drive = str(drive)
    drive = re.findall('[a-zA-Z]',drive)
    drive = drive[4:]

#to open all the drives and all the text files in them, and to create a dictionary of those files with
#their file names as keys
    l1 = []
    for i in drive:
        i = i + ":"
        os.chdir(i)
        if(i != drive[-1]): 
            os.system('cmd /C "dir *.txt > r1.txt"')
        else:
            os.system('cmd /k "dir *.txt > r1.txt"')
        r1 = OpenRead('r1.txt')
        r1 = str(r1)
        l1 += re.findall('([a-zA-Z0-9]+.txt)',r1)
        h = ['drive.txt','r1.txt','r2.txt','r3.txt']
        l1 = list(set(l1) - set(h))
        for y in l1:
            y = OpenRead(y)
            List_Files.append(y)
            new_list = RemEscSeq(List_Files)
            FC_dict = dict(zip(l1,new_list))

#calculating the ratio of matches among sequences by comparing files and
#printing similarity percentage only if the file names are not same
    sim(FC_dict,new_list,l1)


#function to offer a choice to the user to select a particular drive to look for files and check plagiarism
def ChoiceForDrive():
    print("Enter the drive's name(eg - 'C:'):")
    d = input()
    os.chdir(d)
    os.system('cmd /C "dir *.txt > r2.txt"')
    r2 = OpenRead('r2.txt')
    r2 = str(r2)
    r2 = re.findall('([a-zA-Z0-9]+.txt)',r2)
    h = ['drive.txt','r1.txt','r2.txt','r3.txt']
    r2 = list(set(r2) - set(h))
    for y in r2:
        y = OpenRead(y)
        List_Files.append(y)
        new_list = RemEscSeq(List_Files)
        FC_dict = dict(zip(r2,new_list))

#calculating the ratio of matches among sequences by comparing files and
#printing similarity percentage only if the file names are not same
    sim(FC_dict,new_list,r2)


#function to offer a choice to the user to select a particular folder in any drive to look for files and check plagiarism
def ChoiceForFol():
    print("Enter the drive and the folder in that drive(eg - C:\folder_name):")
    f = input()
    os.chdir(f)
    os.system('cmd /C "dir *.txt > r3.txt"')
    r3 = OpenRead('r3.txt')
    r3 = str(r3)
    r3 = re.findall('([a-zA-Z0-9]+.txt)',r3)
    h = ['drive.txt','r1.txt','r2.txt','r3.txt']
    r3 = list(set(r3) - set(h))
    for y in r3:
        y = OpenRead(y)
        List_Files.append(y)
        new_list = RemEscSeq(List_Files)
        FC_dict = dict(zip(r3,new_list))

#calculating the ratio of matches among sequences by comparing files and
#printing similarity percentage only if the file names are not same
    sim(FC_dict,new_list,r3)


#the user will have four options; one to upload files, second to check plagiarism for all the text files in the computer,
#third to do the same with all the text files in a particular drive, and fourth to do it with the text files in a particular
#folder
print("Enter your choice(0, 1, 2, or 3):")
c = input()
if(c=='0'):
    Upload()
if(c=='1'):
    AllLocalFiles()
if(c=='2'):
    ChoiceForDrive()
if(c=='3'):
    ChoiceForFol()
