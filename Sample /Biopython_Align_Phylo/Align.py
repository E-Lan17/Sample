# Importing the libraries
from Bio import SeqIO
from Bio import AlignIO
from Bio import Phylo
from Bio.Align.Applications import MuscleCommandline 
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
import matplotlib.pyplot as plt # need for better fig but not use 

#Read gbk data :
unfiltered = SeqIO.parse("./SARS-CoV-2.gbk", "genbank")

#Drop data without full length sequences and store them in a list
full_length_records = []
for record in unfiltered:
    if len(record.seq) > 29000:
        full_length_records.append(record)
               
#Put the list of complete seq in a FASTA file :      
SeqIO.write(full_length_records, "./SARS-CoV-2.fasta", "fasta")

#Align sequences with MUSCLE (change parameters those one are for fast test)
muscle_cline = MuscleCommandline(input="SARS-CoV-2.fasta", 
                                 out="SARS-CoV-2_aligned.fasta", 
                                 diags = True, 
                                 maxiters = 1, 
                                 log="./align_log.txt")
muscle_cline()

#Open aligned FASTA sequence :
with open("SARS-CoV-2_aligned.fasta","r") as aln :
    alignement = AlignIO.read(aln,"fasta") #parse plutot non ? au lieu de read
print(type(alignement))

#Initiate the distance calculator with identity model :
calculator = DistanceCalculator("identity")

#create the distance matrix en print it (optional ?) :
#distance_matrix = calculator.get_distance(alignement)
#print(distance_matrix)

#Initiate the tree construction from the distance calculator
constructor = DistanceTreeConstructor(calculator)

#Show the tree :
SARS_CoV_2_tree = constructor.build_tree(alignement)
SARS_CoV_2_tree.rooted = True
print(SARS_CoV_2_tree)

# and save it :
Phylo.write(SARS_CoV_2_tree,"SARS-CoV-2_tree.xml","phyloxml")

#open the tree (optional or when restart) :
with open("SARS-CoV-2_tree.xml","r") as tree :
    SARS_CoV_2_tree = Phylo.read(tree, "phyloxml")
#fig = Phylo.draw(SARS_CoV_2_tree)

#Better looking tree with matplotlib :
#fig = plt.figure(figsize=(13,5),dpi=100) #create figure and set size and resolution
#plt.rc("font",size=8) # change font size of node and leafs label
#plt.rc("xtick",labelsize=10) # change font size x axis of the tick
#plt.rc("ytick",labelsize=10) # change font size y axis of the tick
#axes = fig.add_subplot(1,1,1)
#Phylo.draw(SARS_CoV_2_tree, axes=axes)

Phylo.convert("SARS-CoV-2_tree.xml", "phyloxml", "SARS-CoV-2_tree.nex", "nexus")
SARS_CoV_2_tree = Phylo.read("SARS-CoV-2_tree.nex", "nexus")

#Better looking tree with matplotlib :
fig = plt.figure(figsize=(13,5),dpi=100) #create figure and set size and resolution
plt.rc("font",size=7) # change font size of node and leafs label
plt.rc("xtick",labelsize=10) # change font size x axis of the tick
plt.rc("ytick",labelsize=10) # change font size y axis of the tick
axes = fig.add_subplot(1,1,1)
Phylo.draw(SARS_CoV_2_tree, axes=axes)

