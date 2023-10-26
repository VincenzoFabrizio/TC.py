import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import textract
import os
import spacy
import pandas as pd


# STEP 1: Pull student bios
flist=[]
for filename in os.listdir('C:/Users/Vincenzo Fabrizio/Documents/development/Team_Clustering/Data/Faculty_Bios'):  # Go through folder of bios in Word files
    ftext=textract.process(f'C:/Users/Vincenzo Fabrizio/Documents/development/Team_Clustering/Data/Faculty_Bios/{filename}')  # Extract bios
    fname = f'{filename}'  # Save the name of the document, which is the last name of the individual
    fname = fname.removesuffix(".docx")  # Remove .docx so only last name is kept
    flist.append(ftext)  # Add the bio texts to a list for processing

# STEP 2: Identify and quantify student characteristics
#  Using BOW
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(flist)  # Transforms list of documents into a matrix, where each row is an individual
                                         # bio and each column is the frequency of each word used in the bio scaled by
                                         # its overall frequency across all documents

bow_matrix=tfidf.toarray()
scores = np.sum(bow_matrix, axis=1)


#print(vectorizer.get_feature_names_out(), '\n')
#for doc_vector in tfidf.toarray():
#    print(doc_vector)

# STEP 3: Assign students to clusters
# Sample data
students = list(scores)
sarray = np.array(students)


kmeans = KMeans(
   init="random",
   n_clusters=5,
   n_init=10,  # How many times clustering is done (selects that with lowest error)
   max_iter=300

)
kmeans.fit(sarray.reshape(-1, 1))  # Clustering is done and result is assigned to kmeans.labels_
clusters = kmeans.labels_

# STEP 4: Assign students to groups
# Attach index to cluster assignment to identify students
id = [i for i in range(1, len(np.asarray(clusters)) + 1)]
clusters_id = np.insert(clusters, 0, id)
student_clusters = clusters_id.reshape(2, len(id))

# Initialize list of tuples 'sc'
sc = [(student_clusters[0][0], student_clusters[1][0])]
# Add id-cluster pairs to list
for i in range(1, len(student_clusters[0])):
    sc.append((student_clusters[0][i], student_clusters[1][i]))

# Separate clusters into individual lists
c0 = list(filter(lambda x: x[1] == 0, sc))
c1 = list(filter(lambda x: x[1] == 1, sc))
c2 = list(filter(lambda x: x[1] == 2, sc))
c3 = list(filter(lambda x: x[1] == 3, sc))
c4 = list(filter(lambda x: x[1] == 4, sc))

# Randomly pick a student from each cluster and place them in a group together.
# Repeat until at least 1 cluster has no more students
clist = [c0, c1, c2, c3, c4]

all_groups = []
while max(len(c0), len(c1), len(c2), len(c3), len(c4)) > 0:  # "While there exists at least 1 unassigned student"
    this_group = []
    for cluster in clist:  # "For each of the clusters"
        if len(cluster) == 0:  # If cluster is empty, go to next cluster
            continue
        cnum = cluster[0][1]  # Identify which cluster is active
        rand = random.randint(0, len(cluster) - 1)  # RNG for which student will be assigned within given cluster
        this_group.append(cluster[rand][0])
        if cnum == 0:   # When a student is selected, they are removed from the list
            c0.remove(c0[rand])
        elif cnum == 1:
            c1.remove(c1[rand])
        elif cnum == 2:
            c2.remove(c2[rand])
        elif cnum == 3:
            c3.remove(c3[rand])
        elif cnum == 4:
            c4.remove(c4[rand])

    all_groups.append(this_group)  # After all clusters have been done, add this group to master list

print(all_groups)
print(clist)

# STEP 5: Reallocate groups for balance
# Groups must have a maximum of 5, minimum of 4

# If group has <4...
for group in all_groups:
    if len(group) > 3:
        continue

    # ...then remove members of this group...

    # ...and allocate to existing groups of <5
    while len(group) != 0:
        transfer = group[0]

        for regroup in all_groups:
            if len(regroup) <5:
                regroup.append(transfer)
                break
        group.remove(transfer)

print(all_groups)
# If there exists only 1 incomplete group among groups of 5...
if len(all_groups[len(all_groups)-1]) < 4 and len(all_groups[len(all_groups)-2])==5:
    lone_group = all_groups[len(all_groups)-1]

# ...take 1 from each group of 5 until incomplete group reaches 4
    gnum = 0
    while len(lone_group) < 4:
        transfer = all_groups[gnum][0]
        lone_group.append(transfer)
        all_groups[gnum].remove(transfer)
        gnum = gnum+1
print(all_groups)

# STEP 6: Replace numbers with student names

