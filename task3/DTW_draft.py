import numpy as np
def euclidean_distance(vector1, vector2):
    dist = (vector1 - vector2)**2
    dist = np.sum(dist)
    dist = np.sqrt(dist)

    return dist


def DTW(word1, word2):
    matrix = [ [ 0 for j in range(len(word2))  ] for i in range(len(word1)) ] 
    sakoe_chiba_ratio = 0.3

    lower_bound = round(len(word1)*sakoe_chiba_ratio)

    upper_bound = round(len(word2)*sakoe_chiba_ratio)
    
    for i in range(len(word1)):
        for j in range(len(word2)):
            distAdd = float("inf")
            distRmv = float("inf")
            distRplc = float("inf")
                
            if i > 0 :
               distAdd = matrix[i-1][j] + euclidean_distance(word1[i],word2[j])
            if j > 0 :
               distRmv = matrix[i][j-1] + euclidean_distance(word1[i],word2[j])
            if i > 0 and j > 0 :
                distRplc = matrix[i-1][j-1] + euclidean_distance(word1[i],word2[j])
            if i == 0 and j == 0 :
                matrix[i][j] = euclidean_distance(word1[i],word2[j])
                continue
            if i > lower_bound + j:
                matrix[i][j] = float("inf")
                continue
            if j > upper_bound + i:
                matrix[i][j] = float("inf")
                continue
            if distAdd < distRmv and distAdd < distRplc:
                matrix[i][j] = distAdd
            elif distRmv  < distRplc and distRmv < distAdd:
                matrix[i][j] = distRmv
            else:
                matrix[i][j] = distRplc
    
    for i in range(len(word1)):
        print(matrix[i])
    
    return matrix[len(word1) - 1][len(word2) -1]



word1 = np.array([[1],[2],[3],[4],[5]])
word2 = np.array([[45],[6],[17],[8],[5],[10],[25]])


print("The distance between {} and {} is {}".format(word1,word2, DTW(word1, word2) )) 
