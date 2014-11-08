from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData,buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt
 
def average(list):
    return sum(list)/float(len(list))
  
def stDeviation(list):
    mean = average(list)
    diffSq = [pow((val-mean),2) for val in list]
    return sqrt(sum(diffSq)/len(list))
  
penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)
  
carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)
 
pokerData = buildExamplesFromExtraData()
def testPokerData(hiddenLayers = [16]):
    return buildNeuralNet(pokerData,maxItr = 200,hiddenLayerList =  hiddenLayers)


penDataList = []
#carDataList = []
for i in range(5):
    a = testPenData()
    #b = testCarData()
    penDataList.append(a[1])
    #carDataList.append(b[1])
aver1 = average(penDataList)
#aver2 = average(carDataList)
max1 = max(penDataList)
#max2 = max(carDataList)
stddev1 = stDeviation(penDataList)
#stddev2 = stDeviation(carDataList)
print(aver1,max1,stddev1)   # average, max, stdeviation of Pen Data
#print(aver2,max2,stddev2)   # average, max, stdeviation of Car Data
# #    
# # #Question 6 Data
# # listMAS = []
# # for j in range(0,45,5): #for car data iterations
# #     penDataList = []
# #     carDataList = []
# #     for i in range(5):
# #         b = testCarData([j])
# #         carDataList.append(b[1])
# #     aver2 = average(carDataList)
# #     max2 = max(carDataList)
# #     stddev2 = stDeviation(carDataList)
# #     v = (aver2,max2,stddev2)
# #     print v
# #     listMAS.append((aver2,max2,stddev2))
# #     listMAS.append((0,0,0))
# #   
# # list2 = []   
# # for j in range(0,45,5): #for pen data iterations
# #     penDataList = []
# #     carDataList = []
# #     for i in range(5):
# #         a = testPenData([j])
# #         penDataList.append(a[1])
# #     aver1 = average(penDataList)
# #     max1 = max(penDataList)
# #     stddev1 = stDeviation(penDataList)
# #     u = (aver1,max1,stddev1)
# #     print u
# #     list2.append((aver1,max1,stddev1))
# # print list2
# 
# Question 7:
# examples = ([([0,0],[0]),([0,1],[1]),([1,1],[0]),([1,0],[1])] , [([0,0],[0]),([1,1],[0]),([1,0],[1])])
# count = 0
# accuracy = buildNeuralNet(examples,maxItr = 300,hiddenLayerList = [count])[1]
# while(accuracy<0.99):
#     count+=1
#     accuracy = buildNeuralNet(examples,maxItr = 300,hiddenLayerList = [count])[1]
#     print(accuracy)
# print(count)    #count at which accuracy reaches 1.0
 
# #Question 8:
# 
# pokerDataList = []
# for i in range(5):
#     a = testPokerData()
#     pokerDataList.append(a[1])
# aver1 = average(pokerDataList)
# max1 = max(pokerDataList)
# stddev1 = stDeviation(pokerDataList)
# print(aver1,max1,stddev1)   # average, max, stdeviation of Poker Data

