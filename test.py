from fbx import *
import FbxCommon
import sys
import numpy as np

FILENAME = "C:/Users/yzhang/Downloads/dog_run.fbx"
POINT_COUNT = 111
POINT_SAMPLE_GAP = 10
POINT_SAMPLE_COUNT = 12
POINT_VALUE_COUNT = 7
STYLE_COUNT = 6
BONE_VALUE_COUNT = 12

def AddAllActiveFbxNodes(fbxAnimLayer, fbxNode, fbxNodeList):
    AddFbxNode(fbxAnimLayer, fbxNode, fbxNodeList)
    for nodeCount in range(fbxNode.GetChildCount()):
        AddAllActiveFbxNodes(fbxAnimLayer, fbxNode.GetChild(nodeCount), fbxNodeList)

def AddFbxNode(fbxAnimLayer, fbxNode, fbxNodeList):
    translationXCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "X")    
    translationYCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "Y")
    translationZCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "Z")

    rotationXCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "X")
    rotationYCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "Y")
    rotationZCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "Z")

    if (translationXCurve is None) or (translationYCurve is None) or (translationZCurve is None) or (rotationXCurve is None) or (rotationYCurve is None) or (rotationZCurve is None):
        print(fbxNode.GetName() + " doesnt have active curve.")
        return
    
    fbxNodeList.append(fbxNode)

def GetRootVelocityAndSpeed(fbxAnimLayer, fbxNodeList):
    rootNode = fbxNodeList[0]

    rootTranslationXCurve = rootNode.LclTranslation.GetCurve(fbxAnimLayer, "X")
    rootTranslationYCurve = rootNode.LclTranslation.GetCurve(fbxAnimLayer, "Y")
    rootTranslationZCurve = rootNode.LclTranslation.GetCurve(fbxAnimLayer, "Z")
    
    rootVelocitySpeedData = np.zeros(rootTranslationXCurve.KeyGetCount(), 4)
    for i in range(rootVelocitySpeedData.shape):
        



def PrepareForData(fbxAnimLayer, fbxNodeList):
    frameCount = fbxNodeList[0].LclTranslation.GetCurve(fbxAnimLayer, "X").GetKeyCount()
    
    for i in range(frameCount):
        time = FbxTime.SetFrame(i)
        currentRootTransform = fbxNodeList[0].EvaluateGlobalTransform(time)

        
        valueCount = (POINT_SAMPLE_COUNT + STYLE_COUNT) * POINT_SAMPLE_COUNT + len(fbxNodeList) * BONE_VALUE_COUNT    
        valueArray = np.arange(valueCount)


if __name__ == "__main__":
    (fbxSdkManager, fbxScene) = FbxCommon.InitializeSdkObjects()
    result = FbxCommon.LoadScene(fbxSdkManager, fbxScene, FILENAME)    

    if not result:
        print("Failed to load fbx.")
        sys.exit(1)

    fbxRootNode = fbxScene.GetRootNode()
    if fbxRootNode is None:
        print("There is no root node in fbx.")
        sys.exit(1)

    if fbxScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId)) != 1:
        print("Only support one AnimaStack in one fbx.")
        sys.exit(1)

    fbxAnimStack = fbxScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), 0)
    
    if fbxAnimStack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId)) != 1:
        print("Only support one AnimLayer in one fbx.")
        sys.exit(1)

    fbxAnimLayer = fbxAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), 0)    
    fbxNodeList = []

    AddAllActiveFbxNodes(fbxAnimLayer, fbxRootNode, fbxNodeList)    

    fbxSdkManager.Destroy()
    sys.exit(0)
    
    

    
    





