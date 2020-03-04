from fbx import *
import FbxCommon
import sys
import numpy as np

FILENAME = "C:/Users/yzhang/Downloads/dog_run.fbx"
HIP_NAME = "Hips"
ROOT_NAME = "root"

POINT_COUNT = 111
POINT_ROOT_INDEX = 60
POINT_SAMPLE_GAP = 10
POINT_SAMPLE_COUNT = 12
POINT_VALUE_COUNT = 7       # positionX/Z, diretionX/Z, velocityX/Z, speed

STATE_COUNT = 6             # idle...
BONE_VALUE_COUNT = 12       # positionX/Y/Z, forwardX/Y/Z, upX/Y/Z, velocityX/Y/Z

def FindRootNode(fbxRootNode):
    for nodeIndex in range(fbxRootNode.GetChildCount()):
        if fbxRootNode.GetChild(nodeIndex).GetName() == ROOT_NAME:            
            return fbxRootNode.GetChild(nodeIndex)
        
        FindRootNode(fbxRootNode)
                
    return None

def AddAllActiveFbxNodes(fbxAnimLayer, fbxNode, fbxNodeList):
    if fbxNode.GetName() != ROOT_NAME:
        if AddFbxNode(fbxAnimLayer, fbxNode, fbxNodeList) == False:
            return False
    
    for nodeIndex in range(fbxNode.GetChildCount()):
        AddAllActiveFbxNodes(fbxAnimLayer, fbxNode.GetChild(nodeIndex), fbxNodeList)

    return fbxNodeList.count != 0    

def AddFbxNode(fbxAnimLayer, fbxNode, fbxNodeList):
    translationXCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "X")    
    translationYCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "Y")
    translationZCurve = fbxNode.LclTranslation.GetCurve(fbxAnimLayer, "Z")

    rotationXCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "X")
    rotationYCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "Y")
    rotationZCurve = fbxNode.LclRotation.GetCurve(fbxAnimLayer, "Z")

    if (rotationXCurve is None) or (rotationYCurve is None) or (rotationZCurve is None):
        print(fbxNode.GetName() + " doesnt have active rotation curve.")
        return False
    
    if fbxNode.GetName() == HIP_NAME and ((translationXCurve is None) or (translationYCurve is None) or (translationZCurve is None)):
        print(HIP_NAME + " doesnt have active translation curve.")
        return False

    fbxNodeList.append(fbxNode)    
    return True   

def UpdateDefaultSampleValue(defaultSampleValue, targetFrameIndex, deltaTime, fbxNode, rootPointFbxMatrixInverse):
    if targetFrameIndex < 1:
        print("FrameIndex should be more than 1.")
        return False    

    targetFrameTime = FbxTime(targetFrameIndex)    
    targetFrameLocalTransform = fbxNode.EvaluateGlobalTransform(targetFrameTime) * rootPointFbxMatrixInverse

    targetFrameTime.SetFrame(targetFrameIndex - 1)    
    previousTargetFrameLocalTransform = fbxNode.EvaluateGlobalTransform(targetFrameTime) * rootPointFbxMatrixInverse
    
    localTranslation = targetFrameLocalTransform.GetT()
    defaultSampleValue[0] = localTranslation[0] # position x
    defaultSampleValue[1] = localTranslation[2] # position z

    localDirection = targetFrameLocalTransform.GetRow(2)
    defaultSampleValue[2] = localDirection[0] # direction x
    defaultSampleValue[3] = localDirection[2] # direction z

    previousLocalTranslation = previousTargetFrameLocalTransform.GetT()
    velocity = (localTranslation - previousLocalTranslation) / deltaTime
    defaultSampleValue[4] = velocity[0]
    defaultSampleValue[5] = velocity[2]

    speed = np.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
    defaultSampleValue[6] = speed    

    # to do, how to set the animation state value
    return True

def PrepareTrainingPointSampleData(trainingData, deltaTime, hipNode, fbxNodeList):        
    if hipNode.GetName() != HIP_NAME:
        print("Please use hipnode to prepare training point sample data.")
        return False
    
    frameCount = trainingData.shape[0]

    # start from frame 1 instead of frame 0 to get the speed and velocity
    frameTime = FbxTime()
    rootPointFbxMatrixInverse = FbxMatrix()       
    defaultSampleValue = np.zeros(POINT_VALUE_COUNT + STATE_COUNT)
    for frameIndex in range(1, frameCount + 1):                                
        # sample point root
        frameTime.SetFrame(frameIndex)
        rootPointFbxMatrixInverse = hipNode.EvaluateGlobalTransform(frameTime).Inverse()
        if UpdateDefaultSampleValue(defaultSampleValue, frameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
            return False

        rootSampleValue = np.copy(defaultSampleValue)
        rootSampleIndex = POINT_ROOT_INDEX // POINT_SAMPLE_GAP
        trainingData[frameIndex - 1, rootSampleIndex * (POINT_VALUE_COUNT + STATE_COUNT) : (rootSampleIndex + 1) * (POINT_VALUE_COUNT + STATE_COUNT)] = rootSampleValue

        # sample point 0 ~ (root - 1)        
        for sampleIndex in range(1, rootSampleIndex + 1):
            sampleFrameIndex = frameIndex - sampleIndex * POINT_SAMPLE_GAP
            if sampleFrameIndex > 0 :                
                if UpdateDefaultSampleValue(defaultSampleValue, sampleFrameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
                    return False
            
            trainingData[frameIndex - 1, sampleIndex * (POINT_VALUE_COUNT + STATE_COUNT) : (sampleIndex + 1) * (POINT_VALUE_COUNT + STATE_COUNT)] = defaultSampleValue
            
        # sample point (root + 1) ~ (POINT_SAMPLE_COUNT - 1)
        for sampleIndex in range(rootSampleIndex + 1, POINT_SAMPLE_COUNT):
            sampleFrameIndex = frameIndex + sampleIndex * POINT_SAMPLE_GAP
            if sampleFrameIndex < frameCount:
                if UpdateDefaultSampleValue(defaultSampleValue, sampleFrameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
                    return False

            trainingData[frameIndex - 1, sampleIndex * (POINT_VALUE_COUNT + STATE_COUNT) : (sampleIndex + 1) * (POINT_VALUE_COUNT + STATE_COUNT)] = rootSampleValue

        print("Preparing Training Point Sample Data: {0}/{1}".format(frameIndex, frameCount))

    print("Preparing Training Point Sample Data Complete!")
    return True

def PrepareTrainingBoneData(trainingData, deltaTime, hipNode, fbxNodeList):
    if hipNode.GetName() != HIP_NAME:
        print("Please use hipnode to prepare training bone data.")
        return False
    
    frameCount = trainingData.shape[0]

    frameTime = FbxTime()
    previousFrameTime = FbxTime()
    boneData = np.zeros(BONE_VALUE_COUNT)    
    boneVelocity = FbxVector4()
    boneTransform = FbxMatrix()
    previousBoneTransform = FbxMatrix()
    previousRootInverseTransform = FbxMatrix()
    dataIndexOffset = (POINT_VALUE_COUNT + STATE_COUNT) * POINT_SAMPLE_COUNT
    for frameIndex in range(1, frameCount + 1):        
        previousFrameTime.SetFrame(frameIndex - 1)
        previousRootInverseTransform = hipNode.EvaluateGlobalTransform(previousFrameTime).Inverse()

        frameTime.SetFrame(frameIndex)
        for boneIndex in range(len(fbxNodeList)):
            boneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(frameTime)
            
            boneData[0] = boneTransform.GetT()[0]       # position x
            boneData[1] = boneTransform.GetT()[1]       # position y
            boneData[2] = boneTransform.GetT()[2]       # position z
            
            boneData[3] = boneTransform.GetRow(2)[0]    # forward x
            boneData[4] = boneTransform.GetRow(2)[1]    # forward y
            boneData[5] = boneTransform.GetRow(2)[2]    # forward z

            boneData[6] = boneTransform.GetRow(1)[0]    # up x
            boneData[7] = boneTransform.GetRow(1)[1]    # up y
            boneData[8] = boneTransform.GetRow(1)[2]    # up z

            previousBoneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(previousFrameTime)
            boneVelocity = (boneTransform.GetT() - previousBoneTransform.GetT()) / deltaTime
            boneData[9] = boneVelocity[0]               # velocity x
            boneData[10] = boneVelocity[1]              # velocity y
            boneData[11] = boneVelocity[2]              # velocity z

            trainingData[frameIndex - 1, dataIndexOffset + boneIndex * BONE_VALUE_COUNT : dataIndexOffset + (boneIndex + 1) * BONE_VALUE_COUNT] = boneData

        print("Preparing Training Bone Data: {0}/{1}".format(frameIndex, frameCount))

    print("Preparing Training Bone Data Complete!")
    return True

if __name__ == "__main__":
    (fbxSdkManager, fbxScene) = FbxCommon.InitializeSdkObjects()
    result = FbxCommon.LoadScene(fbxSdkManager, fbxScene, FILENAME)    

    if not result:
        print("Failed to load fbx.")
        sys.exit(1)

    fbxRootNode = FindRootNode(fbxScene.GetRootNode())
    if fbxRootNode is None:
        print("There is no node in fbx named as {0}.".format(ROOT_NAME))
        sys.exit(1)
    
    if fbxScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId)) != 1:
        print("Only support one AnimStack in one fbx.")
        sys.exit(1)

    fbxAnimStack = fbxScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), 0)    
    if fbxAnimStack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId)) != 1:
        print("Only support one AnimLayer in one fbx.")
        sys.exit(1)

    fbxAnimLayer = fbxAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), 0)    
    fbxNodeList = []
    if AddAllActiveFbxNodes(fbxAnimLayer, fbxRootNode, fbxNodeList) == False:
        print("Failed to add all fbx node.")
        sys.exit(1)

    hipTranslationXCurve = fbxNodeList[0].LclTranslation.GetCurve(fbxAnimLayer, "X")
    frameCount = hipTranslationXCurve.KeyGetCount()
    boneCount = len(fbxNodeList)
    dataCount = POINT_SAMPLE_COUNT * (POINT_VALUE_COUNT + STATE_COUNT) + boneCount * BONE_VALUE_COUNT
    if frameCount < 2 or dataCount == 0:
        print("Frame count or data count is wrong.")
        sys.exit(1)

    trainingData = np.zeros((frameCount - 2, dataCount))
    frameRate = FbxTime.GetFrameRate(fbxScene.GetGlobalSettings().GetTimeMode()) 
    if PrepareTrainingPointSampleData(trainingData, 1 / frameRate, fbxNodeList[0], fbxNodeList) == False:
        print("Failed to prepare training point sample data.")
        sys.exit(1)

    if PrepareTrainingBoneData(trainingData, 1 / frameRate, fbxNodeList[0], fbxNodeList) == False:
        print("Failed to prepare training bone data.")
        sys.exit(1)

    

    fbxSdkManager.Destroy()
    sys.exit(0)
    
    

    
    





