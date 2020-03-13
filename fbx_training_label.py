# install python fbx sdk
# http://download.autodesk.com/us/fbx/20112/FBX_SDK_HELP/index.html?url=WS73099cc142f48755-751de9951262947c01c-6dc7.htm,topicNumber=d0e8430

# python fbx sdk guide
# https://help.autodesk.com/view/FBX/2020/ENU/

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

INPUT_STATE_COUNT = 6             # idle...
INPUT_POINT_VALUE_COUNT = 7       # positionX/Z, diretionX/Z, velocityX/Z, speed`
INPUT_BONE_VALUE_COUNT = 12       # positionX/Y/Z, forwardX/Y/Z, upX/Y/Z, velocityX/Y/Z

OUTPUT_POINT_VALUE_COUNT = 6      # positionX/Z, diretionX/Z, velocityX/Z
OUTPUT_BONE_VALUE_COUNT = 12      # positionX/Y/Z, forwardX/Y/Z, upX/Y/Z, velocityX/Y/Z

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

def UpdateDefaultTrainingSampleValue(defaultSampleValue, targetFrameIndex, deltaTime, fbxNode, rootPointFbxMatrixInverse):
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
    defaultSampleValue = np.zeros(INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT)
    for frameIndex in range(1, frameCount + 1):                                
        # sample point root
        frameTime.SetFrame(frameIndex)
        rootPointFbxMatrixInverse = hipNode.EvaluateGlobalTransform(frameTime).Inverse()
        if UpdateDefaultTrainingSampleValue(defaultSampleValue, frameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
            return False

        rootSampleValue = np.copy(defaultSampleValue)
        rootSampleIndex = POINT_ROOT_INDEX // POINT_SAMPLE_GAP
        trainingData[frameIndex - 1, rootSampleIndex * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT) : (rootSampleIndex + 1) * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT)] = rootSampleValue

        # sample point 0 ~ (root - 1)        
        for sampleIndex in range(1, rootSampleIndex + 1):
            sampleFrameIndex = frameIndex - sampleIndex * POINT_SAMPLE_GAP
            if sampleFrameIndex > 0 :                
                if UpdateDefaultTrainingSampleValue(defaultSampleValue, sampleFrameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
                    return False
            
            trainingData[frameIndex - 1, sampleIndex * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT) : (sampleIndex + 1) * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT)] = defaultSampleValue
            
        # sample point (root + 1) ~ (POINT_SAMPLE_COUNT - 1)
        for sampleIndex in range(rootSampleIndex + 1, POINT_SAMPLE_COUNT):
            sampleFrameIndex = frameIndex + sampleIndex * POINT_SAMPLE_GAP
            if sampleFrameIndex < frameCount:
                if UpdateDefaultTrainingSampleValue(defaultSampleValue, sampleFrameIndex, deltaTime, hipNode, rootPointFbxMatrixInverse) == False:
                    return False

            trainingData[frameIndex - 1, sampleIndex * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT) : (sampleIndex + 1) * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT)] = rootSampleValue

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
    boneData = np.zeros(INPUT_BONE_VALUE_COUNT)    
    boneVelocity = FbxVector4()
    localBoneTransform = FbxMatrix()    
    previousLocalBoneTransform = FbxMatrix()
    rootInverseTransform = FbxMatrix()
    dataIndexOffset = (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT) * POINT_SAMPLE_COUNT
    for frameIndex in range(1, frameCount + 1):        
        frameTime.SetFrame(frameIndex)
        previousFrameTime.SetFrame(frameIndex - 1)
        rootInverseTransform = hipNode.EvaluateGlobalTransform(frameTime).Inverse()
        
        for boneIndex in range(len(fbxNodeList)):
            localBoneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(frameTime) * rootInverseTransform
            
            boneData[0] = localBoneTransform.GetT()[0]       # position x
            boneData[1] = localBoneTransform.GetT()[1]       # position y
            boneData[2] = localBoneTransform.GetT()[2]       # position z
            
            boneData[3] = localBoneTransform.GetRow(2)[0]    # forward x
            boneData[4] = localBoneTransform.GetRow(2)[1]    # forward y
            boneData[5] = localBoneTransform.GetRow(2)[2]    # forward z

            boneData[6] = localBoneTransform.GetRow(1)[0]    # up x
            boneData[7] = localBoneTransform.GetRow(1)[1]    # up y
            boneData[8] = localBoneTransform.GetRow(1)[2]    # up z
            
            previousLocalBoneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(previousFrameTime) * rootInverseTransform
            boneVelocity = (localBoneTransform.GetT() - previousLocalBoneTransform.GetT()) / deltaTime
            boneData[9] = boneVelocity[0]               # velocity x
            boneData[10] = boneVelocity[1]              # velocity y
            boneData[11] = boneVelocity[2]              # velocity z

            trainingData[frameIndex - 1, dataIndexOffset + boneIndex * INPUT_BONE_VALUE_COUNT : dataIndexOffset + (boneIndex + 1) * INPUT_BONE_VALUE_COUNT] = boneData

        print("Preparing Training Bone Data: {0}/{1}".format(frameIndex, frameCount))

    print("Preparing Training Bone Data Complete!")
    return True

def PrepareLabelPointData(labelData, deltaTime, hipNode):
    frameCount = labelData.shape[0]

    pointData = np.zeros(OUTPUT_POINT_VALUE_COUNT)
    frameTime = FbxTime()
    previousFrameTime = FbxTime()
    localSampleTransform = FbxMatrix()
    rootInverseTransform = FbxMatrix()
    for frameIndex in range(2, frameCount + 2):
        frameTime.SetFrame(frameIndex)
        rootInverseTransform = hipNode.EvaluateGlobalTransform(frameTime).Inverse()
        for sampleIndex in range(POINT_SAMPLE_COUNT - POINT_ROOT_INDEX // POINT_SAMPLE_COUNT):
            targetFrameIndex = frameIndex + sampleIndex * POINT_SAMPLE_GAP
            if targetFrameIndex < frameCount + 2: 
                frameTime.SetFrame(targetFrameIndex)
                localSampleTransform = hipNode.EvaluateGlobalTransform(frameTime) * rootInverseTransform

                pointData[0] = localSampleTransform.GetT()[0]
                pointData[1] = localSampleTransform.GetT()[2]

                pointData[2] = localSampleTransform.GetRow(2)[0]
                pointData[3] = localSampleTransform.GetRow(2)[2]

                previousFrameTime.SetFrame(targetFrameIndex - 1)
                previouslocalSampleTransform = hipNode.EvaluateGlobalTransform(previousFrameTime) * rootInverseTransform
                velocity = (localSampleTransform.GetT() - previouslocalSampleTransform.GetT()) / deltaTime
                pointData[4] = velocity[0]
                pointData[5] = velocity[2]
            
            labelData[frameIndex - 2, sampleIndex * OUTPUT_POINT_VALUE_COUNT : (sampleIndex + 1) * OUTPUT_POINT_VALUE_COUNT] = pointData

        print("Preparing Label Point Data: {0}/{1}".format(frameIndex - 1, frameCount))

    print("Preparing Label Point Data Complete!")

def PrepareLabelBoneData(labelData, deltaTime, hipNode, fbxNodeList):
    frameCount = labelData.shape[0]
    dataIndexOffset = OUTPUT_POINT_VALUE_COUNT * (POINT_SAMPLE_COUNT - POINT_ROOT_INDEX // POINT_SAMPLE_COUNT)

    boneData = np.zeros(OUTPUT_BONE_VALUE_COUNT)
    frameTime = FbxTime()
    previousFrameTime = FbxTime()
    rootInverseTransform = FbxMatrix()
    localBoneTransform = FbxMatrix()
    previousLocalBoneTransform = FbxMatrix()
    for frameIndex in range(2, frameCount + 2):
        frameTime.SetFrame(frameIndex)
        previousFrameTime.SetFrame(frameIndex - 1)
        rootInverseTransform = hipNode.EvaluateGlobalTransform(frameTime).Inverse()
        for boneIndex in range(len(fbxNodeList)):
            localBoneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(frameTime) * rootInverseTransform

            boneData[0] = localBoneTransform.GetT()[0]       # position x
            boneData[1] = localBoneTransform.GetT()[1]       # position y
            boneData[2] = localBoneTransform.GetT()[2]       # position z
            
            boneData[3] = localBoneTransform.GetRow(2)[0]    # forward x
            boneData[4] = localBoneTransform.GetRow(2)[1]    # forward y
            boneData[5] = localBoneTransform.GetRow(2)[2]    # forward z

            boneData[6] = localBoneTransform.GetRow(1)[0]    # up x
            boneData[7] = localBoneTransform.GetRow(1)[1]    # up y
            boneData[8] = localBoneTransform.GetRow(1)[2]    # up z

            previousLocalBoneTransform = fbxNodeList[boneIndex].EvaluateGlobalTransform(previousFrameTime) * rootInverseTransform
            boneVelocity = (localBoneTransform.GetT() - previousLocalBoneTransform.GetT()) / deltaTime
            boneData[9] = boneVelocity[0]               # velocity x
            boneData[10] = boneVelocity[1]              # velocity y
            boneData[11] = boneVelocity[2]              # velocity z

            labelData[frameIndex - 2, dataIndexOffset + boneIndex * OUTPUT_BONE_VALUE_COUNT : dataIndexOffset + (boneIndex + 1) * OUTPUT_BONE_VALUE_COUNT] = boneData

        print("Preparing Label Bone Data: {0}/{1}".format(frameIndex - 1, frameCount))

    print("Preparing Label Bone Data Complete!")

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
    trainingDataCount = POINT_SAMPLE_COUNT * (INPUT_POINT_VALUE_COUNT + INPUT_STATE_COUNT) + boneCount * INPUT_BONE_VALUE_COUNT
    if frameCount < 2 or dataCount == 0:
        print("Frame count or data count is wrong.")
        sys.exit(1)

    trainingData = np.zeros((frameCount - 2, trainingDataCount))
    frameRate = FbxTime.GetFrameRate(fbxScene.GetGlobalSettings().GetTimeMode()) 
    if PrepareTrainingPointSampleData(trainingData, 1 / frameRate, fbxNodeList[0], fbxNodeList) == False:
        print("Failed to prepare training point sample data.")
        sys.exit(1)

    if PrepareTrainingBoneData(trainingData, 1 / frameRate, fbxNodeList[0], fbxNodeList) == False:
        print("Failed to prepare training bone data.")
        sys.exit(1)

    labelDataCount = OUTPUT_POINT_VALUE_COUNT * (POINT_SAMPLE_COUNT - POINT_ROOT_INDEX // POINT_SAMPLE_COUNT) + OUTPUT_BONE_VALUE_COUNT * boneCount
    labelData = np.zeros((frameCount - 2, labelDataCount))
    PrepareLabelPointData(labelData, 1/ frameRate, fbxNodeList[0])
    PrepareLabelBoneData(labelData, 1/ frameRate, fbxNodeList[0], fbxNodeList)

    fbxSdkManager.Destroy()
    sys.exit(0)