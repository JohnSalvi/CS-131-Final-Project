import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from template_matcher import computeIntersectionOverUnion, runTemplateMatcherWithMultiScale


def loadLabeledDataset(labelsJsonPath, screenshotsDirectoryPath):
    labelsByImageName = json.loads(Path(labelsJsonPath).read_text())
    datasetSamples = []
    for imageName, groundTruthBoxes in labelsByImageName.items():
        imagePath = Path(screenshotsDirectoryPath) / imageName
        bgrImage = cv2.imread(str(imagePath))
        rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
        datasetSamples.append({"name": imageName, "image": rgbImage, "ground_truth": groundTruthBoxes})
    return datasetSamples


def buildSyntheticDataset(templatesDirectoryPath, sampleCount, canvasWidth=1280, canvasHeight=800, randomSeed=131):
    randomGenerator = random.Random(randomSeed)
    templateImagePaths = sorted(Path(templatesDirectoryPath).glob("*.png"))
    datasetSamples = []

    for sampleIndex in range(sampleCount):
        canvasRgbImage = np.full((canvasHeight, canvasWidth, 3), 245, dtype=np.uint8)
        groundTruthBoxes = []
        chosenTemplatePaths = randomGenerator.sample(templateImagePaths, k=min(4, len(templateImagePaths)))

        for templateImagePath in chosenTemplatePaths:
            templateGrayImage = cv2.imread(str(templateImagePath), cv2.IMREAD_GRAYSCALE)
            templateHeight, templateWidth = templateGrayImage.shape[:2]
            if templateWidth >= canvasWidth or templateHeight >= canvasHeight:
                continue

            placementSucceeded = False
            for _attempt in range(40):
                positionX = randomGenerator.randint(0, canvasWidth - templateWidth)
                positionY = randomGenerator.randint(0, canvasHeight - templateHeight)
                candidateBox = {"x": positionX, "y": positionY, "width": templateWidth, "height": templateHeight}
                overlaps = any(computeIntersectionOverUnion(candidateBox, existingBox) > 0.0 for existingBox in groundTruthBoxes)
                if not overlaps:
                    placementSucceeded = True
                    break

            if not placementSucceeded:
                continue

            templateRgbPatch = cv2.cvtColor(templateGrayImage, cv2.COLOR_GRAY2RGB)
            canvasRgbImage[positionY:positionY + templateHeight, positionX:positionX + templateWidth] = templateRgbPatch
            groundTruthBoxes.append({
                "template_name": templateImagePath.stem,
                "x": positionX,
                "y": positionY,
                "width": templateWidth,
                "height": templateHeight,
            })

        datasetSamples.append({"name": f"synthetic_{sampleIndex:02d}.png", "image": canvasRgbImage, "ground_truth": groundTruthBoxes})

    return datasetSamples


def matchPredictionsToGroundTruth(predictedBoxes, groundTruthBoxes, iouThresholdValue):
    sortedPredictions = sorted(predictedBoxes, key=lambda boxRow: boxRow["confidence_score"], reverse=True)
    usedGroundTruthIndices = set()
    matchedPairs = []
    falsePositiveCount = 0

    for predictedBox in sortedPredictions:
        bestIouValue = 0.0
        bestGroundTruthIndex = -1
        for groundTruthIndex, groundTruthBox in enumerate(groundTruthBoxes):
            if groundTruthIndex in usedGroundTruthIndices:
                continue
            iouValue = computeIntersectionOverUnion(predictedBox, groundTruthBox)
            if iouValue > bestIouValue:
                bestIouValue = iouValue
                bestGroundTruthIndex = groundTruthIndex

        if bestGroundTruthIndex >= 0 and bestIouValue >= iouThresholdValue:
            usedGroundTruthIndices.add(bestGroundTruthIndex)
            matchedPairs.append({"prediction": predictedBox, "ground_truth": groundTruthBoxes[bestGroundTruthIndex], "iou": bestIouValue})
        else:
            falsePositiveCount += 1

    falseNegativeCount = len(groundTruthBoxes) - len(matchedPairs)
    return matchedPairs, falsePositiveCount, falseNegativeCount


def isClickPointInsideBox(predictedBox, groundTruthBox):
    insideX = groundTruthBox["x"] <= predictedBox["center_x"] <= groundTruthBox["x"] + groundTruthBox["width"]
    insideY = groundTruthBox["y"] <= predictedBox["center_y"] <= groundTruthBox["y"] + groundTruthBox["height"]
    return insideX and insideY


def runEvaluation(datasetSamples, matchThresholdValue, iouThresholdValue, outputDirectoryPath):
    Path(outputDirectoryPath).mkdir(parents=True, exist_ok=True)
    totalTruePositives = 0
    totalFalsePositives = 0
    totalFalseNegatives = 0
    matchedIouValues = []
    clickInsideCount = 0

    for datasetSample in datasetSamples:
        _summaryText, _annotatedPath, predictedBoxes = runTemplateMatcherWithMultiScale(
            screenshotRgbImage=datasetSample["image"],
            templatesDirectoryPath="templates",
            matchThresholdValue=matchThresholdValue,
            maxResultsCount=10,
            annotatedOutputDirectoryPath=outputDirectoryPath,
        )
        matchedPairs, falsePositiveCount, falseNegativeCount = matchPredictionsToGroundTruth(predictedBoxes, datasetSample["ground_truth"], iouThresholdValue)

        totalTruePositives += len(matchedPairs)
        totalFalsePositives += falsePositiveCount
        totalFalseNegatives += falseNegativeCount
        for matchedPair in matchedPairs:
            matchedIouValues.append(matchedPair["iou"])
            if isClickPointInsideBox(matchedPair["prediction"], matchedPair["ground_truth"]):
                clickInsideCount += 1

        print(f"  {datasetSample['name']}: gt={len(datasetSample['ground_truth'])} pred={len(predictedBoxes)} tp={len(matchedPairs)} fp={falsePositiveCount} fn={falseNegativeCount}")

    precisionValue = totalTruePositives / (totalTruePositives + totalFalsePositives) if (totalTruePositives + totalFalsePositives) else 0.0
    recallValue = totalTruePositives / (totalTruePositives + totalFalseNegatives) if (totalTruePositives + totalFalseNegatives) else 0.0
    f1Value = (2 * precisionValue * recallValue / (precisionValue + recallValue)) if (precisionValue + recallValue) else 0.0
    meanIouValue = float(np.mean(matchedIouValues)) if matchedIouValues else 0.0
    clickAccuracyValue = clickInsideCount / totalTruePositives if totalTruePositives else 0.0

    print("\n=== Evaluation summary ===")
    print(f"IoU threshold:        {iouThresholdValue}")
    print(f"True positives:       {totalTruePositives}")
    print(f"False positives:      {totalFalsePositives}")
    print(f"False negatives:      {totalFalseNegatives}")
    print(f"Precision:            {precisionValue:.3f}")
    print(f"Recall:               {recallValue:.3f}")
    print(f"F1 score:             {f1Value:.3f}")
    print(f"Mean IoU (matched):   {meanIouValue:.3f}")
    print(f"Click-inside-target:  {clickAccuracyValue:.3f}")


def main():
    argumentParser = argparse.ArgumentParser(description="Evaluate the template-matching UI detector against labeled boxes.")
    argumentParser.add_argument("--synthetic", type=int, default=0, help="Generate N synthetic labeled screenshots from templates and evaluate on them.")
    argumentParser.add_argument("--labels", type=str, default=None, help="Path to labels.json for real labeled screenshots.")
    argumentParser.add_argument("--screenshots", type=str, default="eval/screenshots", help="Directory of labeled screenshots.")
    argumentParser.add_argument("--match-threshold", type=float, default=0.45)
    argumentParser.add_argument("--iou-threshold", type=float, default=0.5)
    argumentParser.add_argument("--outputs", type=str, default="eval/outputs")
    parsedArguments = argumentParser.parse_args()

    if parsedArguments.labels:
        datasetSamples = loadLabeledDataset(parsedArguments.labels, parsedArguments.screenshots)
        print(f"Loaded {len(datasetSamples)} labeled screenshots from {parsedArguments.screenshots}")
    else:
        sampleCount = parsedArguments.synthetic if parsedArguments.synthetic > 0 else 5
        datasetSamples = buildSyntheticDataset("templates", sampleCount)
        print(f"Built {len(datasetSamples)} synthetic labeled screenshots from templates/")

    runEvaluation(datasetSamples, parsedArguments.match_threshold, parsedArguments.iou_threshold, parsedArguments.outputs)


if __name__ == "__main__":
    main()
