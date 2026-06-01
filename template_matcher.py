from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


def getTemplateHintSentence(templateName):
    # short UI hints so the agent can talk normal without guessing too hard
    if templateName == "homebar":
        return "Bar with Home selected; you are likely on your feed."
    elif templateName == "mynetworkbar":
        return "Bar with My Network selected; this looks like your network page."
    elif templateName == "notificaitonsbar":
        return "Bar with Notifications selected; you are likely in notifications."
    elif templateName == "notificationssection":
        return "Notifications section is visible on screen."
    elif templateName == "profile":
        return "Your profile header is visible."
    elif templateName == "connections":
        return "Connections area is on screen."
    elif templateName == "connectioncount":
        return "Connection count element is visible."
    elif templateName == "sortconnections":
        return "Sort control for connections is visible."
    else:
        return "Known template matched on screen."


def runTemplateMatcherWithMultiScale(screenshotRgbImage, templatesDirectoryPath="templates", matchThresholdValue=0.45, maxResultsCount=4, annotatedOutputDirectoryPath="cv_milestone_outputs", scaleValues=None):
    # this is the scale mismatch fix: test each template at multiple sizes and keep best hit.
    if scaleValues is None:
        scaleValues = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

    screenshotGrayImage = cv2.cvtColor(screenshotRgbImage, cv2.COLOR_RGB2GRAY)
    screenshotHeight, screenshotWidth = screenshotGrayImage.shape[:2]
    templateImagePaths = sorted(Path(templatesDirectoryPath).glob("*.png"))
    if not templateImagePaths:
        return "no template files found", None, []

    bestMatchRows = []
    for templateImagePath in templateImagePaths:
        templateGrayImageOriginal = cv2.imread(str(templateImagePath), cv2.IMREAD_GRAYSCALE)
        if templateGrayImageOriginal is None:
            continue

        templateName = templateImagePath.stem
        bestScoreValue = -1.0
        bestLocation = (0, 0)
        bestScaleValue = 1.0
        bestWidth = 0
        bestHeight = 0

        for scaleValue in scaleValues:
            scaledWidth = int(round(templateGrayImageOriginal.shape[1] * scaleValue))
            scaledHeight = int(round(templateGrayImageOriginal.shape[0] * scaleValue))
            if scaledWidth < 3 or scaledHeight < 3:
                continue
            if scaledWidth > screenshotWidth or scaledHeight > screenshotHeight:
                continue

            scaledTemplateGrayImage = cv2.resize(templateGrayImageOriginal, (scaledWidth, scaledHeight), interpolation=cv2.INTER_AREA if scaleValue < 1.0 else cv2.INTER_LINEAR)
            matchResponseMap = cv2.matchTemplate(screenshotGrayImage, scaledTemplateGrayImage, cv2.TM_CCOEFF_NORMED)
            _ignoredMinValue, maxScoreValue, _ignoredMinLocation, maxScoreLocation = cv2.minMaxLoc(matchResponseMap)
            if maxScoreValue > bestScoreValue:
                bestScoreValue = float(maxScoreValue)
                bestLocation = (int(maxScoreLocation[0]), int(maxScoreLocation[1]))
                bestScaleValue = float(scaleValue)
                bestWidth = int(scaledWidth)
                bestHeight = int(scaledHeight)

        if bestScoreValue >= matchThresholdValue:
            centerPositionX = bestLocation[0] + bestWidth // 2
            centerPositionY = bestLocation[1] + bestHeight // 2
            bestMatchRows.append({
                "template_name": templateName,
                "confidence_score": bestScoreValue,
                "x": bestLocation[0],
                "y": bestLocation[1],
                "width": bestWidth,
                "height": bestHeight,
                "center_x": centerPositionX,
                "center_y": centerPositionY,
                "scale": bestScaleValue,
            })

    if not bestMatchRows:
        return "no confident template matches", None, []

    bestMatchRows.sort(key=lambda matchRow: matchRow["confidence_score"], reverse=True)
    bestMatchRows = keepOnlyHighestBarTemplate(bestMatchRows)
    topMatchRows = selectNonOverlappingMatches(bestMatchRows, maxResultsCount=maxResultsCount, iouThresholdValue=0.35)

    summaryText = " | ".join(
        f"name={matchRow['template_name']}, "
        f"confidence={matchRow['confidence_score']:.2f}, "
        f"x={matchRow['x']}, y={matchRow['y']}, "
        f"width={matchRow['width']}, height={matchRow['height']}, "
        f"center_x={matchRow['center_x']}, center_y={matchRow['center_y']}, "
        f"scale={matchRow['scale']:.2f}"
        for matchRow in topMatchRows
    )

    annotatedOutputPathObject = Path(annotatedOutputDirectoryPath)
    annotatedOutputPathObject.mkdir(exist_ok=True)
    annotatedFilePath = annotatedOutputPathObject / f"template_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    annotatedBgrImage = cv2.cvtColor(screenshotRgbImage.copy(), cv2.COLOR_RGB2BGR)
    for matchRow in topMatchRows:
        topLeftX = matchRow["x"]
        topLeftY = matchRow["y"]
        bottomRightX = topLeftX + matchRow["width"]
        bottomRightY = topLeftY + matchRow["height"]
        cv2.rectangle(annotatedBgrImage, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0, 255, 0), 2)
        cv2.circle(annotatedBgrImage, (matchRow["center_x"], matchRow["center_y"]), 4, (0, 255, 255), -1)
        cv2.putText(annotatedBgrImage, f"{matchRow['template_name']} {matchRow['confidence_score']:.2f} s={matchRow['scale']:.2f}", (topLeftX, max(18, topLeftY - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(annotatedFilePath), annotatedBgrImage)

    return summaryText, str(annotatedFilePath), topMatchRows


def computeIntersectionOverUnion(matchRowOne, matchRowTwo):
    rowOneLeft = matchRowOne["x"]
    rowOneTop = matchRowOne["y"]
    rowOneRight = rowOneLeft + matchRowOne["width"]
    rowOneBottom = rowOneTop + matchRowOne["height"]

    rowTwoLeft = matchRowTwo["x"]
    rowTwoTop = matchRowTwo["y"]
    rowTwoRight = rowTwoLeft + matchRowTwo["width"]
    rowTwoBottom = rowTwoTop + matchRowTwo["height"]

    intersectionLeft = max(rowOneLeft, rowTwoLeft)
    intersectionTop = max(rowOneTop, rowTwoTop)
    intersectionRight = min(rowOneRight, rowTwoRight)
    intersectionBottom = min(rowOneBottom, rowTwoBottom)

    intersectionWidth = max(0, intersectionRight - intersectionLeft)
    intersectionHeight = max(0, intersectionBottom - intersectionTop)
    intersectionArea = intersectionWidth * intersectionHeight
    if intersectionArea <= 0:
        return 0.0

    rowOneArea = max(1, matchRowOne["width"] * matchRowOne["height"])
    rowTwoArea = max(1, matchRowTwo["width"] * matchRowTwo["height"])
    unionArea = rowOneArea + rowTwoArea - intersectionArea
    if unionArea <= 0:
        return 0.0
    return float(intersectionArea / unionArea)


def selectNonOverlappingMatches(sortedMatchRows, maxResultsCount=4, iouThresholdValue=0.35):
    keptMatchRows = []
    for candidateMatchRow in sortedMatchRows:
        overlapsExistingRow = any(computeIntersectionOverUnion(candidateMatchRow, keptMatchRow) > iouThresholdValue for keptMatchRow in keptMatchRows)
        if not overlapsExistingRow:
            keptMatchRows.append(candidateMatchRow)
        if len(keptMatchRows) >= maxResultsCount:
            break
    return keptMatchRows


def keepOnlyHighestBarTemplate(sortedMatchRows):
    highestBarTemplateRow = None
    nonBarTemplateRows = []
    for matchRow in sortedMatchRows:
        templateName = matchRow["template_name"]
        if templateName.endswith("bar"):
            if highestBarTemplateRow is None:
                highestBarTemplateRow = matchRow
        else:
            nonBarTemplateRows.append(matchRow)

    if highestBarTemplateRow is None:
        return sortedMatchRows

    return [highestBarTemplateRow] + nonBarTemplateRows


def summarizeTemplateMatches(screenshotRgbImage, templatesDirectoryPath="templates", matchThresholdValue=0.55, maxResultsCount=3):
    summaryText, _, _ = runTemplateMatcherWithMultiScale(screenshotRgbImage=screenshotRgbImage, templatesDirectoryPath=templatesDirectoryPath, matchThresholdValue=matchThresholdValue, maxResultsCount=maxResultsCount)
    return summaryText


def summarize_template_matches(screenshot_rgb, templates_dir="templates", match_threshold=0.62, max_results=3):
    return summarizeTemplateMatches(screenshotRgbImage=screenshot_rgb, templatesDirectoryPath=templates_dir, matchThresholdValue=match_threshold, maxResultsCount=max_results)
