from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from google import genai

from private_config import GEMINI_API_KEY
from template_matcher import runTemplateMatcherWithMultiScale


outputDirectoryPath = Path("cv_milestone_outputs")
outputDirectoryPath.mkdir(exist_ok=True)
fallbackModelNames = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]


def needsScreen(userPromptText: str) -> bool:
    loweredPromptText = userPromptText.lower()
    screenRelatedKeywords = (
        "screen",
        "screenshot",
        "what am i looking",
        "what's on",
        "what is on",
        "visible",
        "look at",
    )
    return any(keywordText in loweredPromptText for keywordText in screenRelatedKeywords)


def buildScreenBundle():
    # grab one full-res screenshot for CV matching, then make a smaller upload copy for Gemini.
    fullResolutionScreenshotImage = pyautogui.screenshot()
    fullResolutionScreenshotArray = np.array(fullResolutionScreenshotImage)

    geminiUploadScreenshotImage = fullResolutionScreenshotImage.copy()
    geminiUploadScreenshotImage.thumbnail((1280, 1280))
    geminiUploadScreenshotArray = np.array(geminiUploadScreenshotImage)

    grayscaleImage = cv2.cvtColor(geminiUploadScreenshotArray, cv2.COLOR_RGB2GRAY)
    edgeImage = cv2.Canny(grayscaleImage, 100, 200)
    metricsByName = {
        "edge_density": float(np.mean(edgeImage > 0)),
        "brightness_mean": float(np.mean(grayscaleImage)),
        "brightness_std": float(np.std(grayscaleImage)),
    }
    milestoneImagePath = outputDirectoryPath / f"milestone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    figureObject, axisArray = plt.subplots(1, 3, figsize=(11, 4))
    axisArray[0].imshow(geminiUploadScreenshotArray); axisArray[0].set_title("RGB"); axisArray[0].axis("off")
    axisArray[1].imshow(grayscaleImage, cmap="gray"); axisArray[1].set_title("Gray"); axisArray[1].axis("off")
    axisArray[2].imshow(edgeImage, cmap="magma"); axisArray[2].set_title(f"Edges {metricsByName['edge_density']:.3f}"); axisArray[2].axis("off")
    plt.tight_layout(); figureObject.savefig(milestoneImagePath, dpi=140); plt.close(figureObject)
    return fullResolutionScreenshotArray, geminiUploadScreenshotImage, metricsByName, milestoneImagePath


def generateWithFallbackModels(agentClient, contentParts):
    lastErrorMessage = None
    for modelName in fallbackModelNames:
        try:
            modelResponse = agentClient.models.generate_content(model=modelName, contents=contentParts)
            return modelResponse, modelName
        except Exception as errorMessage:
            lastErrorMessage = errorMessage
            if "503" in str(errorMessage) or "UNAVAILABLE" in str(errorMessage):
                print(f"[model] {modelName} busy, trying another model...")
                continue
            raise
    raise lastErrorMessage


def runAgent():
    agentClient = genai.Client(api_key=GEMINI_API_KEY)
    print("Type prompts. q = quit. Screenshot is auto-attached for screen-related asks.\n")

    while True:
        userPromptText = input("You> ").strip()
        if userPromptText.lower() == "q":
            break
        if not userPromptText:
            continue

        if needsScreen(userPromptText):
            fullResolutionScreenshotArray, geminiUploadScreenshotImage, metricsByName, milestoneImagePath = buildScreenBundle()
            templateSummaryText, annotatedTemplateImagePath, _topMatchRows = runTemplateMatcherWithMultiScale(
                screenshotRgbImage=fullResolutionScreenshotArray,
                templatesDirectoryPath="templates",
                matchThresholdValue=0.55,
                maxResultsCount=4,
                annotatedOutputDirectoryPath=str(outputDirectoryPath),
            )
            contentParts = [
                (
                    f"{userPromptText}\n\n"
                    f"CV template matching hints. {templateSummaryText}"
                ),
                geminiUploadScreenshotImage,
            ]
            roundedMetricsByName = {
                "edge_density": round(metricsByName["edge_density"], 4),
                "brightness_mean": round(metricsByName["brightness_mean"], 4),
                "brightness_std": round(metricsByName["brightness_std"], 4),
            }
            print(f"[tool] screenshot attached | saved {milestoneImagePath.name} | metrics={roundedMetricsByName}")
            print(f"[matcher] {templateSummaryText}")
            if annotatedTemplateImagePath:
                print(f"[matcher] saved annotated match image: {annotatedTemplateImagePath}")
        else:
            contentParts = [userPromptText]
            print("[tool] text-only route")

        try:
            modelResponse, usedModelName = generateWithFallbackModels(agentClient, contentParts)
            print(f"[model] used {usedModelName}")
            print(f"\nAgent: {modelResponse.text}\n{'-' * 80}")
        except Exception as errorMessage:
            # if API has a bad moment just print it and keep going
            print(f"\nAgent error: {errorMessage}\n{'-' * 80}")


if __name__ == "__main__":
    runAgent()

