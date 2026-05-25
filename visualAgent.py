from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from google import genai

from private_config import GEMINI_API_KEY


outputDirectoryPath = Path("cv_milestone_outputs")
outputDirectoryPath.mkdir(exist_ok=True)


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
    # grab screen once, squeeze size, and get quick CV stats
    screenshotImage = pyautogui.screenshot()
    screenshotImage.thumbnail((1280, 1280))  # smaller upload so it does not take forever
    screenshotArray = np.array(screenshotImage)
    grayscaleImage = cv2.cvtColor(screenshotArray, cv2.COLOR_RGB2GRAY)
    edgeImage = cv2.Canny(grayscaleImage, 100, 200)
    metricsByName = {
        "edge_density": float(np.mean(edgeImage > 0)),
        "brightness_mean": float(np.mean(grayscaleImage)),
        "brightness_std": float(np.std(grayscaleImage)),
    }
    milestoneImagePath = outputDirectoryPath / f"milestone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    figureObject, axisArray = plt.subplots(1, 3, figsize=(11, 4))
    axisArray[0].imshow(screenshotArray); axisArray[0].set_title("RGB"); axisArray[0].axis("off")
    axisArray[1].imshow(grayscaleImage, cmap="gray"); axisArray[1].set_title("Gray"); axisArray[1].axis("off")
    axisArray[2].imshow(edgeImage, cmap="magma"); axisArray[2].set_title(f"Edges {metricsByName['edge_density']:.3f}"); axisArray[2].axis("off")
    plt.tight_layout(); figureObject.savefig(milestoneImagePath, dpi=140); plt.close(figureObject)
    return screenshotImage, metricsByName, milestoneImagePath


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
            screenshotImage, metricsByName, milestoneImagePath = buildScreenBundle()
            contentParts = [userPromptText, screenshotImage]
            roundedMetricsByName = {
                "edge_density": round(metricsByName["edge_density"], 4),
                "brightness_mean": round(metricsByName["brightness_mean"], 4),
                "brightness_std": round(metricsByName["brightness_std"], 4),
            }
            print(f"[tool] screenshot attached | saved {milestoneImagePath.name} | metrics={roundedMetricsByName}")
        else:
            contentParts = [userPromptText]
            print("[tool] text-only route")

        try:
            modelResponse = agentClient.models.generate_content(model="gemini-2.5-flash", contents=contentParts)
            print(f"\nAgent: {modelResponse.text}\n{'-' * 80}")
        except Exception as errorMessage:
            # if API has a bad moment just print it and keep going
            print(f"\nAgent error: {errorMessage}\n{'-' * 80}")


if __name__ == "__main__":
    runAgent()

