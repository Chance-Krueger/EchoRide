import tkinter as tk
from tkinter import filedialog, ttk
import math
from pathlib import Path
import threading

from predictor import predictWavFile

baseDir = Path(__file__).resolve().parent.parent
rawDataDir = baseDir / "data" / "raw"

directions = [
    "FrontPass",
    "LeftTurn",
    "RightTurn",
]

presetFiles = {
    "FrontPass": rawDataDir / "FrontPass" / "FrontPass_L2R_NightCritters.wav",
    "LeftTurn": rawDataDir / "LeftTurn" / "LeftTurn_UrbanStreet.wav",
    "RightTurn": rawDataDir / "RightTurn" / "RightTurn_HeavyWind.wav",
}

pastelBg = "#f6f1f7"
panelBg = "#e9f3f0"
buttonOff = "#f7d6e0"
buttonOn = "#c7e8d4"
accent = "#7c6f9f"
textColor = "#3d3a4b"


class EchoRideGui:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoRide")
        self.root.geometry("850x500")
        self.root.configure(bg=pastelBg)

        self.isOn = False
        self.selectedAudio = None
        self.selectedPreset = tk.StringVar(value="FrontPass")
        self.currentDirection = tk.StringVar(value="Off")
        self.currentInputType = None

        self.phase = 0
        self.animating = False
        self.isProcessing = False

        self.buildUi()

    def buildUi(self):
        main = tk.Frame(self.root, bg=pastelBg)
        main.pack(fill="both", expand=True, padx=35, pady=35)

        left = tk.Frame(main, bg=pastelBg)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(
            main,
            bg=panelBg,
            width=270,
            highlightbackground="#d4d0dc",
            highlightthickness=1
        )
        right.pack(side="right", fill="y", padx=(25, 0))
        right.pack_propagate(False)

        self.visualCanvas = tk.Canvas(
            left,
            width=500,
            height=300,
            bg=pastelBg,
            highlightthickness=0
        )
        self.visualCanvas.pack(pady=(10, 0))

        self.buttonCanvas = tk.Canvas(
            left,
            width=210,
            height=210,
            bg=pastelBg,
            highlightthickness=0
        )
        self.buttonCanvas.place(relx=0.5, rely=0.43, anchor="center")
        self.drawPowerButton()

        self.directionLabel = tk.Label(
            left,
            textvariable=self.currentDirection,
            font=("Segoe UI", 20, "bold"),
            bg=pastelBg,
            fg=textColor
        )
        self.directionLabel.pack(pady=(0, 0))

        self.statusLabel = tk.Label(
            left,
            text="Choose an input, then press the power button.",
            font=("Segoe UI", 10),
            bg=pastelBg,
            fg="#6d6875"
        )
        self.statusLabel.pack(pady=(8, 0))

        panelTitle = tk.Label(
            right,
            text="Audio Input",
            font=("Segoe UI", 16, "bold"),
            bg=panelBg,
            fg=textColor
        )
        panelTitle.pack(pady=(25, 20))

        self.uploadButton = tk.Button(
            right,
            text="Upload Audio",
            command=self.uploadAudio,
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg=textColor,
            relief="flat",
            height=2
        )
        self.uploadButton.pack(fill="x", padx=25, pady=(0, 12))

        self.presetButton = tk.Button(
            right,
            text="Presets",
            command=self.choosePreset,
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg=textColor,
            relief="flat",
            height=2
        )
        self.presetButton.pack(fill="x", padx=25, pady=(12, 5))

        self.presetDropdown = ttk.Combobox(
            right,
            textvariable=self.selectedPreset,
            values=directions,
            state="readonly"
        )
        self.presetDropdown.pack(fill="x", padx=25, pady=(0, 15))
        self.presetDropdown.bind("<<ComboboxSelected>>", self.onPresetChanged)

        self.fileLabel = tk.Label(
            right,
            text="No audio selected",
            wraplength=210,
            font=("Segoe UI", 8),
            bg=panelBg,
            fg="#6d6875"
        )
        self.fileLabel.pack(padx=25, pady=(25, 0))

    def drawPowerButton(self):
        self.buttonCanvas.delete("all")

        color = buttonOn if self.isOn else buttonOff
        symbol = "👂" if self.isOn else "⏻"

        self.buttonCanvas.create_oval(
            10, 10, 200, 200,
            fill=color,
            outline="#ffffff",
            width=6
        )

        self.buttonCanvas.create_text(
            105, 105,
            text=symbol,
            font=("Segoe UI Emoji", 58),
            fill=textColor
        )

        self.buttonCanvas.bind("<Button-1>", lambda event: self.togglePower())

    def togglePower(self):
        if self.isOn:
            self.turnOff()
        else:
            self.turnOn()

    def turnOn(self):
        if self.currentInputType is None:
            self.statusLabel.config(text="Choose Upload Audio or Presets first.")
            return

        if self.isProcessing:
            return

        self.isOn = True
        self.animating = False
        self.visualCanvas.delete("all")
        self.drawPowerButton()
        self.currentDirection.set("Listening")
        self.statusLabel.config(text="Processing audio...")

        predictionThread = threading.Thread(
            target=self.processCurrentInputInBackground,
            daemon=True
        )
        predictionThread.start()

    def turnOff(self):
        self.isOn = False
        self.isProcessing = False
        self.animating = False
        self.visualCanvas.delete("all")
        self.drawPowerButton()
        self.currentDirection.set("Off")
        self.statusLabel.config(text="Choose an input, then press the power button.")

    def uploadAudio(self):
        path = filedialog.askopenfilename(
            title="Upload Audio File",
            filetypes=[("WAV files", "*.wav")]
        )

        if path:
            self.selectedAudio = path
            self.currentInputType = "UploadAudio"
            self.fileLabel.config(text=Path(path).name)
            self.statusLabel.config(text="Uploaded audio selected. Press the power button.")

    def choosePreset(self):
        self.currentInputType = "Preset"
        preset = self.selectedPreset.get()
        presetPath = self.findPresetAudio(preset)

        if presetPath is None:
            self.selectedAudio = None
            self.fileLabel.config(text=f"No WAV found for preset: {preset}")
            self.statusLabel.config(text=f"Could not find data/raw/{preset}/")
            return

        self.selectedAudio = str(presetPath)
        self.fileLabel.config(text=f"Preset selected: {preset}")
        self.statusLabel.config(text="Preset selected. Press the power button.")

    def onPresetChanged(self, event=None):
        self.choosePreset()

    def findPresetAudio(self, preset):
        presetPath = presetFiles.get(preset)

        if presetPath is None:
            return None

        if not presetPath.exists():
            return None

        return presetPath

    def processCurrentInputInBackground(self):
        self.isProcessing = True

        try:
            if self.currentInputType == "UploadAudio":
                direction = self.processUploadedAudio()
            elif self.currentInputType == "Preset":
                direction = self.processPresetAudio()
            else:
                direction = None

            self.root.after(0, lambda: self.finishProcessing(direction, None))

        except Exception as error:
            self.root.after(0, lambda: self.finishProcessing(None, error))

    def finishProcessing(self, direction, error):
        self.isProcessing = False

        if not self.isOn:
            return

        if error is not None:
            self.currentDirection.set("Error")
            self.statusLabel.config(text=f"Prediction failed: {error}")
            return

        if direction is None:
            self.currentDirection.set("No input")
            self.statusLabel.config(text="No usable input selected.")
            return

        self.currentDirection.set(direction)
        self.statusLabel.config(text=f"Model predicted: {direction}")
        self.startVibrationAnimation(direction)

    def processUploadedAudio(self):
        return predictWavFile(self.selectedAudio)

    def processPresetAudio(self):
        if self.selectedAudio is None:
            raise FileNotFoundError("No preset audio file selected.")

        return predictWavFile(self.selectedAudio)

    def startVibrationAnimation(self, direction):
        self.phase = 0
        self.animating = True
        self.animateVibrations(direction)

    def animateVibrations(self, direction):
        if not self.animating:
            return

        self.visualCanvas.delete("all")
        self.phase += 1

        pattern = self.getBlinkPattern(direction)
        isOn = pattern[int(self.phase) % len(pattern)]

        self.drawSideVibrations(direction, isOn)

        self.root.after(170, lambda: self.animateVibrations(direction))

    def getBlinkPattern(self, direction):
        # RightTurn: 1 blink
        if direction == "RightTurn":
            return [True, False, False, False, False, False, False, False, False]

        # LeftTurn: 2 blinks
        if direction == "LeftTurn":
            return [True, False, True, False, False, False, False, False, False, False]

        # FrontPass: 3 blinks
        if direction == "FrontPass":
            return [True, False, True, False, True, False, False, False, False, False, False, False]

        # RearPass: 4 blinks
        if direction == "RearPass":
            return [True, False, True, False, True, False, True, False, False, False, False, False, False, False]

        # LeftPass: 5 blinks
        if direction == "LeftPass":
            return [True, False, True, False, True, False, True, False, True, False, False, False, False, False, False, False]

        # RightPass: 6 blinks
        if direction == "RightPass":
            return [True, False, True, False, True, False, True, False, True, False, True, False, False, False, False, False, False, False]

        # RearCrash: rapid .
        if direction == "RearCrash":
            return [True, False, True, False, True, False, True, False]

        return [False]

    def drawSideVibrations(self, direction, isOn):
        if not isOn:
            return

        if direction in ["LeftPass", "LeftTurn"]:
            self.drawVibrationSquiggle(125, 150)

        elif direction in ["RightPass", "RightTurn"]:
            self.drawVibrationSquiggle(375, 150)

        elif direction in ["FrontPass", "RearPass", "RearCrash"]:
            self.drawVibrationSquiggle(125, 150)
            self.drawVibrationSquiggle(375, 150)

    def drawVibrationSquiggle(self, centerX, centerY):
        points = []
        width = 110
        amplitude = 18

        for i in range(0, width, 5):
            x = centerX - width / 2 + i
            y = centerY + math.sin(i * 0.25 + self.phase) * amplitude
            points.extend([x, y])

        self.visualCanvas.create_line(
            points,
            smooth=True,
            width=4,
            fill=accent,
            capstyle="round"
        )


def main():
    root = tk.Tk()
    EchoRideGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()