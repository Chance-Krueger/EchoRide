#if UNITY_EDITOR
using System.IO;
using UnityEditor.Recorder;
using UnityEngine;

public static class RecorderManager
{
    private static RecorderController controller;
    private static RecorderControllerSettings controllerSettings;
    private static AudioRecorderSettings audioRecorderSettings;

    public static void StartWavRecording(string fileName)
    {
        if (controller != null && controller.IsRecording())
        {
            Debug.LogWarning("Recording is already in progress.");
            return;
        }

       
        string recordingsFolder = Path.Combine(Directory.GetCurrentDirectory(), "Recordings");
        Directory.CreateDirectory(recordingsFolder);

        controllerSettings = ScriptableObject.CreateInstance<RecorderControllerSettings>();

        audioRecorderSettings = ScriptableObject.CreateInstance<AudioRecorderSettings>();
        audioRecorderSettings.name = "Auto Audio Recorder";
        audioRecorderSettings.Enabled = true;

        // Recorder saves WAV for Audio Recorder
        audioRecorderSettings.OutputFile = Path.Combine("Recordings", fileName);

        controllerSettings.AddRecorderSettings(audioRecorderSettings);
        controllerSettings.SetRecordModeToManual();

        controller = new RecorderController(controllerSettings);

        try
        {
            controller.PrepareRecording();
            bool started = controller.StartRecording();

            if (!started)
            {
                Debug.LogError("Recorder failed to start. Check the Console for Recorder errors.");
            }
            else
            {
                Debug.Log("Recording STARTED: " + fileName);
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to start recording: " + ex.Message);
        }
    }

    public static void StopWavRecording()
    {
        if (controller == null)
        {
            Debug.LogWarning("No RecorderController exists.");
            return;
        }

        if (!controller.IsRecording())
        {
            Debug.LogWarning("Recorder is not currently recording.");
            controller = null;
            controllerSettings = null;
            audioRecorderSettings = null;
            return;
        }

        try
        {
            controller.StopRecording();
            Debug.Log("Recording STOPPED");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to stop recording: " + ex.Message);
        }
        finally
        {
            controller = null;
            controllerSettings = null;
            audioRecorderSettings = null;
        }
    }
}
#endif


