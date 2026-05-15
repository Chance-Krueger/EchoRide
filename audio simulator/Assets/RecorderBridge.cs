using System;
using System.Reflection;
using UnityEngine;

public static class RecorderBridge
{
    public static void StartWavRecording(string fileName)
    {
#if UNITY_EDITOR
        TryInvoke("StartWavRecording", new object[] { fileName });
#endif
    }

    public static void StopWavRecording()
    {
#if UNITY_EDITOR
        TryInvoke("StopWavRecording", null);
#endif
    }

#if UNITY_EDITOR
    private static void TryInvoke(string methodName, object[] args)
    {
        var editorAssembly = Assembly.Load("Assembly-CSharp-Editor");
        if (editorAssembly == null)
        {
            Debug.LogWarning("Could not load Assembly-CSharp-Editor.");
            return;
        }

        var recorderType = editorAssembly.GetType("RecorderManager");
        if (recorderType == null)
        {
            Debug.LogWarning("Could not find RecorderManager in editor assembly.");
            return;
        }

        var method = recorderType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static);
        if (method == null)
        {
            Debug.LogWarning("Could not find method: " + methodName);
            return;
        }

        method.Invoke(null, args);
    }
#endif
}
