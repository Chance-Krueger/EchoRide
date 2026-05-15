using System.Collections;
using UnityEngine.InputSystem;
using UnityEngine;

public class ScenarioRunner : MonoBehaviour
{
    [System.Serializable]
    public class ScenarioData
    {
        public string label;
        public Vector3 start;
        public Vector3 end;
        public float speed;

        public ScenarioData(string label, Vector3 start, Vector3 end, float speed)
        {
            this.label = label;
            this.start = start;
            this.end = end;
            this.speed = speed;
        }
    }

    ScenarioData GenerateFrontPassLeftToRight()
    {
        float z = Random.Range(1f, 5f);
        float startX = Random.Range(-30f, -15f);
        float endX = Random.Range(15f, 30f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(startX, 0, z);
        Vector3 end = new Vector3(endX, 0, z);

        return new ScenarioData("FrontPass_L2R", start, end, speed);
    }

    ScenarioData GenerateFrontPassRightToLeft()
    {
        float z = Random.Range(1f, 5f);
        float startX = Random.Range(15f, 30f);
        float endX = Random.Range(-30f, -15f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(startX, 0, z);
        Vector3 end = new Vector3(endX, 0, z);

        return new ScenarioData("FrontPass_R2L", start, end, speed);
    }

     ScenarioData GenerateRearPassLeftToRight()
    {
        float z = Random.Range(-5f, -1f);
        float startX = Random.Range(-30f, -15f);
        float endX = Random.Range(15f, 30f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(startX, 0, z);
        Vector3 end = new Vector3(endX, 0, z);

        return new ScenarioData("RearPass_L2R", start, end, speed);
    }

    ScenarioData GenerateRearPassRightToLeft()
    {
        float z = Random.Range(-5f, -1f);
        float startX = Random.Range(15f, 30f);
        float endX = Random.Range(-30f, -15f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(startX, 0, z);
        Vector3 end = new Vector3(endX, 0, z);

        return new ScenarioData("RearPass_R2L", start, end, speed);
    }

    ScenarioData GenerateLeftPassFrontToBack()
    {
        float x = Random.Range(-5f, -1f);
        float startZ = Random.Range(15f, 30f);
        float endZ = Random.Range(-30f, -15f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(x, 0, startZ);
        Vector3 end = new Vector3(x, 0, endZ);

        return new ScenarioData("LeftPass_F2B", start, end, speed);
    }

    ScenarioData GenerateLeftPassBackToFront()
    {
        float x = Random.Range(-5f, -1f);
        float startZ = Random.Range(-30f, -15f);
        float endZ = Random.Range(15f, 30f);
        float speed = Random.Range(5f, 11f);

        Vector3 start = new Vector3(x, 0, startZ);
        Vector3 end = new Vector3(x, 0, endZ);

        return new ScenarioData("LeftPass_B2F", start, end, speed);
    }

    ScenarioData GenerateRightPassFrontToBack()
    {
        float x = Random.Range(1f, 5f);
        float startZ = Random.Range(15f, 30f);
        float endZ = Random.Range(-30f, -15f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(x, 0, startZ);
        Vector3 end = new Vector3(x, 0, endZ);

        return new ScenarioData("RightPass_F2B", start, end, speed);
    }

    ScenarioData GenerateRightPassBackToFront()
    {
        float x = Random.Range(1f, 5f);
        float startZ = Random.Range(-30f, -15f);
        float endZ = Random.Range(15f, 30f);
        float speed = Random.Range(1f, 11f);

        Vector3 start = new Vector3(x, 0, startZ);
        Vector3 end = new Vector3(x, 0, endZ);

        return new ScenarioData("RightPass_B2F", start, end, speed);
    }

    public MovingCar movingCar;
    public AmbientManager ambientManager;

    private int clipCounter = 651;

    public string currentFileName;
    private ScenarioData currentScenario;

    void Start()
    {
        StartCoroutine(RunScenarios());
    }

    IEnumerator RunScenarios()
    {
        while (true)
        {
            currentScenario = GenerateRandomScenario();
            currentFileName = currentScenario.label + "_" + clipCounter.ToString("D2");

            Debug.Log("READY: " + currentFileName + " | Starting automatically in 2 seconds");

            yield return new WaitForSeconds(2f);

            RecorderBridge.StartWavRecording(currentFileName);

            StartScenario(currentScenario);

            while (movingCar.IsMoving())
            {
                yield return null;
            }

            RecorderBridge.StopWavRecording();

            Debug.Log("FINISHED: " + currentFileName);

            clipCounter++;

            yield return new WaitForSeconds(1.5f);
        }
    }

    ScenarioData GenerateRandomScenario()
    {
        int choice = Random.Range(0, 8);

        switch (choice)
        {
            case 0: return GenerateFrontPassLeftToRight();
            case 1: return GenerateFrontPassRightToLeft();
            case 2: return GenerateRearPassLeftToRight();
            case 3: return GenerateRearPassRightToLeft();
            case 4: return GenerateLeftPassFrontToBack();
            case 5: return GenerateLeftPassBackToFront();
            case 6: return GenerateRightPassFrontToBack();
            default: return GenerateRightPassBackToFront();
        }
    }

    void StartScenario(ScenarioData scenario)
    {
        if (ambientManager != null)
        {
            ambientManager.ResetAmbient();
        }

        movingCar.BeginMovement(
            scenario.start,
            scenario.end,
            scenario.speed
        );

        Debug.Log("STARTED: " + currentFileName +
                " | Start: " + scenario.start +
                " | End: " + scenario.end +
                " | Speed: " + scenario.speed);
    }
}