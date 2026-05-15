using UnityEngine;

public class MovingCar : MonoBehaviour
{
    public Vector3 start;
    public Vector3 end;
    public float speed = 5f;
    public AudioSource carAudio;

    private bool isMoving = false;

   
    void Start()
    {
        if (carAudio != null)
        {
            carAudio.velocityUpdateMode = AudioVelocityUpdateMode.Dynamic;
            carAudio.playOnAwake = false;
        }
    }

    public void BeginMovement(Vector3 newStart, Vector3 newEnd, float newSpeed)
    {
        start = newStart;
        end = newEnd;
        speed = newSpeed;
        transform.position = start;

        if (carAudio != null)
        {
           // Calculate how long movement will take
            float distance = Vector3.Distance(start, end);
            float moveDuration = distance / speed;

            // Stretch/compress clip to match movement time
            float clipLength = carAudio.clip.length;
            carAudio.pitch = clipLength / moveDuration;

            // Reset clip to start
            carAudio.time = 0f;

            // Start sound
            carAudio.Play();
        }

        isMoving = true;
    }

    void Update()
    {
        if (!isMoving) return;

        transform.position = Vector3.MoveTowards(
            transform.position,
            end,
            speed * Time.deltaTime
        );

        if (carAudio != null)
        {
            // Bike assumed at center (0,0,0)
            float dist = Vector3.Distance(transform.position, Vector3.zero);

        
            carAudio.volume = Mathf.Clamp01(1f - (dist / 40f));
        }

        if (Vector3.Distance(transform.position, end) < 0.01f)
        {
            isMoving = false;

            if (carAudio != null)
               carAudio.Stop();
        }
    }

    public bool IsMoving()
    {
        return isMoving;
    }
}