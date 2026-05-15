using UnityEngine;

public class AmbientManager : MonoBehaviour
{
    public AudioSource heavyWind;
    public AudioSource nightCritters;
    public AudioSource idleEngine;
    public AudioSource dogBarking;
    public AudioSource childrenPlaying;
    public AudioSource jackhammer;
    public AudioSource carHorn;
    public AudioSource siren;

    public void ResetAmbient()
    {
        SetupSource(heavyWind, 0.02f, 0.05f);
        SetupSource(nightCritters, 0.02f, 0.05f);

        SetupSource(idleEngine, 0.05f, 0.1f);
        SetupSource(dogBarking, 0.05f, 0.1f);
        SetupSource(childrenPlaying, 0.05f, 0.1f);
        SetupSource(jackhammer, 0.05f, 0.1f);
        SetupSource(carHorn, 0.05f, 0.1f);
        SetupSource(siren, 0.05f, 0.1f);
    }

    void SetupSource(AudioSource source, float minVolume, float maxVolume)
    {
        if (source == null) return;

        source.Stop();
        source.time = 0f;

        bool include = Random.value > 0.7f;

        if (include)
        {
            source.volume = Random.Range(minVolume, maxVolume);
            source.Play();
        }
    }
}