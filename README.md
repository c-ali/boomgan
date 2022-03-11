# BoomGAN
__A Music-visualization tool utilizing NVIDIA's [StyleGAN 3](https://github.com/NVlabs/stylegan3)__

Boomgan features StyleGAN3 latent space exploration synchronized to the beat of an audio track. [Beat detection](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf) is performed on the audio track and one latent space vector is generated for each pulse. The remaining frames then linearly interpolate these latent space vectors. Different settings for latent space exploration are available.

# Modes

| Mode |        |    |
:-------------------------:|:-------------------------:|:-------------------------:
rjump |![Randomly jumping between random latent space vectors (rjump)](examples/rjump.gif) | ![Cutoff rjump on metfaces dataset (rjump 4:)](examples/faces_rjump_4.gif)
twocircle|![Jumping between two circles in latent space (twocircle 4:)](examples/twocircle_4.gif) | ![Twocircle method but only fine features are changed (rjump 12:)](examples/twocircle_onlyhigh.gif)
rwalk / orwalk|![Random walk in latent space(rwalk 4:)](examples/rwalk.gif) | ![Orthogonal random walk in latent space (orwalk 4:)](examples/orwalk_4.gif)


- "rjump": all generated latent space points are random normal.
- "rwalk": a random walk of fixed step size in latent space is performed.
- "orwalk": a random orthogonal walk of fixed step size in latent space is performed. Orthogonal steps should somewhat make the change rate for each beat similar.
- "twocircle": two circles of different or same radius are instanciated in latent space. The inner circle is rotated by an offset. At each step, we jump from the inner to the outer circle. Gives a "pulsating" style.

# Other Parameters

- "stretch": change rate of the animation
- "latent cutoff": only change latent space parameters at a latent dimension > latent cutoff. Ranges from 0-16.

# Quickstart 

`python main.py --audio_file sample_short.mp3 --network https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl --mode ojump`

# Dependencies
Boomgan includes StyleGAN3 as a submodule. Other dependencies are listed in dependencies.txt and can be installed using `pip install -t dependencies.txt`.

