# Boomgan
__A Music-visualization tool utilizing NVIDIA's [StyleGAN 3](https://github.com/NVlabs/stylegan3)__

Boomgan features StyleGAN3 latent space exploration synchronized to the beat of an audio sample. [Beat detection](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf) is performed on the audio track and one latent space vector is generated for each pulse. The remaining frames then linearly interpolate these latent space vectors. Different settings for latent space exploration are available:

- "rjump": all generated latent space points are random normal
- "rwalk": a random walk of fixed step size in latent space is performed
- "orwalk": a random orthogonal walk of fixed step size in latent space is performed. Orthogonal steps should somewhat make the change rate for each beat similar.

# Quickstart 

`python main.py --audio_file sample_short.mp3 --network https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl --mode ojump`

# Dependencies
Boomgan includes StyleGAN3 as a submodule. Other dependencies are listed in dependencies.txt and can be installed using `pip install -t dependencies.txt`.

# Usage
