import os, sys

sys.path.append("stylegan3")

import torch
import numpy as np
import PIL.Image
from stylegan3 import dnnlib
from stylegan3.legacy import load_network_pkl
from scipy import interpolate
from tqdm import tqdm
import imageio
import librosa
import warnings
import ffmpeg

warnings.filterwarnings("ignore", category=Warning)





def orthogonalize(normal, non_ortho):
    h = normal * non_ortho
    return non_ortho - normal * h


def make_orthonormal_vector(normal):
    candidates = np.array(shape=(3, 3))
    candidates[0] = [1, 0, 0]
    candidates[1] = [0, 1, 0]
    candidates[2] = [0, 0, 1]

    min = 1e20
    cIndex = 0
    for i in range(3):
        p = np.abs(normal * candidates[i])
        if p < min:
            min = p
            cIndex = i
    result = orthogonalize(normal, candidates[cIndex])
    return result.normalized()


class BoomGan():


    def __init__(self,
                 network_pkl="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl",
                 audio_filename="shiko_short.mp3", truncation_psi=1):
        in_dir = "in"
        self.out_dir = "out"
        self.out = os.path.join(self.out_dir, "video.mp4")
        self.input = os.path.join(in_dir, audio_filename)
        self.psi = truncation_psi
        self.fps = 24
        self.batch_size = 10

        # load audio
        self.audio = librosa.load(self.input)[0]
        self.audio_duration = librosa.get_duration(self.audio)
        self.total_frames = int(np.ceil(self.fps * self.audio_duration))
        bpm, self.beats = librosa.beat.beat_track(self.audio, units="time")
        # add first and last frame as beat
        self.beats = np.insert(self.beats, 0, 0)
        self.beats = np.insert(self.beats, -1, self.audio_duration)

        # load network
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore

        # setup output
        os.makedirs(self.out_dir, exist_ok=True)

        # generate latent
        z = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
        z = torch.ones_like(z) / 2

        self.strategies = {}

        def add_strat(func):
            self.strategies[func.__name__] = func

        @add_strat
        def random_walk(*args, stretch=3, **kwargs):
            # generate num_t latent vecs
            latents = np.random.randn(len(self.beats), self.G.z_dim) * stretch
            latents = np.cumsum(latents, axis=0)
            return latents

    def gen_batch(self, latent):
        # generate batch
        ws = self.G.mapping(z=latent, c=None, truncation_psi=self.psi)
        img = self.G.synthesis(ws)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # img = torch.cumsum(img, dim=0)
        return img

    def gen_latent(self, mode="random_walk"):
        latents = self.strategies[mode]()
        breakpoint()
        interpol_f = interpolate.interp1d(self.beats, latents, axis=0)
        interpolated_latents = interpol_f(np.linspace(0, self.audio_duration, self.total_frames))
        interpolated_latents = torch.from_numpy(interpolated_latents).to(self.device)

        return interpolated_latents

    def gen_video(self):
        # gen dataset
        latent_interpol = self.gen_latent()
        print("Computing GAN images...")
        data_loader = torch.utils.data.DataLoader(latent_interpol, batch_size=self.batch_size)
        batches = []

        for i, latent in tqdm(enumerate(data_loader), total=len(data_loader)):
            batches.append(self.gen_batch(latent).cpu())

        frames = torch.cat(batches, axis=0).numpy()
        self.write_video(frames)

    def write_video(self, frames, vcodec='libx264'):
        print("Rendering Video...")
        # render video alone
        n, height, width, channels = frames.shape
        exact_fps = int(n / self.audio_duration)
        temp_vidpath = os.path.join(self.out_dir, "only_vid.mp4")
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=exact_fps)
                .output(temp_vidpath, pix_fmt='yuv420p', vcodec=vcodec, r=exact_fps)
                .global_args('-y')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        for frame in frames:
            process.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )
        process.stdin.close()
        process.wait()

        # add audio and save
        audio = ffmpeg.input(self.input)

        video = ffmpeg.input(temp_vidpath)
        ffmpeg.concat(video, audio, v=1, a=1).output(self.out, pix_fmt='yuv420p', vcodec=vcodec,
                                                     r=exact_fps).global_args('-y').run()


if __name__ == "__main__":
    bg = BoomGan()
    bg.gen_video()
