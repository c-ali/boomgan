import os, sys

sys.path.append("stylegan3")

import torch
import numpy as np
from numpy.linalg import norm
from stylegan3 import dnnlib
from stylegan3.legacy import load_network_pkl
from scipy import interpolate
from tqdm import tqdm
import librosa
import warnings
import ffmpeg
import click
import ast
from util.geometry import make_orthonormal_vector, random_circle

warnings.filterwarnings("ignore", category=Warning)


class BoomGan:
    def __init__(self, network_pkl, audio_file, truncation_psi, in_dir, out_dir, mode, stretch, offset, base_eq, pulse_eq):
        self.out_dir = out_dir
        self.out = os.path.join(self.out_dir, "video.mp4")
        self.input = os.path.join(in_dir, audio_file)
        self.psi = truncation_psi
        self.fps = 24
        self.batch_size = 20
        self.stretch = stretch
        self.mode = mode
        self.first_batch = None
        self.chroma_bins = 12
        self.offset = offset
        self.base_eq = torch.Tensor(ast.literal_eval(base_eq))
        self.pulse_eq = torch.Tensor(ast.literal_eval(pulse_eq))
        
        # load audio
        self.audio, sample_rate = librosa.load(self.input)
        self.audio_duration = librosa.get_duration(self.audio)
        self.total_frames = int(np.ceil(self.fps * self.audio_duration))

        # process audio
        bpm, self.beats = librosa.beat.beat_track(self.audio, units="time")
        print("BMP read: %f" % bpm)
        self.chroma = librosa.feature.chroma_stft(self.audio, sr=sample_rate, n_chroma=self.chroma_bins)

        # add first and last frame as beat
        self.beats = np.insert(self.beats, 0, 0)
        self.beats = np.insert(self.beats, -1, self.audio_duration)

        # load network
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = load_network_pkl(f)['G_ema'].to(self.device)

        # setup output
        os.makedirs(self.out_dir, exist_ok=True)

        # define latent space exploration strategies
        self.strategies = {}

        def add_strat(func):
            self.strategies[func.__name__] = func

        @add_strat
        def rwalk(*args, stretch=3, **kwargs):
            # random walk in latent space
            # generate num_t latent vecs
            latents = np.random.randn(len(self.beats), self.G.z_dim) * stretch
            latents[0] = self.G.mapping.w_avg.cpu().numpy()
            latents = np.cumsum(latents, axis=0)
            return latents

        @add_strat
        def orwalk(*args, stretch=3, **kwargs):
            # orthogonal random walk in latent space
            # initialize first two points, start from w_avg
            latents = [self.G.mapping.w_avg.cpu().numpy()]
            rand_dir = np.random.randn(self.G.z_dim)
            rand_dir /= np.linalg.norm(rand_dir)
            latents.append(latents[0] + rand_dir * stretch)
            for i in range(2, len(self.beats)):
                # make new rand dir orthogonal to old one
                rand_dir = make_orthonormal_vector(rand_dir, self.G.z_dim)
                latents.append(latents[-1] + rand_dir * stretch)
            return np.array(latents)

        @add_strat
        def rjump(*args, stretch=3, **kwargs):
            # randomly jumps between points in latent space
            return np.random.randn(len(self.beats), self.G.z_dim) * stretch

        @add_strat
        def circ(*args, radius=1, **kwargs):
            # walk around in a circle
            c = random_circle(radius, self.G.z_dim)
            x = np.linspace(0, 2 * np.pi, len(self.beats))
            return c(x)

        @add_strat
        def twocirc(*args, inner_rad=1, outer_rad=1, stretch=5, offset=self.offset, **kwargs):
            # walk around in a circle
            c1 = random_circle(inner_rad, self.G.z_dim)
            c2 = random_circle(outer_rad, self.G.z_dim)
            half = int(len(self.beats) / 2)
            x1 = np.linspace(0, stretch * 2 * np.pi, half) + offset
            x2 = np.linspace(0, stretch * 2 * np.pi, len(self.beats) - half)
            y1 = c1(x1)
            y2 = c2(x2)
            y = np.empty(shape=(len(self.beats), self.G.z_dim))
            y[::2] = y2
            y[1::2] = y1
            return y

    def gen_batch(self, latent):
        # generate batch
        latent_base = latent[:, :, 0]
        latent_pulse = latent[:, :, 1]
        ws_base = self.G.mapping(z=latent_base, c=None, truncation_psi=self.psi)
        ws_pulse = self.G.mapping(z=latent_pulse, c=None, truncation_psi=self.psi)
        # use mask from equalizer
        base_mask = self.base_eq.repeat_interleave(4).expand(0,2)
        pulse_mask = self.pulse_eq.repeat_interleave(4).expand(0,2)
        #save first frame
        if self.first_batch is None:
            self.first_batch = ws_base[0].repeat(ws_base.shape[0], 1)
        ws_combined = ws_base * base_mask + ws_pulse * pulse_mask + (1-base_mask-pulse_mask) * self.first_batch
        img = self.G.synthesis(ws_combined)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # img = torch.cumsum(img, dim=0) # uncomment for some trippy shit
        return img

    def gen_latent_base(self):
        # creates base movement in latent space
        latents = self.strategies[self.mode](stretch=self.stretch)
        interpolated_latents = self.interpolate_latents(self.beats, latents)
        return interpolated_latents

    def gen_latent_pulse(self):
        # normalize hist
        self.chroma /= np.linalg.norm(self.chroma, axis=0, keepdims=True)
        # creates a pulsating movement that shifts with the pitch of the music
        latents = np.repeat(self.chroma, np.ceil(512 / self.chroma_bins), axis=0)[:512]
        latents = np.swapaxes(latents, 0, 1)
        interpolated_latents = self.interpolate_latents(np.linspace(0, self.audio_duration, latents.shape[0]), latents)
        return interpolated_latents

    def interpolate_latents(self, x, latents):
        # interpolates a set of latent vector to the beat
        interpol_f = interpolate.interp1d(x, latents, axis=0)
        interpolated_latents = interpol_f(np.linspace(0, self.audio_duration, self.total_frames))
        interpolated_latents = torch.from_numpy(interpolated_latents).to(self.device)
        return interpolated_latents

    def gen_video(self):
        # gen dataset
        latent_base = self.gen_latent_base()
        latent_pulse = self.gen_latent_pulse()
        full_latent = torch.stack([latent_base, latent_pulse], axis=-1)
        print("Computing GAN images...")
        data_loader = torch.utils.data.DataLoader(full_latent, batch_size=self.batch_size)
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


@click.command()
@click.option('--network', 'network_pkl',
              default="https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl",
              help='Network pickle filename', required=True)
@click.option('--audio_file', help='Filename of the audio file to use', type=str, required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--out_dir', help='Where to save the output images', default="out", type=str, required=True,
              metavar='DIR')
@click.option('--in_dir', help='Location of the input images', default="in", type=str, required=True, metavar='DIR')
@click.option('--mode', help='Latent space vector mode. [rjump/rwalk/orwalk/twocirc]', default="rjump", type=str,
              required=True)
@click.option('--stretch', help='How much distortion there is', default=5, type=int, required=True)
@click.option('--base_eq', 'base_eq', help='Contribution of beat movement in style bins. Example: 0.1,0.4,0.1,0', default="0,1,1,0", type=str,
              required=True)
@click.option('--base_eq', 'base_eq', help='Contribution of chroma movement in style bins. Example: 0.1,0.4,0.1,0', default="0,0,0,1", type=str,
              required=True)
@click.option('--offset', 'offset', help='Angular offset for the inner circle', default=np.pi/16, type=click.FloatRange(0,2*np.pi),
              required=True)

def run(network_pkl, audio_file, truncation_psi, in_dir, out_dir, mode, stretch, offset, base_eq, pulse_eq):
    bg = BoomGan(network_pkl, audio_file, truncation_psi, in_dir, out_dir, mode, stretch, offset, base_eq, pulse_eq)
    bg.gen_video()


if __name__ == "__main__":
    run()
