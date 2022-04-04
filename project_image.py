import sys

sys.path.append("stylegan3")

import PIL.Image
from stylegan3 import dnnlib
from stylegan3.legacy import load_network_pkl
import click
import os
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image
import numpy as np


class LatentProjector():
    def __init__(self, network_pkl, image_file, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.device = torch.device('cuda')
        input = os.path.join(in_dir, image_file)
        self.out = os.path.join(self.out_dir, "latents_%s.txt" % os.path.splitext(image_file)[0])
        self.total_epochs = 5000
        self.save_sample_every = 10
        max_lr = 2

        # io stuff
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = load_network_pkl(f)['G_ema'].to(self.device)
        # freeze network
        for param in self.G.parameters():
            param.requires_grad = False

        self.image = Image.open(input)
        self.image = self.image.resize((self.G.img_resolution, self.G.img_resolution))
        self.image = torch.Tensor(np.asarray(self.image)).to(self.device)
        print(self.image.shape)
        self.image = self.image.permute(2,0,1).unsqueeze(0)
        self.image = (self.image - 128) / 127.5
        # setup training stuff
        self.loss = nn.MSELoss()
        self.latents = torch.randn((1, 512)).to(self.device)
        #self.latents = torch.randn((1,16,512)).to(self.device)
        self.latents.requires_grad = True
        self.optimizer = torch.optim.Adam([self.latents], lr=max_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, epochs=self.total_epochs,
                                                             steps_per_epoch=1)

    def project_image(self):
        print("Projecting into latent space...")
        for i in tqdm(range(self.total_epochs)):
            self.optimizer.zero_grad()
            generated_image = self.G(self.latents, c=None)
            #generated_image = self.G.synthesis(self.latents)
            loss = self.loss(self.image, generated_image)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if i % self.save_sample_every == 0:
                out_img = (generated_image.detach() * 127.5 + 128).squeeze(0)
                out_img = out_img.clamp(0, 255).permute(1,2,0)
                out_img = out_img.to(torch.uint8)
                out_img = Image.fromarray(out_img.cpu().numpy())
                out_img.save(os.path.join(self.out_dir, "projected_img.jpg"))
                print(self.latents.abs().mean())
                print("Loss: %.3f" % loss)


        print("Saving latents...")
        # save progress
        out = "".join([str(p) + " " for p in self.latents])
        with open(self.out, 'w') as f:
            f.write(out)
            f.close()


@click.command()
@click.option('--network', 'network_pkl',
              default='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
              help='Network pickle filename', required=True)
@click.option('--image_file', help='Filename of the audio file to use', type=str, required=True)
@click.option('--out_dir', help='Where to save the output images', default="out", type=str, required=True,
              metavar='DIR')
@click.option('--in_dir', help='Location of the input images', default="in", type=str, required=True, metavar='DIR')
def run(network_pkl, image_file, in_dir, out_dir):
    bg = LatentProjector(network_pkl, image_file, in_dir, out_dir)
    bg.project_image()


if __name__ == "__main__":
    run()
