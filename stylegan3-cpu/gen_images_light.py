import click
import glob
import hashlib
import html
import io
import os
import pickle
import re
import tempfile
import time
import urllib
import urllib.parse
import urllib.request
import uuid
from math import ceil
from typing import Any, List, Tuple, Union

import PIL.Image
import numpy as np
import requests
import requests.compat
import torch


class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)


dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global dnnlib_cache_dir
    dnnlib_cache_dir = path


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    if not isinstance(obj, str) or "://" not in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or "." not in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or "." not in res.netloc:
            return False
    except (Exception,):
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True,
             return_filename: bool = False, cache: bool = True) -> Any:
    assert num_attempts >= 1
    assert not (return_filename and (not cache))
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")
    assert is_url(url)
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")
                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if
                                     "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except (Exception,):
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)
        if return_filename:
            return cache_file
    assert not return_filename
    return io.BytesIO(url_data)


def make_cache_dir_path(*paths: str) -> str:
    if dnnlib_cache_dir is not None:
        return os.path.join(dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)


def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='models/network.pkl', show_default=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0',
              show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--scale', 'scale', help='Scale of image', type=float, default=1, show_default=True)
@click.option('--cols', 'cols', help='Number of colums', type=float, default=10, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, default='out', show_default=True, metavar='DIR')
def generate_images(
        network_pkl: str,
        seeds: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        translate: Tuple[float, float],
        rotate: float,
        cols: float,
        scale: float
):
    print('Loading networks from "%s"...' % network_pkl)
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print('cuda is available.')
        device = torch.device('cuda')
    else:
        print('cuda is not available.')
        device = torch.device('cpu')
    print('device: "%s"' % device)
    with open_url(network_pkl) as f:
        G = LegacyUnpickler(f).load()
        if 'training_set_kwargs' not in G:
            G['training_set_kwargs'] = None
        if 'augment_pipe' not in G:
            G['augment_pipe'] = None
        assert isinstance(G['G'], torch.nn.Module)
        assert isinstance(G['D'], torch.nn.Module)
        assert isinstance(G['G_ema'], torch.nn.Module)
        assert isinstance(G['training_set_kwargs'], (dict, type(None)))
        assert isinstance(G['augment_pipe'], (torch.nn.Module, type(None)))
        G = G['G_ema'].to(device)
    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)
    start_time = time.time()
    images = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        if cuda_avail:
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        images.append(img)
    print("total %s seconds, %sit/s, %ss/it" % (
        (time.time() - start_time), len(images) / (time.time() - start_time),
        (time.time() - start_time) / len(images)))
    w, h = images[0].size
    w = int(w * scale)
    h = int(h * scale)
    rows = ceil(len(images) / cols)
    width = cols * w
    height = rows * h
    canvas = PIL.Image.new('RGBA', (int(width), int(height)), 'white')
    for i, img in enumerate(images):
        img = img.resize((w, h), PIL.Image.LANCZOS)
        canvas.paste(img, (int(w * (i % cols)), int(h * (i // cols))))
    canvas.save(f'{outdir}/result.png')


if __name__ == "__main__":
    generate_images()
