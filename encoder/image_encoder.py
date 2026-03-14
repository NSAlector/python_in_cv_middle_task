from __future__ import annotations

import math
import random
import subprocess
import tempfile
import zlib
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class ImageDatasetEncoder:
    """Encodes images from a dataset into integer keys."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(
        self,
        dataset_dir: str | Path,
        threshold: float = 0.5,
        noise_percent: float = 20.0,
        seed: int = 42,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        self.threshold = threshold
        self.noise_percent = noise_percent
        self.seed = seed
        self.paths = sorted(
            path
            for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported image files found in {self.dataset_dir}")

        self._cache: dict[int, tuple[int, tuple[int, ...]]] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> int:
        if index < 0 or index >= len(self.paths):
            raise IndexError(f"Index out of range: {index}")

        if index not in self._cache:
            self._cache[index] = self._encode_file(index)
        encoded_int, _ = self._cache[index]
        return encoded_int

    def get_bits(self, index: int) -> tuple[int, ...]:
        if index < 0 or index >= len(self.paths):
            raise IndexError(f"Index out of range: {index}")

        if index not in self._cache:
            self._cache[index] = self._encode_file(index)
        _, bits = self._cache[index]
        return bits

    def encode_all(self, as_bits: bool = False) -> list[int] | list[tuple[int, ...]]:
        if as_bits:
            return [self.get_bits(i) for i in range(len(self))]
        return [self[i] for i in range(len(self))]

    def _encode_file(self, index: int) -> tuple[int, tuple[int, ...]]:
        path = self.paths[index]
        image = self._load_image_as_grayscale(path)
        patches = self._split_into_patches(image)
        autocorr = self._autocorrelate_patches(patches)
        bits = [1 if value >= self.threshold else 0 for value in autocorr]
        bits = self._apply_noise(bits, index=index)
        encoded_int = self._bits_to_int(bits)
        return encoded_int, tuple(bits)

    @staticmethod
    def _load_image_as_grayscale(path: Path) -> list[float]:
        if Image is not None:
            with Image.open(path) as image:
                resized = image.convert("L").resize((256, 256))
                return [pixel / 255.0 for pixel in resized.getdata()]

        with tempfile.TemporaryDirectory(prefix="encoder_tmp_") as temp_dir:
            output_dir = Path(temp_dir)
            command = [
                "qlmanage",
                "-t",
                "-s",
                "512",
                "-o",
                str(output_dir),
                str(path),
            ]
            run = subprocess.run(command, capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(
                    f"Failed to preprocess image with qlmanage: {path}\n"
                    f"stdout: {run.stdout}\nstderr: {run.stderr}"
                )
            thumbnails = sorted(output_dir.glob("*.png"))
            if not thumbnails:
                raise RuntimeError(f"qlmanage did not produce a thumbnail for {path}")
            width, height, rgb_pixels = ImageDatasetEncoder._read_png_rgb(thumbnails[0])
            return ImageDatasetEncoder._resize_and_grayscale(rgb_pixels, width, height)

    @staticmethod
    def _read_png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
        data = path.read_bytes()
        signature = b"\x89PNG\r\n\x1a\n"
        if data[:8] != signature:
            raise ValueError(f"File is not a PNG image: {path}")

        width = height = bit_depth = color_type = interlace = None
        palette: list[tuple[int, int, int]] | None = None
        compressed = bytearray()
        cursor = 8
        while cursor < len(data):
            length = int.from_bytes(data[cursor : cursor + 4], "big")
            chunk_type = data[cursor + 4 : cursor + 8]
            chunk_data = data[cursor + 8 : cursor + 8 + length]
            cursor += 12 + length

            if chunk_type == b"IHDR":
                width = int.from_bytes(chunk_data[0:4], "big")
                height = int.from_bytes(chunk_data[4:8], "big")
                bit_depth = chunk_data[8]
                color_type = chunk_data[9]
                interlace = chunk_data[12]
            elif chunk_type == b"PLTE":
                palette = [
                    (chunk_data[i], chunk_data[i + 1], chunk_data[i + 2])
                    for i in range(0, len(chunk_data), 3)
                ]
            elif chunk_type == b"IDAT":
                compressed.extend(chunk_data)
            elif chunk_type == b"IEND":
                break

        if None in (width, height, bit_depth, color_type, interlace):
            raise ValueError(f"PNG metadata is incomplete: {path}")
        if interlace != 0:
            raise ValueError(f"Interlaced PNG is not supported: {path}")

        samples_per_pixel = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
        if samples_per_pixel is None:
            raise ValueError(f"Unsupported PNG color type {color_type}: {path}")

        bits_per_pixel = bit_depth * samples_per_pixel
        line_size = (width * bits_per_pixel + 7) // 8
        filter_bpp = max(1, (bits_per_pixel + 7) // 8)

        raw = zlib.decompress(bytes(compressed))
        expected_len = (line_size + 1) * height
        if len(raw) != expected_len:
            raise ValueError(f"PNG raw data length mismatch for {path}")

        rows: list[bytes] = []
        prev_row = bytes(line_size)
        pos = 0
        for _ in range(height):
            filter_type = raw[pos]
            pos += 1
            filtered = raw[pos : pos + line_size]
            pos += line_size
            recon = bytearray(line_size)
            for i in range(line_size):
                left = recon[i - filter_bpp] if i >= filter_bpp else 0
                up = prev_row[i]
                up_left = prev_row[i - filter_bpp] if i >= filter_bpp else 0

                if filter_type == 0:
                    predictor = 0
                elif filter_type == 1:
                    predictor = left
                elif filter_type == 2:
                    predictor = up
                elif filter_type == 3:
                    predictor = (left + up) // 2
                elif filter_type == 4:
                    predictor = _paeth_predictor(left, up, up_left)
                else:
                    raise ValueError(f"Unsupported PNG filter type {filter_type}: {path}")

                recon[i] = (filtered[i] + predictor) & 0xFF

            row = bytes(recon)
            rows.append(row)
            prev_row = row

        pixels: list[tuple[int, int, int]] = []
        for row in rows:
            pixels.extend(_decode_png_row(row, width, bit_depth, color_type, palette))

        return width, height, pixels

    @staticmethod
    def _resize_and_grayscale(
        rgb_pixels: list[tuple[int, int, int]],
        width: int,
        height: int,
    ) -> list[float]:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")

        target = 256
        grayscale: list[float] = []
        for y in range(target):
            src_y = min(height - 1, int(y * height / target))
            for x in range(target):
                src_x = min(width - 1, int(x * width / target))
                red, green, blue = rgb_pixels[src_y * width + src_x]
                value = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
                grayscale.append(value)
        return grayscale

    @staticmethod
    def _split_into_patches(image: list[float]) -> list[list[float]]:
        if len(image) != 256 * 256:
            raise ValueError("Image must be 256x256 after preprocessing.")

        patches: list[list[float]] = []
        patch_size = 64
        for patch_row in range(4):
            for patch_col in range(4):
                patch: list[float] = []
                row_start = patch_row * patch_size
                col_start = patch_col * patch_size
                for row_offset in range(patch_size):
                    row_idx = row_start + row_offset
                    start = row_idx * 256 + col_start
                    patch.extend(image[start : start + patch_size])
                patches.append(patch)
        return patches

    @staticmethod
    def _autocorrelate_patches(patches: list[list[float]]) -> list[float]:
        if len(patches) != 16:
            raise ValueError("Expected exactly 16 patches.")

        norms = [math.sqrt(sum(value * value for value in patch)) for patch in patches]
        result: list[float] = []
        for i in range(16):
            for j in range(16):
                norm_product = norms[i] * norms[j]
                if norm_product == 0:
                    corr = 0.0
                else:
                    dot = sum(a * b for a, b in zip(patches[i], patches[j]))
                    corr = dot / norm_product
                result.append(corr)
        return result

    def _apply_noise(self, bits: list[int], index: int) -> list[int]:
        if not bits:
            return bits
        flips = int(round(len(bits) * self.noise_percent / 100.0))
        flips = max(0, min(flips, len(bits)))
        if flips == 0:
            return bits

        rng = random.Random(self.seed + index)
        for pos in rng.sample(range(len(bits)), k=flips):
            bits[pos] = 1 - bits[pos]
        return bits

    @staticmethod
    def _bits_to_int(bits: list[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png_row(
    row: bytes,
    width: int,
    bit_depth: int,
    color_type: int,
    palette: list[tuple[int, int, int]] | None,
) -> list[tuple[int, int, int]]:
    if bit_depth not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported PNG bit depth: {bit_depth}")

    if color_type == 0:
        samples = _extract_samples(row, bit_depth, width)
        max_sample = (1 << bit_depth) - 1
        return [(_to_u8(value, max_sample),) * 3 for value in samples]

    if color_type == 2:
        samples = _extract_samples(row, bit_depth, width * 3)
        return [
            (
                _to_u8(samples[i], (1 << bit_depth) - 1),
                _to_u8(samples[i + 1], (1 << bit_depth) - 1),
                _to_u8(samples[i + 2], (1 << bit_depth) - 1),
            )
            for i in range(0, len(samples), 3)
        ]

    if color_type == 3:
        if palette is None:
            raise ValueError("Palette PNG has no PLTE chunk.")
        indices = _extract_samples(row, bit_depth, width)
        return [palette[index] for index in indices]

    if color_type == 4:
        samples = _extract_samples(row, bit_depth, width * 2)
        max_sample = (1 << bit_depth) - 1
        return [(_to_u8(samples[i], max_sample),) * 3 for i in range(0, len(samples), 2)]

    if color_type == 6:
        samples = _extract_samples(row, bit_depth, width * 4)
        max_sample = (1 << bit_depth) - 1
        return [
            (
                _to_u8(samples[i], max_sample),
                _to_u8(samples[i + 1], max_sample),
                _to_u8(samples[i + 2], max_sample),
            )
            for i in range(0, len(samples), 4)
        ]

    raise ValueError(f"Unsupported PNG color type: {color_type}")


def _extract_samples(row: bytes, bit_depth: int, sample_count: int) -> list[int]:
    if bit_depth == 8:
        return list(row[:sample_count])

    if bit_depth == 16:
        samples = []
        for idx in range(sample_count):
            offset = idx * 2
            sample = (row[offset] << 8) | row[offset + 1]
            samples.append(sample)
        return samples

    mask = (1 << bit_depth) - 1
    samples = []
    for byte in row:
        shift = 8 - bit_depth
        while shift >= 0 and len(samples) < sample_count:
            samples.append((byte >> shift) & mask)
            shift -= bit_depth
        if len(samples) >= sample_count:
            break
    return samples


def _to_u8(value: int, max_sample: int) -> int:
    if max_sample == 255:
        return value
    if max_sample == 65535:
        return value >> 8
    return int(round((value / max_sample) * 255))
