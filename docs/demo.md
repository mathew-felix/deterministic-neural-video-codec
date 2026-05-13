# Recruiter Demo

This demo shows the codec result on the local `video.mp4` validation clip.

## Visual Comparison

![Original versus decoded reconstruction](../assets/demo_before_after.gif)

Left: original `video.mp4`  
Right: decoded reconstruction from the deterministic INT16 codec bitstream

For portfolios or LinkedIn posts, use the smaller MP4 version:

```text
assets/demo_before_after.mp4
```

## Compression Result

| Artifact | Size |
|---|---:|
| Original MP4 | 84,089,033 bytes, 80.19 MiB |
| Deterministic codec bitstream | 4,817,350 bytes, 4.59 MiB |
| Decoded preview MP4 | 39,068,735 bytes, 37.26 MiB |

The codec bitstream is the compressed output that should be compared against
the original input size.

| Metric | Value |
|---|---:|
| Size reduction versus original MP4 | 94.27% |
| Compression ratio versus original MP4 | 17.45x smaller |
| Full-video frames tested | 2593 |
| Resolution | 1280x720 |
| FPS | 30 |
| Average RGB PSNR | 28.8969 dB |
| Average RGB MS-SSIM | 0.933624 |

## Demo Commands

Create the GIF:

```powershell
ffmpeg -y `
  -ss 00:00:10 -t 6 -i video.mp4 `
  -ss 00:00:10 -t 6 -i outputs\video_full\video_1280x720_30_2593f_q32_decoded.mp4 `
  -filter_complex "[0:v]scale=480:-2,setpts=PTS-STARTPTS[left];[1:v]scale=480:-2,setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2,fps=12,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" `
  assets\demo_before_after.gif
```

Create the smaller MP4 demo:

```powershell
ffmpeg -y `
  -ss 00:00:10 -t 8 -i video.mp4 `
  -ss 00:00:10 -t 8 -i outputs\video_full\video_1280x720_30_2593f_q32_decoded.mp4 `
  -filter_complex "[0:v]scale=640:-2,setpts=PTS-STARTPTS[left];[1:v]scale=640:-2,setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2" `
  -c:v libx264 -crf 20 -preset fast `
  assets\demo_before_after.mp4
```

## Short Pitch

This project converts a DCVC-RT-family FP16 neural video codec workflow into a
deterministic INT16 pipeline. The important output is not only a decoded video;
it is a compact `.bin` bitstream whose identity can be verified with SHA-256 for
reproducible codec experiments.
