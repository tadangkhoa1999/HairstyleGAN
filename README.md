# HairstyleGAN

HairstyleGAN is Graduation Assignment in FPT University of Ta Dang Khoa and Pham Cao Bang

## Note
I have updated from using [stylegan](https://github.com/NVlabs/stylegan) to using [
stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). But I have not updated the stylegan-encoder yet. You can use older commit for better result.

## Directory layout
```
.
├── configs         # store all configs
├── core            # model, loss, optimizer, metrics, dataloader, ...
├── data
├── docs
├── exp             # weights, processed data features, ...
│   ├── features
│   └── weights
├── modules         # pre-process, post-process, ...
├── service         # REST API app
├── tools           # scripts for train, test, ...
└── utils
```

## Generate data
```
python tools/generate.py --outdir=data/raw_images/ --trunc=1 --seeds=1-100 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

## Align images
```
mkdir exp/features/aligned_images
mkdir exp/features/landmarks
mkdir exp/features/aligned_landmarks
python tools/align_images.py data/raw_images/ exp/features/aligned_images/ --landmarks_path=exp/features/landmarks/ --aligned_landmarks_path=exp/features/aligned_landmarks/
```

## Merge hair
```
mkdir exp/features/merge_hair_images
python tools/merge_hair.py --face_dir exp/features/aligned_images/ --hair_dir data/hairs/ --merge_hair_dir exp/features/merge_hair_images/ --face_landmark_dir exp/features/aligned_landmarks/ --hair_landmark_dir data/hair_landmarks/
```

## Generate merge hair-face images
```
python tools/generate_merge_hair_images.py --input_dir exp/features/merge_hair_images/ --output_dir exp/results/
```

## Result
![rs.jpg](docs/rs.jpg)

## License

MIT
