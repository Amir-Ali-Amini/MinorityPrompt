pip install -r requirements.txt

mkdir -p ckpt && wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_unet.safetensors -O ckpt/sdxl_lightning_4step_unet.safetensors

rm -rf ./models
mkdir -p models
# Remove wrong file
rm -f models/shape_predictor_5_face_landmarks.dat

# Download from official dlib.net
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 -O models/shape_predictor_5_face_landmarks.dat.bz2
bzip2 -d models/shape_predictor_5_face_landmarks.dat.bz2

# Download FairFace model
gdown "1fUJSLseDpgilArB_YKep9PnsR7QrPW5I" -O models/res34_fair_align_multi_7_20190809.pt

# # Download dlib shape predictor
# !gdown "11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB" -O models/shape_predictor_5_face_landmarks.dat

# Verify
ls -lh models/
