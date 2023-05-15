
pip install -r requirements.txt -i https://mirrors-ssl.aliyuncs.com/pypi/simple

cd installs

cd transformers-install
pip install -e . -i https://mirrors-ssl.aliyuncs.com/pypi/simple
cd -

cd peft-install
pip install -e . -i https://mirrors-ssl.aliyuncs.com/pypi/simple
cd -

cd apex-install
pip install -e . -i https://mirrors-ssl.aliyuncs.com/pypi/simple
cd -

cd deepspeed-install
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
cd -
cd -
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 -i https://mirrors-ssl.aliyuncs.com/pypi/simple
