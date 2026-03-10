export CUDA_HOME=$CONDA_PREFIX
pip install uv
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-build-isolation --index-strategy unsafe-best-match -r requirements.txt
uv pip install --no-build-isolation -e vggt/
uv pip install --no-build-isolation -e editing/
uv pip install --no-build-isolation -e videogen/
cd third-party/GroundingDINO && python setup.py build && python setup.py install && cd ../..
uv pip install --no-build-isolation -e third-party/sam2