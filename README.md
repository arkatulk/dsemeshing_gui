# [Learning Delaunay Surface Elements for Mesh Reconstruction (DSE meshing)](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/dse_meshing.html)
This is our implementation of the paper ["Learning Delaunay Surface Elements for Mesh Reconstruction"](https://arxiv.org/abs/2012.01203) at CVPR 2021 (oral), a method for mesh recontruction from a point cloud.


![DSE_meshing](img/dse_meshing_teaser.png "DSE meshing")


The triangle selection code is based on  on [Meshing-Point-Clouds-with-IER](https://github.com/Colin97/Point2Mesh) (with a few smaller modifications) that also uses code from the project [annoy](https://github.com/spotify/annoy). Our network code is based on [PointNet](https://github.com/charlesq34/pointnet).

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 3.12
* Pytorch 2.6

## Setup
Install required python packages, if they are not already installed:
``` bash
pip install numpy
pip install scipy
pip install trimesh
```


Clone this repository:
``` bash
git clone https://github.com/arkatulk/dsemeshing_gui.git
cd dsemeshing_gui/backend/extended-dse-meshing
```

Setup the triangle selection step:
``` bash
git submodule update --init --recursive
cd triangle_selection/postprocess
mkdir build
cd build
cmake ..
make
```



 ## Data


- **Training set:** We store our training data in  .npy files. The files contain elements of size N_NEIGHBORSx5 where the first 3 columns contain the 3D coordinates and the last two columns contain the ground truth logmap 2D coordinates. 
- **Pre-trained models** on the famousthingi dataset.
- **Testset** from famousthingi dataset.


## Training
To train the classifier network on the provided dataset:
``` bash
cd train_logmap
python train_classifier.py
```

To train the logmap estimation network on the provided dataset:
``` bash
cd train_logmap
python train_logmap.py
```

- The trained models are saved in `log/log_famousthingi_classifier` and `log/log_famousthingi_logmap` by default. Paths for the training set, output and training parameters can be changed directly in the code.



## Testing
Our testing pipeline has 3 main steps:
1.  **Logmap estimation:** we locally predict the logmap that contains neighboring points at each point using the trained networks.
2. **Logmap alignment:** we locally align the log maps to one another to ensure better consistency.
3. **Triangle selection:** we select the produced triangles to generate an almost manifold mesh.

We provide one example point cloud in `data/test_data`. For easily evaluating your pointclouds or the pointclouds from the testset, move the .xyz files there.

### 1. Logmap estimation

To run the logmap estimation networks on the point clouds in data/test_data directory:
``` bash
cd logmap_estimation
python eval_network.py
```

### 2. Logmap alignment

To align the logmap patches from step 1 and compute the appearance frequency for step 3:
``` bash
cd logmap_alignment
python align_patches.py
python eval_align_meshes.py
```

### 3. Triangle selection

To apply triangle selection on the output from step 2:
``` bash
cd triangle_selection
python select.py
```
The produced meshes can be found in `data/test_data/select` in the format: `final_mesh_(SHAPE_NAME).ply`. For numerical precision reasons we suggest to use point clouds with a bounding box diagonal larger than 1. By default our code evaluates the .xyz point clouds in `data/test_data` please adapt the paths in the code if you wish to evaluate  point clouds at different locations.



---

🚀 Running the Project
🔧 Backend (Uvicorn + FastAPI)


1. Install dependencies
bash
Copy
Edit
cd backend
pip install -r requirements.txt
2. Run the backend
bash
Copy
Edit
uvicorn main:app --reload
The API will be available at http://localhost:8000


💻 Frontend (React + Vite)
The frontend visualizes generated .ply files.

1. Setup
bash
Copy
Edit
cd frontend
npm install
2. Run dev server
bash
Copy
Edit
npm run dev
Then open http://localhost:5173

Now upload the .xyz files using the upload button


📦 Dataset & Pretrained Models
Download links (Google Drive or Hugging Face):

📁 Training Dataset: Download

🤖 Pre-trained Models: Download

📸 Demo
![DSE_meshing](img/demo_image1.png "Demo Image 1")
![DSE_meshing](img/demo_image2.png "Demmo Image 2")
