from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR,'extended-dse-meshing/data/test_data')
OUTPUT_FOLDER = os.path.join(BASE_DIR,'extended-dse-meshing/data/test_data/select')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def run_scripts(input_file: str):
    """Runs multiple Python scripts sequentially and ensures they complete before continuing."""
    scripts = ["./extended-dse-meshing/logmap_estimation/eval_network.py",  
 "./extended-dse-meshing/logmap_alignment/align_patches.py",  
 "./extended-dse-meshing/logmap_alignment/eval_align.py",  
 "./extended-dse-meshing/triangle_selection/select.py"]

    
    for script in scripts:
        script_path = os.path.join(os.getcwd(), script)
        
        if os.path.exists(script_path):  # Ensure script exists
            try:
                print(f"Starting {script}...", flush=True)
                subprocess.run(["python", script_path, input_file], check=True)  # Blocking call
                print(f"✅ Completed {script}", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error executing {script}: {e}", flush=True)
                return False  # Stop execution if any script fails
        else:
            print(f"⚠️ Skipping {script}, file not found", flush=True)

    print("✅ All scripts executed successfully!", flush=True)
    return True  # All scripts executed successfully

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    print("🔄 Receiving file...", flush=True)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"✅ File saved: {file_path}", flush=True)

    # Run scripts and wait for completion
    success = run_scripts(file_path)
    
    if not success:
        return {"error": "Processing failed"}
    
   # Find the generated output file (final_mesh_<filename>)
    output_filename = f"final_mesh_{file.filename.replace('.xyz', '.ply')}"
    output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Ensure the file exists before returning
    if not os.path.exists(output_file_path):
        print(f"❌ Error: Expected output file not found - {output_file_path}", flush=True)
        return {"error": "Output file not found"}

    print(f"✅ Output file generated: {output_file_path}", flush=True)

    return {"filename": os.path.basename(output_file_path)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    print(f"📥 Download requested: {filename}", flush=True)
    
    return {"url": f"http://127.0.0.1:8000/static/{filename}"}

# Serve static files for downloads
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=OUTPUT_FOLDER), name="static")


@app.get("/output")
async def get_output():
    # Find the latest generated PLY file
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".ply")]
    if not files:
        return {"error": "No output files found"}

    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(OUTPUT_FOLDER, f)))
    file_path = os.path.join(OUTPUT_FOLDER, latest_file)

    return FileResponse(file_path, filename=latest_file)