import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import * as THREE from "three";
import { useThree } from "@react-three/fiber"; // Import useThree
import PLYModel from "./plymodel"

function Trial() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(null);
   const [zoom, setZoom] = useState(65); 

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        alert("Error processing file: " + data.error);
        setLoading(false);
        return;
      }

      const filename = data.filename;
      fetchModel(filename);
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed!");
      setLoading(false);
    }
  };

  const fetchModel = async (filename) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/output`);
      console.log(response);
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Blob([arrayBuffer], { type: "application/octet-stream" });
      console.log("Fetched PLY file:", blob);

      const modelUrl = URL.createObjectURL(blob);
      setModel(modelUrl);

      

      
      setLoading(false);
    } catch (error) {
      console.error("Error fetching model:", error);
      alert("Failed to fetch processed model!");
      setLoading(false);
    }
  };

  function Scene({ fileUrl, zoom }) {
    const { camera } = useThree();
    camera.position.set(0, 0, zoom); // Dynamically update camera zoom
  
    return (
      <>
        <ambientLight intensity={1} />
        <pointLight position={[10, 10, 10]} />
        <OrbitControls enableRotate={true} enablePan={true} />
        {fileUrl && <PLYModel url={fileUrl} position={[0, 0, -10]} />}
      </>
    );
  }

  return (
    <div className="flex w-full h-screen bg-gray-900 text-white">
      {/* Left Panel - Upload Section (1/3 of screen width, full height) */}
      <div className="w-1/3 h-full flex flex-col items-center justify-center p-10">
        <img src="./src/assets/logo.png" alt="Upload" className="w-60 h-65 mb-15" />

        <div className="bg-gray-800 p-6 rounded-lg shadow-xl w-80">
          <input
            type="file"
            accept=".xyz"
            className="w-full text-sm text-gray-300
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-600 file:text-white
                     hover:file:bg-blue-700"
            onChange={handleFileChange}
          />

          {preview && <p className="mt-4 text-green-400">File selected: {file?.name}</p>}

          <button
            className="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg w-full"
            onClick={handleUpload}
          >
            {loading ? "Processing..." : "Upload & Convert"}
          </button>
        </div>
      </div>

      {/* Right Panel - 3D Viewer (2/3 of screen width, full height) */}
      <div className="w-2/3 h-full flex items-center justify-center p-6">
        <div className="w-full h-full border-4 border-gray-300 rounded-xl shadow-xl bg-black flex justify-center items-center relative">
          <Canvas
            camera={{ position: [0, 0, zoom], fov: 75 }}
            className="w-full h-full"
          >
            <Scene fileUrl={model} zoom={zoom} />
          </Canvas>

          {/* Zoom Buttons */}
          <div className="absolute bottom-4 flex gap-4">
           {/*<button
              onClick={() => setZoom((prev) => Math.min(prev + 10, 200))}
              className="px-4 py-2 bg-white-500 text-white font-semibold rounded-lg shadow-md hover:bg-red-600 transition duration-300"
            >
              ➖
            </button>
            <button
              onClick={() => setZoom((prev) => Math.max(prev - 10, 20))}
              className="px-4 py-2 bg-white-500 text-white font-semibold rounded-lg shadow-md hover:bg-green-600 transition duration-300"
            >
              ➕
            </button>*/}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Trial;