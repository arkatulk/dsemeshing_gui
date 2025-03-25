import React, { useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import PLYModel from "./plymodel";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);
  const [zoom, setZoom] = useState(75); // Initial Zoom

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile && uploadedFile.name.endsWith(".xyz")) {
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
    } else {
      alert("Please upload a valid .xyz file");
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("Uploaded:", data);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-gradient-to-r from-white-800 via-blue-500 to-black-500">
      {/* Upload Button */}
      <div className="p-6 bg-white shadow-lg rounded-lg flex flex-col items-center m-5">
        <button
          className="px-6 py-3 bg-blue-700 text-white font-semibold rounded-lg shadow-md hover:bg-blue-800 transition duration-300"
          onClick={() => fileInputRef.current.click()}
        >
          Upload XYZ File
        </button>
        <input
          type="file"
          accept=".xyz"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
        {file && (
          <button
            onClick={handleUpload}
            className="mt-4 px-6 py-3 bg-green-500 text-white font-semibold rounded-lg shadow-md hover:bg-green-600 transition duration-300"
          >
            Upload to Server
          </button>
        )}
      </div>

      {/* 3D Viewer */}
      <div className="flex-1 p-6 flex justify-center items-center w-full">
        <div className="w-[40%] h-[60vh] border-4 border-gray-300 rounded-xl shadow-xl bg-black flex justify-center items-center relative">
          <Canvas
            camera={{ position: [0, 0, zoom], fov: 75 }}
            className="w-full h-full"
          >
            <Scene fileUrl={preview} zoom={zoom} />
          </Canvas>

          {/* Zoom Buttons */}
          <div className="absolute bottom-4 flex gap-4">
            <button
              onClick={() => setZoom((prev) => Math.min(prev + 10, 200))}
              className="px-4 py-2 bg-red-500 text-white font-semibold rounded-lg shadow-md hover:bg-red-600 transition duration-300"
            >
              ➖
            </button>
            <button
              onClick={() => setZoom((prev) => Math.max(prev - 10, 20))}
              className="px-4 py-2 bg-green-500 text-white font-semibold rounded-lg shadow-md hover:bg-green-600 transition duration-300"
            >
              ➕
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Scene Component
function Scene({ fileUrl, zoom }) {
  const { camera } = useThree();
  camera.position.set(0, 0, zoom); // Dynamically update camera zoom

  return (
    <>
      <ambientLight intensity={4} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls enableRotate={true} enablePan={true} />
      {fileUrl && <PLYModel url={fileUrl} position={[0, 0, -10]} />}
    </>
  );
}