import React, { useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import PLYModel from "./plymodel";

export default function Tester() {
  const [fileUrl, setFileUrl] = useState(null);
  const fileInputRef = useRef(null);
  const [zoom, setZoom] = useState(75); // 🔥 Initial Zoom

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith(".ply")) {
      const reader = new FileReader();
      reader.readAsArrayBuffer(file);
      reader.onload = function (e) {
        const blob = new Blob([e.target.result]);
        const url = URL.createObjectURL(blob);
        setFileUrl(url);
      };
    } else {
      alert("Please upload a valid .ply file");
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-gradient-to-r from-white-800 via-blue-500 to-black-500">
      {/* Upload and View Buttons */}
      <div className="p-6 bg-white shadow-lg rounded-lg flex justify-center space-x-4">
        <button
          className="px-6 py-3 bg-blue-700 text-white font-semibold rounded-lg shadow-md hover:bg-blue-800 transition duration-300"
          onClick={() => fileInputRef.current.click()}
        >
          Upload PLY File
        </button>

        <button
          className="px-6 py-3 bg-green-700 text-white font-semibold rounded-lg shadow-md hover:bg-green-800 transition duration-300"
          onClick={() => fileInputRef.current.click()} // View button opens file menu
        >
          View PLY File
        </button>

        <input
          type="file"
          accept=".ply"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>

      {/* 3D Viewer */}
      <div className="flex-1 p-6 flex justify-center items-center w-full">
        <div className="w-[40%] h-[60vh] border-4 border-gray-300 rounded-xl shadow-xl bg-black flex justify-center items-center relative">
          <Canvas
            camera={{ position: [0, 0, zoom], fov: 75 }}
            className="w-full h-full"
          >
            <Scene fileUrl={fileUrl} zoom={zoom} />
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
  camera.position.set(0, 0, zoom); // 🔥 Dynamically update camera zoom

  return (
    <>
      <ambientLight intensity={1} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls enableRotate={true} enablePan={true} />
      {fileUrl && <PLYModel url={fileUrl} position={[0, 0, 0]} />}
    </>
  );
}
