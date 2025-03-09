import React, { useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import * as THREE from "three";

export default function Tester() {
  const [model, setModel] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith(".ply")) {
      const reader = new FileReader();
      reader.onload = function (e) {
        const loader = new PLYLoader();
        const geometry = loader.parse(e.target.result);
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({ color: "skyblue", wireframe: false });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(0, 0, 0); // Center it
        setModel(mesh);
      };
      reader.readAsArrayBuffer(file);
    } else {
      alert("Please upload a valid .ply file");
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-100">
      {/* Upload Button */}
      <div className="p-4 bg-white shadow-md flex justify-center">
        <button
          className="px-6 py-3 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600"
          onClick={() => fileInputRef.current.click()}
        >
          Upload PLY File
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
      <div className="flex-1">
        <Canvas camera={{ position: [0,0,0] }} className="w-full h-full">
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 10]} />
          <OrbitControls enableZoom enableRotate enablePan />
          <group>{model && <primitive object={model} />}</group>
        </Canvas>
      </div>
    </div>
  );
}
