import { useRef, useEffect } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import * as THREE from "three";

const PLYModel = ({ url }) => {
  const meshRef = useRef(); // Reference for the model
  const geometry = useLoader(PLYLoader, url);

  useEffect(() => {
    if (geometry) {
      geometry.computeVertexNormals();

      // Compute bounding box to adjust positioning
      const boundingBox = new THREE.Box3().setFromObject(meshRef.current);
      const size = boundingBox.getSize(new THREE.Vector3());
      const min = boundingBox.min;

      // Position the model so the bottom aligns with the bottom of the screen
      meshRef.current.position.set(0, -min.y, 0);
    }
  }, [geometry]);

  // Automatically rotate the model
  useFrame(() => {
    if (meshRef.current) {
      //meshRef.current.rotation.z += 0.01; // Smooth rotation
      meshRef.current.position.y = -40
    }
  });

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      onPointerOver={() => (document.body.style.cursor = "pointer")} // Hand cursor
      onPointerOut={() => (document.body.style.cursor = "default")} // Default cursor
    >
      <meshStandardMaterial color="white" />
    </mesh>
  );
};

export default PLYModel;