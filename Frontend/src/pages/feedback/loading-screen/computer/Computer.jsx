import React, { useRef } from 'react';
import { useGLTF, useTexture } from '@react-three/drei';
import { MeshMatcapMaterial } from 'three';
import { useFrame } from '@react-three/fiber';

function Computer(props) {
   const { nodes, materials } = useGLTF('/models/old_pc.glb');
   const screenTexture = useTexture(`/textures/rubric/${props.currentRubric.id}.png`);

   const screenMatcapMaterial = new MeshMatcapMaterial({ map: screenTexture });

   const groupRef = useRef();

   // Add infinite levitation animation
   useFrame((state, delta) => {
      if (groupRef.current) {
         // Sine wave for smooth up-and-down motion
         groupRef.current.position.y += (Math.sin(state.clock.elapsedTime * 1.3) * 0.14) / 20; // Y oscillation
         // groupRef.current.position.y += delta * 1; // Slowly rotate around Y-axis
         groupRef.current.rotation.y += (Math.sin(state.clock.elapsedTime * 1.08) * 0.016) / 20; // Slowly rotate around Y-axis
      }
   });

   return (
      <group ref={groupRef} {...props} dispose={null}>
         <group position={[0, 1.923, 0]} rotation={[Math.PI / 2, 0, 0]} scale={0.01}>
            {nodes?.Mesh014 && (
               <mesh
                  castShadow
                  receiveShadow
                  geometry={nodes.Mesh014.geometry}
                  material={materials?.glass}
               />
            )}
            {nodes?.Mesh014_1 && (
               <mesh
                  castShadow
                  receiveShadow
                  geometry={nodes.Mesh014_1.geometry}
                  material={materials?.computer_mat}
               />
            )}
            {nodes?.Mesh014_2 && (
               <mesh
                  castShadow
                  receiveShadow
                  geometry={nodes.Mesh014_2.geometry}
                  material={materials?.cables_mat}
               />
            )}
            {nodes?.Mesh014_3 && (
               <mesh
                  castShadow
                  receiveShadow
                  geometry={nodes.Mesh014_3.geometry}
                  material={screenMatcapMaterial}
               />
            )}
         </group>
      </group>
   );
}

useGLTF.preload('/models/old_pc.glb');

export default Computer;
