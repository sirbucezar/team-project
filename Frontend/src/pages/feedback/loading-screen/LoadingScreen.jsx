import { Canvas, useFrame } from '@react-three/fiber';
import React, { Suspense, useEffect, useRef, useState } from 'react';
import CanvasLoader from '../../../components/canvas-loader/CanvasLoader';
import { PerspectiveCamera } from '@react-three/drei';
import Computer from './computer/Computer';
import s from './styles.module.scss';
import { useMediaQuery } from 'react-responsive';
import * as THREE from 'three';

/// 🚀 Particle Component (Handles Regular + Explosion Particles)
const Particle = ({ position, velocity, lifetime, isExploding, onComplete }) => {
   const ref = useRef();
   const elapsed = useRef(0);

   useFrame((state, delta) => {
      if (ref.current) {
         elapsed.current += delta;
         if (elapsed.current >= lifetime) {
            onComplete();
            return;
         }

         // Move in velocity direction
         ref.current.position.addScaledVector(velocity, delta * (isExploding ? 20 : 10));

         if (isExploding) {
            const progress = elapsed.current / lifetime; // 0 → 1 over lifetime
            const scaleFactor = progress * 10; // Controlled growth
            ref.current.scale.set(scaleFactor, scaleFactor, scaleFactor);

            // **Smoothly decrease opacity** until the particle disappears
            ref.current.material.opacity = Math.max(1 - progress, 0);
         } else {
            // Regular particles shrink normally
            ref.current.scale.lerp(new THREE.Vector3(0, 0, 0), delta * 3);
         }
      }
   });

   return (
      <mesh ref={ref} position={position}>
         <circleGeometry args={[0.1, 32]} />
         <meshBasicMaterial color="#f1f1f1" transparent opacity={0.8} />
      </mesh>
   );
};

// 🚀 Particle System (Handles Normal + Explosion Spawning)
const Particles = ({ isAnimating, isExploding, onExplosionComplete }) => {
   const [particles, setParticles] = useState([]);
   const targetPosition = new THREE.Vector3(0, 0, 0);
   const spawnIntervalRef = useRef();
   const explosionTriggered = useRef(false);

   // Regular Circle Spawning
   useEffect(() => {
      if (isAnimating) {
         spawnIntervalRef.current = setInterval(() => {
            for (let i = 0; i < 6; i++) {
               const randomPosition = [
                  Math.random() * 30 - 17,
                  Math.random() * 30 - 17,
                  Math.random() * 30 + 15,
               ];
               const velocity = new THREE.Vector3()
                  .subVectors(targetPosition, new THREE.Vector3(...randomPosition))
                  .normalize()
                  .multiplyScalar(2);

               setParticles((prev) => [
                  ...prev,
                  {
                     id: crypto.randomUUID(),
                     position: randomPosition,
                     velocity,
                     lifetime: Math.random() * 0.5 + 1.5,
                     isExploding: false,
                  },
               ]);
            }
         }, 50);
      } else {
         clearInterval(spawnIntervalRef.current);
      }

      return () => clearInterval(spawnIntervalRef.current);
   }, [isAnimating]);

   // 💥 Explosion Effect
   useEffect(() => {
      if (isExploding && !explosionTriggered.current) {
         explosionTriggered.current = true;

         for (let i = 0; i < 400; i++) {
            const randomVelocity = new THREE.Vector3(
               (Math.random() - 0.5) * 10, // Faster outward movement
               (Math.random() - 0.5) * 10,
               (Math.random() - 0.5) * 10,
            );

            setParticles((prev) => [
               ...prev,
               {
                  id: crypto.randomUUID(),
                  position: [0, 0, 0], // Start from the center
                  velocity: randomVelocity,
                  lifetime: Math.random() * 1.5 + 1, // Explosion particles last 1-2.5 seconds
                  isExploding: true, // Mark as explosion particle
               },
            ]);
         }

         // Auto-remove explosion effect after delay
         setTimeout(() => {
            setParticles([]);
            onExplosionComplete();
         }, 2000);
      }
   }, [isExploding]);

   const handleParticleComplete = (id) => {
      setParticles((prev) => prev.filter((particle) => particle.id !== id));
   };

   return (
      <>
         {particles.map((particle) => (
            <Particle
               key={particle.id}
               position={particle.position}
               velocity={particle.velocity}
               lifetime={particle.lifetime}
               isExploding={particle.isExploding}
               onComplete={() => handleParticleComplete(particle.id)}
            />
         ))}
      </>
   );
};

const LoadingScreen = ({ isExploding, setIsExploding, isLoading, currentRubric }) => {
   const [isAnimating, setIsAnimating] = useState(true);

   useEffect(() => {
      if (!isLoading) {
         setIsAnimating(false);
         setTimeout(() => setIsExploding(true), 500);
      }
   }, [isLoading]);

   const handleExplosionComplete = () => {
      // setIsExploding(false);
   };

   const isSmall = useMediaQuery({ maxWidth: 440 });
   const isMobile = useMediaQuery({ maxWidth: 768 });
   const isTablet = useMediaQuery({ minWidth: 768, maxWidth: 1024 });

   const calculateSizes = (isSmall, isMobile, isTablet) => {
      return {
         deskScale: isSmall ? 0.05 : isMobile ? 0.06 : 0.11,
         deskPosition: isMobile ? [0.5, -4.5, 0] : [0.25, -5.5, 0],
         cubePosition: isSmall
            ? [4, -5, 0]
            : isMobile
            ? [5, -5, 0]
            : isTablet
            ? [5, -5, 0]
            : [9, -5.5, 0],
         reactLogoPosition: isSmall
            ? [3, 4, 0]
            : isMobile
            ? [5, 4, 0]
            : isTablet
            ? [5, 4, 0]
            : [12, 3, 0],
         ringPosition: isSmall
            ? [-5, 7, 0]
            : isMobile
            ? [-10, 10, 0]
            : isTablet
            ? [-12, 10, 0]
            : [-24, 10, 0],
         targetPosition: isSmall
            ? [-5, -10, -10]
            : isMobile
            ? [-9, -10, -10]
            : isTablet
            ? [-11, -7, -10]
            : [-13, -13, -10],
      };
   };

   const sizes = calculateSizes(isSmall, isMobile, isTablet);
   return (
      <div>
         <Canvas
            className={`${s.canvas} ${isExploding ? s.hide : ''}`}
            style={{ position: 'fixed' }}>
            <Suspense fallback={<CanvasLoader />}>
               {/* <CanvasLoader /> */}
               <PerspectiveCamera
                  makeDefault
                  position={[0, 0, 30]}
                  // rotation={[-Math.PI / 2, 0, Math.PI / 4]}
                  // onUpdate={handleCameraReady}
               />
               <Computer
                  position={[2.2, -42, -15]}
                  rotation={[0.2, 3.8, 0]}
                  scale={[15, 15, 15]}
                  currentRubric={currentRubric}
                  // scale={sizes.deskScale}
                  // position={sizes.deskPosition}
                  // rotation={[0.1, -Math.PI, 0]}
               />
               <ambientLight intensity={0.1} />
               <directionalLight position={[100, 20, 10]} intensity={0.8} />
               <Particles
                  isAnimating={isAnimating}
                  isExploding={isExploding}
                  onExplosionComplete={handleExplosionComplete}
               />
            </Suspense>
         </Canvas>
         <div className={`${s.wrapper} ${isExploding ? s.hide : ''}`}>
            <div className={s.body}>
               <div className={s.status}>Loading...</div>
            </div>
         </div>
      </div>
   );
};

export default LoadingScreen;
