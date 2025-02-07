import React, { useEffect, useRef } from 'react';
import s from './styles.module.scss';
import ExampleSrc from '/video/example.mp4';

const ExampleVideo = () => {
   const videoRef = useRef(null);

   useEffect(() => {
      if (videoRef.current) {
         videoRef.current.play().catch((error) => {
            console.error('Autoplay failed: ', error);
         });
      }
   }, []);

   return (
      <div className={s.exampleVideo}>
         <div className={s.exampleVideo__video}>
            <video ref={videoRef} src={ExampleSrc} autoPlay muted playsInline loop></video>
         </div>
         <div className={s.exampleVideo__text}>
            <div className={s.exampleVideo__body}>
               <div className={s.exampleVideo__title}>
                  Enhance Your Performance with <span>AI-Powered</span> Video Analysis
               </div>
               <div className={s.exampleVideo__descr}>
                  Our web platform helps students and coaches analyze sports performances through
                  AI-driven models. Upload your training videos, receive detailed feedback, and
                  improve your techniques with precision
               </div>
            </div>
         </div>
      </div>
   );
};

export default ExampleVideo;
