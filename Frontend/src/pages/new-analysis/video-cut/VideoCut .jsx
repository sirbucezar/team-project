import React, { useState, useRef } from 'react';
import s from './styles.module.scss';

const VideoCut = ({ videoSrc, rawFile, onTrim, onClose }) => {
   const [startTime, setStartTime] = useState(0);
   const [endTime, setEndTime] = useState(30);
   const videoRef = useRef(null);

   const handleTrim = async () => {
      const video = videoRef.current;
      if (!video) return;

      // Simulating video trim by getting the first 30 seconds (you can implement real trimming here)
      const trimmedBlob = rawFile.slice(0, rawFile.size * (endTime / video.duration));
      const trimmedFile = new File([trimmedBlob], `trimmed_${rawFile.name}`, {
         type: rawFile.type,
      });

      onTrim(trimmedFile);
   };

   return (
      <div className={s.videoCutPopup}>
         <div className={s.videoCutPopup__content}>
            <h2>Trim your video to 30 seconds</h2>
            <video ref={videoRef} src={videoSrc} controls />
            <div className={s.controls}>
               <label>
                  Start time (sec):
                  <input
                     type="number"
                     min="0"
                     max={endTime}
                     value={startTime}
                     onChange={(e) => setStartTime(Number(e.target.value))}
                  />
               </label>
               <label>
                  End time (sec):
                  <input
                     type="number"
                     min={startTime}
                     max="30"
                     value={endTime}
                     onChange={(e) => setEndTime(Number(e.target.value))}
                  />
               </label>
            </div>
            <button onClick={handleTrim}>Trim Video</button>
            <button onClick={onClose}>Cancel</button>
         </div>
      </div>
   );
};

export default VideoCut;
