import React, { useEffect } from 'react';
import s from './styles.module.scss';

const VideoHint = ({ currentStage, hints }) => {
   useEffect(() => {
      console.log(currentStage);
   }, [currentStage]);

   return (
      <div className={s.hint}>
         <div className={s.hint__body}>
            <div className={s.hint__title}>
               <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg">
                  <path
                     d="M12 13V21M12 3V6M4 6C3.73478 6 3.48043 6.10536 3.29289 6.29289C3.10536 6.48043 3 6.73478 3 7V12C3 12.2652 3.10536 12.5196 3.29289 12.7071C3.48043 12.8946 3.73478 13 4 13H17C17.4124 13 17.8148 12.8725 18.152 12.635L21.576 10.318C21.707 10.2257 21.8139 10.1032 21.8877 9.96097C21.9615 9.8187 22.0001 9.66077 22.0001 9.5005C22.0001 9.34022 21.9615 9.1823 21.8877 9.04003C21.8139 8.89775 21.707 8.77531 21.576 8.683L18.152 6.365C17.8148 6.12746 17.4124 5.99997 17 6H4Z"
                     stroke="black"
                     strokeWidth="2"
                     strokeLinecap="round"
                     strokeLinejoin="round"
                  />
               </svg>
               <span>Hint for stage</span>
            </div>
            <div className={s.hint__descr}>{hints[currentStage]}</div>
         </div>
      </div>
   );
};

export default VideoHint;
