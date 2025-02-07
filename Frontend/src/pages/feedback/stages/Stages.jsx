import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ExampleSrc from '/video/example.mp4';
import s from './styles.module.scss';

const BorderSnakeAnimation = ({ color, isExpanded }) => {
   return (
      <motion.div
         className={s.borderAnimation}
         initial={{ strokeDashoffset: 500 }}
         animate={{ strokeDashoffset: isExpanded ? 0 : 500 }}
         transition={{ duration: 4, ease: 'easeInOut' }}
         style={{
            borderColor: color,
         }}
      />
   );
};

const Stages = ({ stageEntries, currentRubric }) => {
   const [expandedStage, setExpandedStage] = useState(null);
   const [isFullyExpanded, setIsFullyExpanded] = useState(false);
   const [animateStages, setAnimateStages] = useState(false);
   const originalPositionRef = useRef(null);
   const expandedRef = useRef(null);

   useEffect(() => {
      setAnimateStages(true);
   }, []);

   useEffect(() => {
      const handleOutsideClick = (event) => {
         if (
            expandedRef.current &&
            !expandedRef.current.contains(event.target) &&
            expandedStage !== null
         ) {
            handleClose();
         }
      };

      document.addEventListener('mousedown', handleOutsideClick);

      return () => {
         document.removeEventListener('mousedown', handleOutsideClick);
      };
   }, [expandedStage]);

   const getStageClass = (predScore) => {
      const score = Number(predScore);
      if (score === 0) return s.red;
      if (score === 0.5) return s.purple;
      if (score === 1) return s.green;
      return '';
   };

   const getScoreTagClass = (predScore) => {
      const score = Number(predScore);
      if (score === 0) return s.tagRed;
      if (score === 0.5) return s.tagPurple;
      if (score === 1) return s.tagGreen;
      return s.tagDefault;
   };

   const getScoreConfidenceClass = (predConfidence) => {
      const confidence = Number(predConfidence);
      if (confidence < 0.3) return s.confidenceBad;
      else if (confidence < 0.8) return s.confidenceMid;
      else if (confidence <= 1) return s.confidenceGood;
      return s.confidenceDefault;
   };

   const getBorderColor = (predScore) => {
      const score = Number(predScore);
      if (score === 0) return '#d93030';
      if (score === 0.5) return '#8638eb';
      if (score === 1) return '#2ecc71';
      return 'transparent';
   };

   const handleStageClick = (index, event) => {
      const rect = event.currentTarget.getBoundingClientRect();
      originalPositionRef.current = rect;
      setExpandedStage(index);
      setTimeout(() => {
         setIsFullyExpanded(true);
      }, 0);
   };

   const handleClose = () => {
      setIsFullyExpanded(false);
      setTimeout(() => {
         setExpandedStage(null);
      }, 400);
   };

   return (
      <div className={s.stages}>
         <ul className={s.stages__list}>
            {stageEntries.map((stage, index) => (
               <motion.li
                  key={index}
                  ref={index === expandedStage ? originalPositionRef : null}
                  className={`${s.stages__item} ${getStageClass(stage.score)}`}
                  onClick={(e) => handleStageClick(index, e)}
                  initial={{ scale: 1 }}
                  animate={animateStages ? { scale: [1, 1.8, 1] } : {}}
                  transition={{
                     duration: 0.2,
                     delay: index * 0.1,
                  }}>
                  {stage.criterion}
               </motion.li>
            ))}
         </ul>

         <AnimatePresence>
            {expandedStage !== null && (
               <>
                  <div
                     className={`${s.stages__overlay} ${isFullyExpanded ? s.active : ''}`}
                     // onClick={handleClose}
                  >
                     <div className={s.stages__container}>
                        <li
                           ref={expandedRef}
                           className={`
                        ${s.stages__itemExpanded}
                        ${getStageClass(stageEntries[expandedStage].score)}
                        `}>
                           <button className={s.closeButton} onClick={handleClose}>
                              <svg
                                 width="24"
                                 height="24"
                                 viewBox="0 0 24 24"
                                 fill="none"
                                 xmlns="http://www.w3.org/2000/svg">
                                 <path
                                    d="M15 9L9 15M9 9L15 15M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z"
                                    stroke="black"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                 />
                              </svg>
                           </button>
                           <div className={s.expandedStage__content}>
                              <div className={s.expandedStage__left}>
                                 <h2 className={s.expandedStage__title}>
                                    {stageEntries[expandedStage].stage}
                                 </h2>

                                 <div className={s.expandedStage__tags}>
                                    <span
                                       className={`${s.tag} ${getScoreTagClass(
                                          stageEntries[expandedStage].score,
                                       )}`}>
                                       Score: {stageEntries[expandedStage].score}
                                    </span>

                                    <span
                                       className={`${s.confidence} ${getScoreConfidenceClass(
                                          stageEntries[expandedStage].confidence,
                                       )}`}>
                                       Confidence: {stageEntries[expandedStage].confidence}
                                    </span>
                                    {stageEntries[expandedStage].injury_risk.high_risk && (
                                       <span className={s.injuryRisk}>
                                          <svg
                                             width="31"
                                             height="30"
                                             viewBox="0 0 31 30"
                                             fill="none"
                                             xmlns="http://www.w3.org/2000/svg">
                                             <g filter="url(#filter0_d_381_2)">
                                                <path
                                                   d="M16.0004 9V13M16.0004 17H16.0104M25.7304 18L17.7304 4C17.556 3.69221 17.303 3.43619 16.9973 3.25807C16.6917 3.07995 16.3442 2.98611 15.9904 2.98611C15.6366 2.98611 15.2892 3.07995 14.9835 3.25807C14.6778 3.43619 14.4249 3.69221 14.2504 4L6.25042 18C6.0741 18.3054 5.98165 18.6519 5.98243 19.0045C5.98321 19.3571 6.0772 19.7033 6.25486 20.0078C6.43253 20.3124 6.68757 20.5646 6.99411 20.7388C7.30066 20.9131 7.64783 21.0032 8.00042 21H24.0004C24.3513 20.9996 24.6959 20.907 24.9997 20.7313C25.3035 20.5556 25.5556 20.3031 25.7309 19.9991C25.9062 19.6951 25.9985 19.3504 25.9984 18.9995C25.9983 18.6486 25.9059 18.3039 25.7304 18Z"
                                                   stroke="#FFCC33"
                                                   strokeWidth="2"
                                                   strokeLinecap="round"
                                                   strokeLinejoin="round"
                                                />
                                             </g>
                                             <defs>
                                                <filter
                                                   id="filter0_d_381_2"
                                                   x="0"
                                                   y="0"
                                                   width="32"
                                                   height="32"
                                                   filterUnits="userSpaceOnUse"
                                                   colorInterpolationFilters="sRGB">
                                                   <feFlood
                                                      floodOpacity="0"
                                                      result="BackgroundImageFix"
                                                   />
                                                   <feColorMatrix
                                                      in="SourceAlpha"
                                                      type="matrix"
                                                      values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                                                      result="hardAlpha"
                                                   />
                                                   <feOffset dy="4" />
                                                   <feGaussianBlur stdDeviation="2" />
                                                   <feComposite in2="hardAlpha" operator="out" />
                                                   <feColorMatrix
                                                      type="matrix"
                                                      values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0"
                                                   />
                                                   <feBlend
                                                      mode="normal"
                                                      in2="BackgroundImageFix"
                                                      result="effect1_dropShadow_381_2"
                                                   />
                                                   <feBlend
                                                      mode="normal"
                                                      in="SourceGraphic"
                                                      in2="effect1_dropShadow_381_2"
                                                      result="shape"
                                                   />
                                                </filter>
                                             </defs>
                                          </svg>
                                          <span>Injury risk</span>
                                       </span>
                                    )}
                                 </div>

                                 <div className={s.expandedStage__divider}></div>

                                 <h3 className={s.expandedStage__feedbackTitle}>Feedback:</h3>
                                 <div className={s.expandedStage__feedbackText}>
                                    <div className={s.expandedStage__feedbackItem}>
                                       <div className={s.expandedStage__feedbackItemDescr}>
                                          {stageEntries[expandedStage].feedback.Observation.body}
                                       </div>
                                    </div>
                                    <div className={s.expandedStage__feedbackItem}>
                                       <div className={s.expandedStage__feedbackItemDescr}>
                                          {
                                             stageEntries[expandedStage].feedback[
                                                'Improvement Suggestion'
                                             ].body
                                          }
                                       </div>
                                    </div>
                                    <div className={s.expandedStage__feedbackItem}>
                                       <div className={s.expandedStage__feedbackItemDescr}>
                                          {stageEntries[expandedStage].feedback.Justification.body}
                                       </div>
                                    </div>
                                    <div className={s.expandedStage__feedbackItem}>
                                       <div className={s.expandedStage__feedbackItemDescr}>
                                          {stageEntries[expandedStage].feedback.Encouragement.body}
                                       </div>
                                    </div>
                                 </div>
                              </div>

                              <div className={s.expandedStage__right}>
                                 <video
                                    autoPlay
                                    muted
                                    playsInline
                                    loop
                                    src={`/video/stages2/${expandedStage + 1}.mp4`}
                                    className={s.stages__video}
                                 />
                              </div>
                           </div>
                        </li>
                     </div>
                  </div>
               </>
            )}
         </AnimatePresence>
      </div>
   );
};

export default Stages;
