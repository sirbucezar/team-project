import React, { useRef, useState, useEffect } from 'react';
import s from './styles.module.scss';
import VideoTicks from './video-ticks/VideoTicks';
import VideoInfo from './video-info/VideoInfo';
import VideoStages from './video-stages/VideoStages';
import { toast } from 'sonner';

const VideoEditor = ({
   isVideoCut,
   videoSrc,
   setIsStagesSaved,
   rubric,
   setRubric,
   fromTime,
   setFromTime,
   toTime,
   setToTime,
   startFrame,
   setStartFrame,
   endFrame,
   setEndFrame,
   isDurationValid,
   setIsDurationValid,
   currentStage,
   setCurrentStage,
}) => {
   const videoRef = useRef(null);
   const trackRef = useRef(null);
   const [progress, setProgress] = useState(0);
   const [duration, setDuration] = useState(0);
   const [isDraggingStart, setIsDraggingStart] = useState(false);
   const [isDraggingEnd, setIsDraggingEnd] = useState(false);
   const [isScrubbing, setIsScrubbing] = useState(false);
   const minFrameSelect = 3;
   const [videoLength, setVideoLength] = useState(null);
   const [lastChange, setLastChange] = useState(null);
   const [isPlaying, setIsPlaying] = useState(false);
   const [isPlayingChanged, setIsPlayingChanged] = useState(false);
   const [currentTime, setCurrentTime] = useState(0);
   const [frameRate, setFrameRate] = useState(30);
   const StickyThreshold = 0.2; // Adjust the threshold in seconds to define proximity
   const StickyThresholdPercentage = 1;

   // Define zoom scale levels
   const zoomScales = [1, 1.5, 2]; // Corresponding to zoom levels 0, 1, 2
   const [zoomLevel, setZoomLevel] = useState(0); // 0: normal, 1: zoom in, 2: zoom in more
   const scrollIntervalRef = useRef(null);
   const bottomContainerRef = useRef(null);
   const progressRef = useRef(null);

   const saveStage = () => {
      const newStages = rubric.stages.map((stage, index) => {
         return index === currentStage
            ? { ...stage, start_time: startFrame, end_time: endFrame }
            : stage;
      });
      const newRubric = { ...rubric, stages: newStages };
      toast.success(`Stage ${currentStage + 1} was saved!`);

      // console.log(newStages);

      let rubricSaved = 0;
      let nextStage = NaN;
      newRubric.stages.map((stage, index) => {
         if (stage.start_time !== null && stage.end_time !== null) {
            rubricSaved++;
         } else {
            if (isNaN(nextStage)) {
               nextStage = index;
            }
         }
      });
      if (rubricSaved === newStages.length) {
         setIsStagesSaved(true);
      } else {
         handleStageChange(nextStage);
      }
      setRubric(newRubric);
   };

   const handleStageChange = (index) => {
      setLastChange(null);
      const newCurrentStage = rubric.stages[index];
      if (newCurrentStage.start_time !== null && newCurrentStage.end_time !== null) {
         setStartFrame(newCurrentStage.start_time);
         setEndFrame(newCurrentStage.end_time);
      } else {
         setStartFrame(0);

         setEndFrame(videoLength);
      }

      setCurrentStage(index);

      if (videoRef.current) {
         videoRef.current.pause();
         setIsPlaying(false);
      }
   };

   useEffect(() => {
      const video = videoRef.current;

      const handleLoadedMetadata = () => {
         setDuration(video.duration); // Set the video duration
      };

      if (video) {
         video.addEventListener('loadedmetadata', handleLoadedMetadata);
      }

      return () => {
         if (video) {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
         }
      };
   }, []);

   useEffect(() => {
      const updateProgress = () => {
         const video = videoRef.current;
         if (video && !isScrubbing) {
            const percentage = (video.currentTime / video.duration) * 100;
            if (!isDraggingStart && !isDraggingEnd) {
               setCurrentTime(video.currentTime);
               setProgress(percentage);
            }
         }
      };

      const video = videoRef.current;
      if (video) {
         video.addEventListener('timeupdate', updateProgress);
      }

      return () => {
         if (video) {
            video.removeEventListener('timeupdate', updateProgress);
         }
      };
   }, [isScrubbing, isDraggingStart, isDraggingEnd]);

   const handleScrub = (e) => {
      // const track = trackRef.current;
      // const video = videoRef.current;

      // if (track && video) {
      //    const rect = track.getBoundingClientRect();
      //    const clientX = e.clientX || e.touches[0].clientX;
      //    const clickX = Math.max(0, Math.min(clientX - rect.left, rect.width));
      //    const clickPercentage = clickX / rect.width;
      //    video.currentTime = clickPercentage * video.duration;
      //    if (!isDraggingStart && !isDraggingEnd) {
      //       setProgress(clickPercentage * 100);
      //       const newCurrentTime = video.currentTime;

      //       setCurrentTime(newCurrentTime);
      //    }
      // }

      const track = trackRef.current;
      const video = videoRef.current;

      if (track && video) {
         const rect = track.getBoundingClientRect();
         const clientX = e.clientX || e.touches[0].clientX;
         const clickX = Math.max(0, Math.min(clientX - rect.left, rect.width));
         const clickPercentage = clickX / rect.width;
         let newCurrentTime = clickPercentage * video.duration;

         // Calculate threshold dynamically based on percentage of video duration
         const stickyThreshold = (StickyThresholdPercentage / 100) * video.duration;

         // Apply snapping logic during scrubbing
         const startTime = startFrame / frameRate;
         const endTime = endFrame / frameRate;

         if (Math.abs(newCurrentTime - startTime) < stickyThreshold) {
            newCurrentTime = startTime;
         } else if (Math.abs(newCurrentTime - endTime) < stickyThreshold) {
            newCurrentTime = endTime;
         }

         video.currentTime = newCurrentTime;

         if (!isDraggingStart && !isDraggingEnd) {
            setCurrentTime(newCurrentTime);
            setProgress((newCurrentTime / video.duration) * 100);
         }
      }
   };

   useEffect(() => {
      const video = videoRef.current;

      const handleLoadedMetadata = () => {
         setDuration(video.duration); // Set the video duration in seconds

         // Extract frame rate dynamically (approximated using total frames)
         const totalFrames = video.webkitVideoDecodedByteCount || video.duration * 30; // Use a fallback frame rate of 30
         setFrameRate(totalFrames / video.duration);

         const newVideoLength = Math.floor(video.duration * frameRate);
         setVideoLength(newVideoLength);

         setStartFrame(0); // Set end position as total frames
         setEndFrame(newVideoLength); // Set end position as total frames
      };

      if (video) {
         video.addEventListener('loadedmetadata', handleLoadedMetadata);
      }

      return () => {
         if (video) {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
         }
      };
   }, [frameRate]);

   useEffect(() => {
      const updateProgress = () => {
         const video = videoRef.current;
         if (video) {
            const percentage = (video.currentTime / video.duration) * 100;
            if (!isDraggingStart && !isDraggingEnd) {
               // console.log(percentage, progress);
               // setProgress(percentage);
            }

            if (isVideoCut) {
               const startMinutes = Math.floor(startFrame / frameRate / 60);
               const startSeconds = Math.floor((startFrame / frameRate) % 60);
               setFromTime(`${startMinutes}:${startSeconds.toString().padStart(2, '0')}`);

               const endMinutes = Math.floor(endFrame / frameRate / 60);
               const endSeconds = Math.floor((endFrame / frameRate) % 60);
               setToTime(`${endMinutes}:${endSeconds.toString().padStart(2, '0')}`);

               const selectedDuration = (endFrame - startFrame) / frameRate;
               setIsDurationValid(selectedDuration <= 30);
            }
         }
      };

      const video = videoRef.current;
      if (video) {
         video.addEventListener('timeupdate', updateProgress);
      }

      return () => {
         if (video) {
            video.removeEventListener('timeupdate', updateProgress);
         }
      };
   }, [isDraggingStart, isDraggingEnd]);

   const startScrubbing = (e) => {
      setIsScrubbing(true);
      handleScrub(e);
   };

   const scrubbing = (e) => {
      if (isScrubbing) {
         handleScrub(e);
      }
   };

   const stopScrubbing = () => {
      setIsScrubbing(false);
   };

   useEffect(() => {
      const handleKeyDown = (e) => {
         const video = videoRef.current;
         if (!video) return;

         const shiftMultiplier = e.shiftKey ? 10 : 1;
         const frameTime = 1 / frameRate; // Time per frame in seconds
         let newCurrentTime;
         switch (e.key.toLowerCase()) {
            case ' ': // Space key to toggle play/pause
               e.preventDefault(); // Prevent default scrolling behavior when pressing space
               togglePlayPause();
               break;
            case 'a': // Move backward
            case 'arrowleft':
               newCurrentTime = Math.max(video.currentTime - frameTime * shiftMultiplier, 0);
               video.currentTime = newCurrentTime;
               setCurrentTime(newCurrentTime);
               break;
            case 'd': // Move forward
            case 'arrowright':
               newCurrentTime = Math.min(video.currentTime + frameTime * shiftMultiplier, duration);
               video.currentTime = newCurrentTime;
               setCurrentTime(newCurrentTime);
               break;
            case 'b': // create a breakpoint
               if (lastChange) {
                  const frameTime = 1 / frameRate; // Time per frame in seconds
                  const newCurrentFrame = Math.round(video.currentTime * frameRate); // Current frame

                  if (lastChange === 'start') {
                     if (endFrame - minFrameSelect > newCurrentFrame) {
                        setStartFrame(newCurrentFrame);
                     }
                  } else if (lastChange === 'end') {
                     // setEndFrame(currentFrame);
                     if (startFrame + minFrameSelect < newCurrentFrame) {
                        setEndFrame(newCurrentFrame);
                     }
                  }
               }
               break;
            case '+':
            case '=':
               handleZoom('in');
               break;
            case '-':
               handleZoom('out');
               break;
            default:
               break;
         }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => {
         window.removeEventListener('keydown', handleKeyDown);
      };
   }, [frameRate, duration, lastChange]);

   useEffect(() => {
      if (isScrubbing) {
         document.addEventListener('mousemove', scrubbing);
         document.addEventListener('mouseup', stopScrubbing);
         document.addEventListener('touchmove', scrubbing);
         document.addEventListener('touchend', stopScrubbing);
      } else {
         document.removeEventListener('mousemove', scrubbing);
         document.removeEventListener('mouseup', stopScrubbing);
         document.removeEventListener('touchmove', scrubbing);
         document.removeEventListener('touchend', stopScrubbing);
      }

      return () => {
         document.removeEventListener('mousemove', scrubbing);
         document.removeEventListener('mouseup', stopScrubbing);
         document.removeEventListener('touchmove', scrubbing);
         document.removeEventListener('touchend', stopScrubbing);
      };
   }, [isScrubbing]);

   useEffect(() => {
      if (isDraggingStart || isDraggingEnd) {
         document.addEventListener('mousemove', handleDragging);
         document.addEventListener('mouseup', handleDragEnd);
         document.addEventListener('touchmove', handleDragging);
         document.addEventListener('touchend', handleDragEnd);
      }

      return () => {
         document.removeEventListener('mousemove', handleDragging);
         document.removeEventListener('mouseup', handleDragEnd);
         document.removeEventListener('touchmove', handleDragging);
         document.removeEventListener('touchend', handleDragEnd);
      };
   }, [isDraggingStart, isDraggingEnd]);

   const handleDragStart = (e, type) => {
      const video = videoRef.current;
      if (!video) return;

      video.pause();
      setIsPlaying(false);

      if (type === 'start') {
         setIsDraggingStart(true);
         setLastChange('start');
      } else if (type === 'end') {
         setIsDraggingEnd(true);
         setLastChange('end');
      }
   };

   const handleDragging = (e) => {
      const track = trackRef.current;
      const bottomContainer = document.querySelector(`.${s.videoEditor__bottom}`);

      if (track) {
         const rect = track.getBoundingClientRect();
         const clientX = e.clientX || e.touches?.[0]?.clientX;
         const positionX = Math.max(0, Math.min(clientX - rect.left, rect.width));
         const totalFrames = Math.floor(duration * frameRate);
         const newFrame = Math.round((positionX / rect.width) * totalFrames);

         if (isDraggingStart) {
            if (newFrame < endFrame - minFrameSelect) {
               handleScrub(e);
               setStartFrame(newFrame);
            }
         } else if (isDraggingEnd) {
            if (newFrame > startFrame + minFrameSelect) {
               handleScrub(e);
               setEndFrame(newFrame);
            }
         }

         // Convert position to percentage
         const handlePositionPercent = (positionX / rect.width) * 100;
         const containerScrollPercent =
            (bottomContainer.scrollLeft / bottomContainer.scrollWidth) * 100;
         const containerVisiblePercent =
            (bottomContainer.clientWidth / bottomContainer.scrollWidth) * 100;

         // Adjust scrolling using percentage logic

         // Clear any existing interval to prevent multiple calls
         clearInterval(scrollIntervalRef.current);

         // Start automatic scrolling if the handle is near the edges
         if (handlePositionPercent < containerScrollPercent + 2) {
            scrollIntervalRef.current = setInterval(() => {
               bottomContainer.scrollLeft -= bottomContainer.scrollWidth * 0.002; // Scroll left smoothly
            }, 10);
         } else if (handlePositionPercent > containerScrollPercent + containerVisiblePercent - 2) {
            scrollIntervalRef.current = setInterval(() => {
               bottomContainer.scrollLeft += bottomContainer.scrollWidth * 0.002; // Scroll right smoothly
            }, 10);
         }
      }
   };

   const handleDragEnd = () => {
      setIsDraggingStart(false);
      setIsDraggingEnd(false);

      const video = videoRef.current;
      if (!video) return;

      video.currentTime = currentTime;

      // Clear the scrolling interval when dragging ends
      clearInterval(scrollIntervalRef.current);

      // console.log(`Start Frame: ${startFrame}, End Frame: ${endFrame}`);
   };

   const calculatePositionPercentage = (frame) => {
      const totalFrames = Math.floor(duration * frameRate);
      return (frame / totalFrames) * 100;
   };

   const handleZoom = (direction) => {
      if (!bottomContainerRef.current || !progressRef.current) return;

      const bottomContainer = bottomContainerRef.current;
      const progressIndicator = progressRef.current;

      // Get current zoom level and determine new zoom level
      setZoomLevel((prevZoom) => {
         const newZoom = direction === 'in' ? Math.min(prevZoom + 1, 2) : Math.max(prevZoom - 1, 0);

         return newZoom;
      });

      setTimeout(() => {
         // Get progress indicator position relative to the track
         const progressRect = progressIndicator.getBoundingClientRect();
         const bottomContainerRect = bottomContainer.getBoundingClientRect();

         const progressCenterPosition = progressRect.left + progressRect.width / 2;
         const bottomContainerCenterPosition =
            bottomContainerRect.left + bottomContainer.clientWidth / 2;

         let scrollOffset = progressCenterPosition - bottomContainerCenterPosition;

         // Ensure it stays within bounds of the container
         if (progressRect.left < bottomContainerRect.left) {
            scrollOffset = progressRect.left - bottomContainerRect.left;
         } else if (progressRect.right > bottomContainerRect.right) {
            scrollOffset = bottomContainerRect.right + bottomContainer.scrollLeft;
         }

         // Smoothly scroll to the calculated position
         bottomContainer.scrollBy({
            left: scrollOffset,
            behavior: 'smooth',
         });
      }, 0);
   };

   const togglePlayPause = () => {
      const video = videoRef.current;

      if (video) {
         setIsPlayingChanged(false);
         setIsPlaying((prevIsPlaying) => {
            if (prevIsPlaying) {
               video.pause();
            } else {
               video.play();
            }
            return !prevIsPlaying; // Toggle the isPlaying state
         });
         setTimeout(() => {
            setIsPlayingChanged(true);
         }, 0);
      }
   };

   return (
      <div className={s.videoEditor}>
         {!isVideoCut && (
            <VideoStages
               rubric={rubric}
               setRubric={setRubric}
               currentStage={currentStage}
               setCurrentStage={setCurrentStage}
               saveStage={saveStage}
               handleStageChange={handleStageChange}
            />
         )}
         <div className={s.videoEditor__main}>
            <div className={s.videoEditor__body}>
               <VideoInfo />
               <video
                  ref={videoRef}
                  src={videoSrc}
                  // controls
                  className={s.videoEditor__video}
                  controls={false}></video>
               <div className={s.videoEditor__controls} onClick={togglePlayPause}>
                  <button
                     className={`${s.videoEditor__playPause} ${isPlayingChanged ? s.active : ''}`}>
                     {isPlaying ? (
                        <svg
                           width="24"
                           height="24"
                           viewBox="0 0 24 24"
                           fill="none"
                           xmlns="http://www.w3.org/2000/svg">
                           <path
                              fill="#f1f1f1"
                              d="M6 3L20 12L6 21V3Z"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                           />
                        </svg>
                     ) : (
                        <svg
                           width="24"
                           height="24"
                           viewBox="0 0 24 24"
                           fill="none"
                           xmlns="http://www.w3.org/2000/svg">
                           <path
                              fill="#f1f1f1"
                              d="M17 4H15C14.4477 4 14 4.44772 14 5V19C14 19.5523 14.4477 20 15 20H17C17.5523 20 18 19.5523 18 19V5C18 4.44772 17.5523 4 17 4Z"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                           />
                           <path
                              fill="#f1f1f1"
                              d="M9 4H7C6.44772 4 6 4.44772 6 5V19C6 19.5523 6.44772 20 7 20H9C9.55228 20 10 19.5523 10 19V5C10 4.44772 9.55228 4 9 4Z"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                           />
                        </svg>
                     )}
                  </button>
               </div>
            </div>
            <div className={s.videoEditor__bottom} ref={bottomContainerRef}>
               <div className={`${s.videoEditor__trackWrapper} ${s[`zoomLevel${zoomLevel}`]}`}>
                  <VideoTicks duration={duration} startScrubbing={startScrubbing} />
                  <div ref={trackRef} className={s.videoEditor__track}>
                     {/* Range Highlight */}
                     <div
                        className={s.videoEditor__rangeHighlight}
                        style={{
                           left: `${calculatePositionPercentage(startFrame)}%`,
                           width: `${calculatePositionPercentage(endFrame - startFrame)}%`,
                        }}
                        // onMouseDown={(e) => handleDragStart(e, 'range')}
                        // onTouchStart={(e) => handleDragStart(e, 'range')}
                     ></div>
                     <div>
                        {/* Start Handle */}
                        <div
                           className={s.videoEditor__rangeHandle}
                           style={{ left: `${calculatePositionPercentage(startFrame)}%` }}
                           onMouseDown={(e) => handleDragStart(e, 'start')}
                           onTouchStart={(e) => handleDragStart(e, 'start')}>
                           <div className={s.videoEditor__handleSmall}></div>
                        </div>

                        {/* End Handle */}
                        <div
                           className={s.videoEditor__rangeHandle}
                           style={{ left: `${calculatePositionPercentage(endFrame)}%` }}
                           onMouseDown={(e) => handleDragStart(e, 'end')}
                           onTouchStart={(e) => handleDragStart(e, 'end')}>
                           <div className={s.videoEditor__handleSmall}></div>
                        </div>
                     </div>

                     {/* Progress */}
                     <div
                        ref={progressRef}
                        className={s.videoEditor__progress}
                        style={{ left: `${progress}%` }}>
                        <div className={s.videoEditor__progressBody}>
                           <div className={s.videoEditor__progressTriangle}></div>
                           <div className={s.videoEditor__progressLine}></div>
                        </div>
                     </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
   );
};

export default VideoEditor;
