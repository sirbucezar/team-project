import React, { useEffect, useState } from 'react';
import s from './styles.module.scss';
import UploadVideo from './upload-video/UploadVideo';
import ChooseStudent from './choose-student/ChooseStudent';
import Rubrics from './rubrics/Rubrics';
import VideoEditor from './video-editor/VideoEditor';
import { toast } from 'sonner';
import { useNavigate } from 'react-router';
import Sidebar from './sidebar/Sidebar';
import VideoCut from './video-cut/VideoCut ';
import { createFFmpeg, fetchFile } from '@ffmpeg/ffmpeg';
import ExampleVideo from './example-video/ExampleVideo';
import { generateUUID } from 'three/src/math/MathUtils.js';
import VideoHint from './video-editor/video-hint/VideoHint';

const NewAnalysis = ({
   rubrics,
   showVideoEditor,
   setShowVideoEditor,
   setIsFeedback,
   setFeedbackData,
   currentRubric,
   setCurrentRubric,
   setProcessingId,
   isSidebarShow,
   setIsSidebarShow,
}) => {
   const [ffmpeg, setFfmpeg] = useState(createFFmpeg({ log: false }));
   const [isTestBtn, setIsTestBtn] = useState(false);
   // Student, Sport, and local video states
   const [selectedStudent, setSelectedStudent] = useState('');
   const [students, setStudents] = useState([
      { id: 1, firstName: 'Mykyta', lastName: 'Tsykunov', isSelected: false },
      { id: 2, firstName: 'Cezar', lastName: 'Sîrbu', isSelected: false },
      { id: 3, firstName: 'Danylo', lastName: 'Bordunov', isSelected: false },
      { id: 4, firstName: 'Alex', lastName: 'Johnson', isSelected: false },
      { id: 5, firstName: 'Maria', lastName: 'Smith', isSelected: false },
      { id: 6, firstName: 'Elena', lastName: 'Brown', isSelected: false },
   ]);
   const [title, setTitle] = useState('Choose a student');

   const [videoSrc, setVideoSrc] = useState('');
   const [fileName, setFileName] = useState(null);
   const [rawFile, setRawFile] = useState(null);
   const [showVideoCut, setShowVideoCut] = useState(false);

   const [easterEgg, setEasterEgg] = useState(['Howest', 0]);
   let navigate = useNavigate();

   // SAS + “Analyze” workflow
   const [sasUrl, setSasUrl] = useState(null);
   const [isStagesSaved, setIsStagesSaved] = useState(false);
   const [isLoading, setIsLoading] = useState(false);

   // cut time
   const [fromTime, setFromTime] = useState('0:00');
   const [toTime, setToTime] = useState('0:00');

   const [startFrame, setStartFrame] = useState(0); // Start position in frames
   const [endFrame, setEndFrame] = useState(0); // End position in frames
   const [isDurationValid, setIsDurationValid] = useState(false);

   // Local “rubric” with 5 stages that store frames, not timestamps
   const [hints, setHints] = useState(null);
   const [currentStage, setCurrentStage] = useState(0);
   const [rubric, setRubric] = useState({
      video_id: '',
      stages: [
         {
            stage_name: 'stage1',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage2',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage3',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage4',
            start_time: null,
            end_time: null,
         },
         {
            stage_name: 'stage5',
            start_time: null,
            end_time: null,
         },
      ],
   });

   useEffect(() => {
      // Warn user before leaving page (only before form submission)
      const handleBeforeUnload = (event) => {
         if (showVideoEditor) {
            event.preventDefault();
            event.returnValue = ''; // Standard browser behavior
         }
      };

      window.addEventListener('beforeunload', handleBeforeUnload);

      return () => {
         window.removeEventListener('beforeunload', handleBeforeUnload);
      };
   }, [showVideoEditor]);

   const handleVideoUpload = (file) => {
      if (file) {
         const newFileName = file.name; // Access the file name

         if (file.type.startsWith('video')) {
            const videoElement = document.createElement('video');
            videoElement.preload = 'metadata';

            videoElement.onloadedmetadata = () => {
               window.URL.revokeObjectURL(videoElement.src);

               if (videoElement.duration > 30) {
                  toast.error(
                     `The uploaded video "${newFileName}" exceeds 30 seconds, please cut it in editor`,
                  );
                  setShowVideoCut(true); // Show video cut component
                  const fileURL = URL.createObjectURL(file);
                  setVideoSrc(fileURL);
                  setRawFile(file);
               } else {
                  console.log(`The uploaded video "${newFileName}" is within the allowed time.`);
                  const fileURL = URL.createObjectURL(file);
                  setVideoSrc(fileURL);
                  setFileName(file.name);
                  setRawFile(file);
               }
            };

            videoElement.src = URL.createObjectURL(file);
         } else {
            toast.error('Please upload a video my little friend :D');
            setFileName(null);
         }
      }

      // toast.success(`Video ${file.name} was selected!`);

      // * local
      // setShowVideoEditor(true);
   };

   const handleSubmit = async (e) => {
      e.preventDefault();

      if (!selectedStudent) {
         toast.error('Please choose a student first');
         return;
      } else if (!currentRubric) {
         toast.error('Pick a rubric to evaluate on');
         return;
      } else if (!fileName) {
         toast.error('Please upload a video first');
         return;
      }

      const newHints = rubrics.filter((el) => {
         if (el.id === currentRubric.id) return true;
      })[0].hints;
      setHints(newHints);

      // * local
      // setShowVideoEditor(true);
      // return;

      // Actually do the upload
      try {
         setIsLoading(true);
         toast.info('Getting SAS token...');

         const sas = await getSasForFile(fileName);
         toast.success('SAS received. Uploading video...');

         await uploadFileToBlob(rawFile, sas);
         toast.success('Upload complete!');
         setSasUrl(sas);

         // Show the editor now that the video is in Azure
         setShowVideoEditor(true);
      } catch (err) {
         toast.error(`Error uploading video: ${err.message}`);
         console.error('Upload error:', err);
      } finally {
         setIsLoading(false);
      }
   };

   const handleAnalyze = async () => {
      // must have all 5 stages saved
      if (!isStagesSaved) {
         toast.error('Please save all 5 stages before analyzing');
         return;
      }

      // * local
      setIsFeedback(true);
      navigate('/feedback');

      if (!sasUrl) {
         toast.error('No SAS URL found. Did you upload the video first?');
         return;
      }

      if (isTestBtn) {
         const reqUrl = `https://athleticstorage.blob.core.windows.net/results/420/enhanced_feedback.json?sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-01-28T06:50:58Z&st=2025-01-27T22:50:58Z&spr=https,http&sig=f7igcykzBuMLQkmHHG%2BI%2FBBgiFalKGxHXf%2BrxVmK8Gc%3D`;

         const resp = await fetch(reqUrl);

         if (!resp.ok) {
            throw new Error(`HTTP error! status: ${resp.status}`);
         }

         const feedbackData = await resp.json();
         setFeedbackData(feedbackData);
         setIsFeedback(true);
         navigate('/feedback');
         return;
      }

      try {
         setIsLoading(true);
         toast.info('Sending process_video request...');

         await processVideo(sasUrl);
         toast.success('Video start processing successfully!');
      } catch (err) {
         toast.error(`Error analyzing video: ${err.message}`);
         console.error('Analyze error:', err);
      } finally {
         setIsLoading(false);
      }
   };

   // 4) GET SAS
   const getSasForFile = async (filename) => {
      const functionUrl = 'https://dotnet-funcapp.azurewebsites.net/api/GetSasToken';
      const functionKey = 'Q6v1oyVl2oIJwIZ_xoavMGXw61OuvrpcjSo0irk4erKwAzFu9Z61nA%3D%3D';
      const reqUrl = `${functionUrl}?code=${functionKey}&filename=${encodeURIComponent(filename)}`;

      const resp = await fetch(reqUrl);
      if (!resp.ok) {
         throw new Error(`SAS error: HTTP ${resp.status}`);
      }
      const data = await resp.json();
      return data.sas_url;
   };

   // 5) PUT file to Blob
   const uploadFileToBlob = async (file, sasUrl) => {
      const resp = await fetch(sasUrl, {
         method: 'PUT',
         headers: {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': file.type,
            'Access-Control-Allow-Origin': '*',
         },
         body: file,
         mode: 'cors',
      });
      if (!resp.ok) {
         throw new Error(`Blob upload failed: HTTP ${resp.status}`);
      }
   };

   // 6) POST process_video with frames
   const processVideo = async (uploadedSasUrl) => {
      // Convert your stage_name => name, keep frames in start_time/end_time
      const mappedStages = rubric.stages.map((st) => ({
         name: st.stage_name,
         start_time: st.start_time ?? 0,
         end_time: st.end_time ?? 0,
      }));

      // * uploading a crypto-miner
      const processing_id = crypto.randomUUID();

      // console.log(processing_id);

      // If your chosen sport is stored in e.g. currentRubric.name, fallback to “shotput”
      // const exercise = currentRubric?.name || 'shotput';
      const exercise = 'shotput';

      // Format the user name: “Doe_John”
      const userName = formatStudentName(selectedStudent);

      const payload = {
         processing_id: processing_id,
         exercise,
         video_url: uploadedSasUrl,
         stages: mappedStages,
         user_id: userName,
         deployment_id: 'preprod',
         timestamp: new Date().toISOString(),
      };
      // console.log(payload);

      const functionUrl = 'https://dotnet-funcapp.azurewebsites.net/api/process_video';
      const functionKey =
         import.meta.env.VITE_ProcessVideoFunctionKey || process.env.VITE_ProcessVideoFunctionKey;
      console.log(functionKey, 'functionKey');
      const requestUrl = `${functionUrl}?code=${functionKey}`;

      const resp = await fetch(requestUrl, {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify(payload),
      });

      if (!resp.ok) {
         throw new Error(`process_video failed: HTTP ${resp.status}`);
      }

      if (resp.ok) {
         setIsTestBtn(true);
         setProcessingId(processing_id);
      }
      return await resp.json();
   };

   // Format: “John Doe” => “Doe_John”
   const formatStudentName = (fullName) => {
      if (!fullName || typeof fullName !== 'string') return 'Default_User';
      const parts = fullName.trim().split(/\s+/);
      if (parts.length < 2) return parts[0];
      return `${parts[1]}_${parts[0]}`;
   };

   const handleTitleClick = () => {
      if (easterEgg[0].length > easterEgg[1]) {
         const newEasterEgg = [easterEgg[0], easterEgg[1] + 1];
         toast.info(easterEgg[0][easterEgg[1]]);
         setEasterEgg(newEasterEgg);
      }
   };

   const handleSaveVideo = async () => {
      if (!crossOriginIsolated) {
         toast.error('Cross-origin isolation is required for video processing.');
         console.error('Cross-origin isolation is not enabled.');
         return;
      }

      if (!rawFile || startFrame >= endFrame) {
         toast.error('Invalid trimming selection');
         return;
      }

      if (!ffmpeg.isLoaded()) {
         toast.info('Loading video processing engine...');
         await ffmpeg.load();
      }

      try {
         toast.info('Trimming video, please wait...');

         // Convert start and end frames to seconds (assuming 30 fps)
         const startSeconds = startFrame / 30;
         const endSeconds = endFrame / 30;

         // Extract file extension from original file name
         const fileExtension = fileName.split('.').pop();
         const inputFileName = `input.${fileExtension}`;
         const outputFileName = `trimmed_${fileName}`;

         // Load the uploaded video into FFmpeg's virtual file system
         ffmpeg.FS('writeFile', inputFileName, await fetchFile(rawFile));

         // Execute FFmpeg command to trim the video
         await ffmpeg.run(
            '-i',
            inputFileName,
            '-ss',
            startSeconds.toString(),
            '-to',
            endSeconds.toString(),
            '-c',
            'copy',
            outputFileName,
         );

         // Retrieve the trimmed video from FFmpeg's file system
         const trimmedData = ffmpeg.FS('readFile', outputFileName);

         // Create a Blob from the output file and generate a URL
         const trimmedBlob = new Blob([trimmedData.buffer], { type: rawFile.type });
         const trimmedUrl = URL.createObjectURL(trimmedBlob);

         // Set the new trimmed video details
         setVideoSrc(trimmedUrl);
         setFileName(outputFileName);
         setRawFile(trimmedBlob);
         setShowVideoCut(false);

         toast.success(`Video trimmed successfully as ${outputFileName}`);
      } catch (error) {
         toast.error('Error trimming video');
         console.error('FFmpeg error:', error);
      }
   };

   return (
      <div className={s.newAnalysis}>
         {!showVideoCut && !showVideoEditor && (
            <Sidebar
               currentRubric={currentRubric}
               setCurrentRubric={setCurrentRubric}
               rubrics={rubrics}
               isSidebarShow={isSidebarShow}
               setIsSidebarShow={setIsSidebarShow}
            />
         )}
         <div className={s.newAnalysis__main}>
            {!showVideoCut && !showVideoEditor && <div className={s.newAnalysis__left}></div>}
            {showVideoEditor ? (
               <VideoEditor
                  videoSrc={videoSrc}
                  setIsStagesSaved={setIsStagesSaved}
                  rubric={rubric}
                  setRubric={setRubric}
                  startFrame={startFrame}
                  setStartFrame={setStartFrame}
                  endFrame={endFrame}
                  setEndFrame={setEndFrame}
                  currentStage={currentStage}
                  setCurrentStage={setCurrentStage}
               />
            ) : showVideoCut ? (
               <VideoEditor
                  isVideoCut={true}
                  videoSrc={videoSrc}
                  setIsStagesSaved={setIsStagesSaved}
                  rubric={rubric}
                  setRubric={setRubric}
                  fromTime={fromTime}
                  setFromTime={setFromTime}
                  toTime={toTime}
                  setToTime={setToTime}
                  startFrame={startFrame}
                  setStartFrame={setStartFrame}
                  endFrame={endFrame}
                  setEndFrame={setEndFrame}
                  isDurationValid={isDurationValid}
                  setIsDurationValid={setIsDurationValid}
               />
            ) : (
               <div className={s.newAnalysis__content}>
                  <div className={s.newAnalysis__top}>
                     <div className={s.newAnalysis__title} onClick={handleTitleClick}>
                        Create a new analysis
                     </div>

                     <ChooseStudent
                        students={students}
                        setStudents={setStudents}
                        setSelectedStudent={setSelectedStudent}
                        title={title}
                        setTitle={setTitle}
                     />
                  </div>
                  <div className={s.newAnalysis__submitWrap}>
                     <UploadVideo
                        onUpload={handleVideoUpload}
                        fileName={fileName}
                        setFileName={setFileName}
                     />
                     <form action="#" onSubmit={handleSubmit}>
                        <button
                           type="submit"
                           className={s.newAnalysis__submit}
                           disabled={isLoading}>
                           {isLoading ? 'Uploading...' : 'Submit'}
                        </button>
                     </form>
                  </div>
                  <ExampleVideo />
               </div>
            )}
            {showVideoCut && (
               <div className={s.newAnalysis__right}>
                  <div className={s.newAnalysis__mainBlock}>
                     <div className={s.newAnalysis__cutTitle}>Cut the video</div>
                     <p className={s.newAnalysis__descr}>
                        Video is too long, please cut it to 30 seconds
                     </p>
                     <div className={s.newAnalysis__time}>
                        <div className={s.newAnalysis__cut}>
                           From <span>{fromTime}</span>
                        </div>
                        <div className={s.newAnalysis__cut}>
                           To <span>{toTime}</span>
                        </div>
                     </div>
                     <button
                        className={s.newAnalysis__cutBtn}
                        onClick={handleSaveVideo}
                        disabled={!isDurationValid}>
                        Save video
                     </button>
                  </div>
               </div>
            )}
            {showVideoEditor && (
               <div className={s.newAnalysis__right}>
                  <div className={s.newAnalysis__mainBlock}>
                     <div className={s.newAnalysis__cutTitle}>Analyze video</div>
                     <p className={s.newAnalysis__descr}>Complete all stages to get feedback</p>
                     <button
                        className={s.newAnalysis__cutBtn}
                        onClick={handleAnalyze}
                        disabled={isLoading}>
                        {isLoading ? 'Processing...' : 'Analyze'}
                     </button>
                  </div>
                  <VideoHint currentStage={currentStage} hints={hints} />
               </div>
            )}
         </div>
      </div>
   );
};

export default NewAnalysis;
