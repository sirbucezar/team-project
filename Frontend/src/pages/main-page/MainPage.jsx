import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router';
import TopBar from './top-bar/TopBar.jsx';
import NewAnalysis from '../new-analysis/NewAnalysis.jsx';
import Overview from '../overview/Overview.jsx';
import Search from '../search/Search.jsx';
import Feedback from '../feedback/Feedback.jsx';
import s from './styles.module.scss';
import { Toaster } from 'sonner';
import { io } from 'socket.io-client';
import Header from './header/Header.jsx';

const SOCKET_SERVER_URL = 'https://dotnet-funcapp.azurewebsites.net'; // Adjust if needed

const useSocket = (processingId) => {
   const [logs, setLogs] = useState([]);
   const socketRef = useRef(null);

   useEffect(() => {
      if (!processingId) return;

      console.log(`🔌 Connecting to WebSocket for Processing ID: ${processingId}`);

      socketRef.current = io(SOCKET_SERVER_URL, {
         path: `/api/status/${processingId}`,
         transports: ['websocket'],
      });

      socketRef.current.on('log_update', (log) => {
         setLogs((prevLogs) => [...prevLogs, log]);
      });

      socketRef.current.on('connect_error', (err) => {
         console.error('WebSocket Connection Error:', err);
      });

      return () => {
         console.log('🛑 Disconnecting WebSocket');
         if (socketRef.current) {
            socketRef.current.disconnect();
            socketRef.current = null;
         }
      };
   }, [processingId]);

   // ✅ Function to manually disconnect WebSocket
   const disconnect = () => {
      if (socketRef.current) {
         console.log('🚫 Manually Disconnecting WebSocket');
         socketRef.current.disconnect();
         socketRef.current = null;
      }
   };

   return { logs, disconnect };
};

const MainPage = () => {
   const [showVideoEditor, setShowVideoEditor] = useState(false);
   const [isFeedback, setIsFeedback] = useState(true);
   const [feedbackData, setFeedbackData] = useState(null);
   // const [currentRubric, setCurrentRubric] = useState({ id: 2, name: 'shotup' });
   const [currentRubric, setCurrentRubric] = useState(null);
   const [processingId, setProcessingId] = useState(null);
   const [isSidebarShow, setIsSidebarShow] = useState(false);

   // const { logs, disconnect } = processingId
   //    ? useSocket(processingId)
   //    : { logs: [], disconnect: () => {} };

   // useEffect(() => {
   //    if (logs.length > 0) {
   //       console.log('Socket Logs:', logs);
   //    }
   // }, [logs]);

   const [rubrics, setRubrics] = useState([
      {
         id: 0,
         name: 'Start',
         rubrics: [
            'Pelvis slightly higher than shoulders at "set"',
            'Head aligned with torso, looking at start line',
            'Both legs push off powerfully, causing imbalance',
            'Body positioned like "a spear" during push-off',
            'Gaze directed slightly forward toward ground',
         ],
         hints: [
            'Select the starting position when the athlete is in the "set" stance.',
            'Cut to include the push-off phase where both legs powerfully extend.',
            'Include the transition where the body moves into a spear-like position.',
            'Ensure the gaze and head alignment are visible in the video.',
            'End the cut just after the athlete begins forward motion.',
         ],
      },
      {
         id: 1,
         name: 'Sprint',
         rubrics: [
            'Runs on the balls of the feet',
            'Knees are high',
            'Active clawing motion of legs',
            'Arms at 90° actively moving',
            'Center of mass leans forward',
         ],
         hints: [
            'Start the cut when the athlete begins the sprint.',
            'Capture the running form showing knees lifting high.',
            'Ensure the clawing motion of legs is visible.',
            'Include the movement of arms at 90° angles.',
            'End the cut when the sprint reaches full speed.',
         ],
      },
      {
         id: 2,
         name: 'Shot Put',
         rubrics: [
            'Glide phase initiated with bent low leg, back to throw',
            'Trailing leg pulled under pelvis after hop',
            'Bracing leg planted; push leg bent after hop',
            'Push leg extends, hip-torso-arm sequence begins',
            'Shot remains at neck until 45° release',
         ],
         hints: [
            'Begin the cut with the athlete initiating the glide phase.',
            'Include the moment when the trailing leg pulls under the pelvis.',
            'Capture the bracing leg plant and push leg action.',
            'Focus on the hip-torso-arm movement leading to the throw.',
            'End the cut right after the shot is released.',
         ],
      },
      {
         id: 3,
         name: 'Height Jump',
         rubrics: [
            'Accelerating approach with upright posture',
            'Lean into curve',
            'Full knee lift during takeoff, arm lifted high',
            'Clears bar with an arched back',
            'Lands on mat in "L" shape, perpendicular to bar',
         ],
         hints: [
            'Start the cut during the accelerating approach.',
            'Include the leaning motion into the curve.',
            'Focus on the takeoff with knee lift and arm motion.',
            'Capture the athlete clearing the bar with an arched back.',
            'End the cut when the athlete lands on the mat.',
         ],
      },
      {
         id: 4,
         name: 'Hurdles',
         rubrics: [
            '8 steps in approach',
            'First hurdle cleared',
            'Fully extended lead leg passes hurdle',
            'Torso and opposite arm align with lead leg',
            'Large stride for second contact post-hurdle',
         ],
         hints: [
            'Start the cut during the 8-step approach.',
            'Capture the clearing of the first hurdle.',
            'Focus on the lead leg fully extending over the hurdle.',
            'Include the alignment of torso and opposite arm.',
            'End the cut after the large stride following the hurdle.',
         ],
      },
      {
         id: 5,
         name: 'Long Jump',
         rubrics: [
            'Accelerating approach without slowing before takeoff',
            'Foot lands on white plank; no gaze at board',
            'Takeoff foot flat, center of mass above it',
            'Knight stance maintained during first half',
            'Landing with sliding technique',
         ],
         hints: [
            'Begin the cut with the accelerating approach.',
            'Include the moment when the foot lands on the white plank.',
            'Focus on the takeoff phase with proper form.',
            'Capture the knight stance in the air.',
            'End the cut after the athlete lands with a sliding technique.',
         ],
      },
      {
         id: 6,
         name: 'Discus Throw',
         rubrics: [
            'Throwing arm swings back after initial motion',
            'Pivot initiated from ball of foot',
            'Pivot performed flat toward circle center',
            'Throw starts low-to-high with hip engagement',
            'Discus released via index finger',
         ],
         hints: [
            'Start the cut as the throwing arm swings back.',
            'Include the pivot initiated from the ball of the foot.',
            'Focus on the flat pivot toward the circle center.',
            'Capture the throw starting low-to-high with hip motion.',
            'End the cut at the discus release.',
         ],
      },
      {
         id: 7,
         name: 'Javelin Throw',
         rubrics: [
            'Javelin brought backward during last 5 steps',
            'Pelvis rotated, javelin fully retracted',
            'Impulse step executed',
            'Blocking step executed',
            'Throw initiated through hip-torso involvement',
         ],
         hints: [
            'Begin the cut with the javelin brought backward in the last steps.',
            'Include the rotation of the pelvis and javelin retraction.',
            'Focus on the execution of the impulse step.',
            'Capture the blocking step before the throw.',
            'End the cut at the moment of javelin release.',
         ],
      },
      {
         id: 8,
         name: 'Relay Race',
         rubrics: [
            'Receiver starts after runner crosses mark without looking back',
            'Receiver reaches maximum speed',
            'Baton exchange occurs at speed after agreed signal',
            'Baton switched hands, runner stays in lane',
            'Exchange occurs within the zone',
         ],
         hints: [
            'Start the cut when the receiver begins moving after the mark.',
            'Include the receiver reaching maximum speed.',
            'Capture the baton exchange at full speed.',
            'Focus on the baton switching hands and the runner staying in lane.',
            'End the cut after the baton is exchanged within the zone.',
         ],
      },
   ]);

   return (
      <div className="wrapper">
         <Toaster closeButton richColors position="bottom-right" className="toaster" />
         <div className={s.mainPage}>
            <div className={`${s.mainPage__container} _container`}>
               <Routes>
                  <Route
                     path="/"
                     element={
                        <>
                           <Header
                              isSidebarShow={isSidebarShow}
                              setIsSidebarShow={setIsSidebarShow}
                           />
                           {!showVideoEditor && <TopBar />}
                           <NewAnalysis
                              setFeedbackData={setFeedbackData}
                              setIsFeedback={setIsFeedback}
                              showVideoEditor={showVideoEditor}
                              setShowVideoEditor={setShowVideoEditor}
                              rubrics={rubrics}
                              currentRubric={currentRubric}
                              setCurrentRubric={setCurrentRubric}
                              setProcessingId={setProcessingId}
                              isSidebarShow={isSidebarShow}
                              setIsSidebarShow={setIsSidebarShow}
                           />
                        </>
                     }
                  />
                  <Route path="/search" element={<Search />} />
                  <Route
                     path="/feedback"
                     element={
                        <Feedback
                           feedbackData={feedbackData}
                           isFeedback={isFeedback}
                           rubrics={rubrics}
                           currentRubric={currentRubric}
                        />
                     }
                  />
               </Routes>
            </div>
         </div>
      </div>
   );
};

export default MainPage;
