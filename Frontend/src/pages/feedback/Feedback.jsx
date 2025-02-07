import React, { useEffect, useState } from 'react';
import { Navigate, useNavigate } from 'react-router';
import Stages from './stages/Stages';
import s from './styles.module.scss';
import LoadingScreen from './loading-screen/LoadingScreen';
import { Link } from 'react-router';

const Feedback = ({ feedbackData, isFeedback, currentRubric }) => {
   let navigate = useNavigate();
   // For now, let's just keep dummy data in state
   const [isLoading, setIsLoading] = useState(true);
   const [isLoadingShown, setIsLoadingShown] = useState(true);
   const [isExploding, setIsExploding] = useState(false);
   const [analysisData, setAnalysisData] = useState(null);

   useEffect(() => {
      if (isExploding) {
         setTimeout(() => {
            setIsLoadingShown(false);
         }, 500);
      }
   }, [isExploding]);

   useEffect(() => {
      if (!isFeedback) {
         navigate('/'); // Redirect to '/'
         return;
      }
      // SIMULATE an API call with setTimeout
      // Replace with real fetch to .NET or your DB
      setTimeout(() => {
         const dummyResponse = {
            status: 'Augmentation successful',
            processing_id: '421',
            feedback: [
               {
                  stage: 'Shot Put - Stage1',
                  criterion: 'Glide phase initiated with bent low leg, back to throw',
                  score: '1',
                  confidence: '0.90',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your low leg remains properly bent, creating an explosive, controlled glide backward.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Maintain that approach; add plyometric exercises to further enhance your starting leg power.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A sustained bend optimizes hip-knee-ankle alignment, driving a powerful transition into the power stance.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Great setup—keeping that leg flexed at the start is fueling a dynamic glide every time.',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'No significant injury risk noted. Keep executing with control.',
                  },
                  visualization_tip:
                     'Imagine storing power in your bent leg like a compressed spring before pushing off.',
               },
               {
                  stage: 'Shot Put - Stage2',
                  criterion: 'Trailing leg pulled under pelvis after hop',
                  score: '0.5',
                  confidence: '0.74',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'You sometimes manage to tuck the trailing leg under, but it’s inconsistent and timing varies.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Use rhythmic drills: hop and count ‘one-two,’ ensuring the leg tucks under on ‘two’ every time.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'Consistent leg tuck helps you land in a strong power position, stabilizing your body for the put.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'You’re halfway there—sharpen that timing, and the rest of the throw will feel smoother.',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'Minor risk of imbalance. Focus on controlled foot placement.',
                  },
                  visualization_tip:
                     'Visualize snapping your trailing foot underneath like a pendulum swinging into place.',
               },
               {
                  stage: 'Shot Put - Stage3',
                  criterion: 'Bracing leg planted; push leg bent after hop',
                  score: '0',
                  confidence: '0.76',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your front (bracing) leg drifts forward without proper contact, and the push leg remains too straight.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Consciously drive your front foot into the ground; keep your back knee flexed for a stronger drive.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A solid front plant anchors your body, while a bent push leg coils energy for the final extension.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'No worries—nailing that front-back leg setup will drastically boost throw consistency.',
                     },
                  },
                  injury_risk: {
                     high_risk: true,
                     disclaimer:
                        'Improper bracing could strain the knee joint. Proceed with caution.',
                  },
                  visualization_tip:
                     'Picture your bracing leg as an anchor keeping you grounded while the push leg builds power.',
               },
               {
                  stage: 'Shot Put - Stage4',
                  criterion: 'Push leg extends, hip-torso-arm sequence begins',
                  score: '1',
                  confidence: '0.92',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'The hip-to-arm sequence was well-timed and powerful.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Focus on maintaining this timing for consistency.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'Proper sequence ensures maximum force transfer to the shot.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Fantastic job—your sequence is setting you up for success!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'No significant injury risk noted. Keep up the good work!',
                  },
                  visualization_tip:
                     'Picture the force traveling seamlessly from your hips to your arm in one fluid motion.',
               },
               {
                  stage: 'Shot Put - Stage5',
                  criterion: 'Shot remains at neck until 45° release',
                  score: '1',
                  confidence: '0.90',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'You consistently secure the shot at your neck and fire it at a textbook 45°, maximizing distance.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Keep refining your rhythm—slight changes in hip drive or speed might tweak the release angle perfectly.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A well-timed 45° launch uses gravity and forward velocity for a long flight path with minimal energy loss.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Fantastic work—your shot put form is an excellent blend of power and precision!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'Ensure proper wrist and arm alignment to prevent strain.',
                  },
                  visualization_tip:
                     'Visualize the shot traveling in a smooth arc towards the target at a perfect 45° angle.',
               },
            ],
         };

         const dummyResponse2 = {
            status: 'Augmentation successful',
            processing_id: '422',
            feedback: [
               {
                  stage: 'Start - Stage1',
                  criterion: 'Explosive push-off from blocks',
                  score: '1',
                  confidence: '0.88',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your start is strong, with an explosive push-off and good reaction time.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Maintain your drive phase longer to optimize acceleration.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A powerful push-off generates the necessary speed to transition smoothly into the sprint.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Great start—your explosiveness sets the foundation for a fast sprint!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer:
                        'No significant injury risk noted. Continue executing with precision.',
                  },
                  visualization_tip:
                     'Picture yourself launching like a rocket, pushing the track behind you.',
               },
               {
                  stage: 'Start - Stage2',
                  criterion: 'First step length and angle',
                  score: '0',
                  confidence: '0.72',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your first step is too short, causing inefficient acceleration.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Work on extending your first stride while keeping a low drive angle.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A well-extended first step ensures optimal force application and seamless momentum buildup.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Keep at it—adjusting this step will drastically improve your acceleration phase!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'Minor risk of strain if overextending. Maintain proper balance.',
                  },
                  visualization_tip:
                     'Imagine stepping over a hurdle just beyond the blocks to encourage full leg extension.',
               },
               {
                  stage: 'Start - Stage3',
                  criterion: 'Arm drive synchronization',
                  score: '1',
                  confidence: '0.91',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your arm drive is powerful and well-coordinated with your lower body.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Continue emphasizing a strong arm pump to maximize forward momentum.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'Efficient arm drive enhances stability and contributes to overall acceleration.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Awesome arm mechanics—this is helping you generate extra speed!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer: 'No injury risk detected. Keep focusing on fluid movement.',
                  },
                  visualization_tip:
                     'Think of your arms pulling the track behind you as you sprint forward.',
               },
               {
                  stage: 'Start - Stage4',
                  criterion: 'Head and torso alignment',
                  score: '1',
                  confidence: '0.87',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your head remains neutral, and your torso leans forward at an ideal angle.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Sustain this posture throughout the drive phase for optimal acceleration.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A forward-leaning torso helps maintain momentum and reduces aerodynamic resistance.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Excellent posture—keeping this form will help you reach top speed efficiently!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer:
                        'No injury risk detected. Maintain this posture as speed increases.',
                  },
                  visualization_tip:
                     'Visualize yourself leaning into the race, using gravity to propel forward.',
               },
               {
                  stage: 'Start - Stage5',
                  criterion: 'Transition into upright sprinting',
                  score: '1',
                  confidence: '0.92',
                  feedback: {
                     Observation: {
                        title: 'Observation',
                        body: 'Your transition from acceleration to upright sprinting is seamless and controlled.',
                     },
                     'Improvement Suggestion': {
                        title: 'Improvement Suggestion',
                        body: 'Maintain gradual progression to full sprint posture to maximize speed retention.',
                     },
                     Justification: {
                        title: 'Justification',
                        body: 'A smooth transition prevents deceleration and allows peak velocity to be sustained.',
                     },
                     Encouragement: {
                        title: 'Encouragement',
                        body: 'Fantastic transition—you are setting yourself up for a strong finish!',
                     },
                  },
                  injury_risk: {
                     high_risk: false,
                     disclaimer:
                        'No injury risk detected. Continue refining this phase for optimal results.',
                  },
                  visualization_tip:
                     'Imagine yourself smoothly rising from a crouched acceleration phase into a full-speed sprint.',
               },
            ],
         };

         setAnalysisData(currentRubric.id === 2 ? dummyResponse : dummyResponse2);
         setIsLoading(false);
      }, 10000);

      console.log('Server says:', analysisData);
   }, []);

   // if (isLoading) {
   //    return (

   //    );
   // }

   // if (!analysisData) {
   //    return;
   // }

   // useEffect(() => {}, [analysisData]);

   //    // const { stageAnalysis, metrics } = analysisData;
   //    // const overallScore = metrics?.overall_score ?? 0;

   //    // Convert stageAnalysis object to an array of stage details
   //    // const stageEntries = Object.keys(stageAnalysis).map((stageKey) => ({
   //    //    name: stageKey,
   //    //    score: stageAnalysis[stageKey]?.score,
   //    //    videoUrl: stageAnalysis[stageKey]?.video_url,
   //    //    feedback: stageAnalysis[stageKey]?.feedback,
   //    //    confidence: stageAnalysis[stageKey]?.confidence,
   //    // }));

   useEffect(() => {
      // Warn user before leaving page (only before form submission)
      const handleBeforeUnload = (event) => {
         if (isFeedback) {
            event.preventDefault();
            event.returnValue = ''; // Standard browser behavior
         }
      };

      window.addEventListener('beforeunload', handleBeforeUnload);

      return () => {
         window.removeEventListener('beforeunload', handleBeforeUnload);
      };
   }, [isFeedback]);

   return (
      <>
         {isLoadingShown && (
            <div className={s.loading}>
               <LoadingScreen
                  isExploding={isExploding}
                  setIsExploding={setIsExploding}
                  isLoading={isLoading}
                  currentRubric={currentRubric}
               />
            </div>
         )}
         {!isLoading && (
            <div className={s.feedback}>
               <div className={s.feedback__top}>
                  <h1 className={s.feedback__title}>Rubric: {currentRubric.name}</h1>
                  <div className={s.feedback__score}>{currentRubric.id === 2 ? 3.5 : 4}/5</div>
               </div>

               {/* Render stage cards */}
               {/* {!!analysisData && } */}
               <Stages currentRubric={currentRubric} stageEntries={analysisData.feedback} />
               <div className={s.feedback__bottom}>
                  <Link to="/" className={s.feedback__btnNew}>
                     New analysis
                  </Link>
                  {/* <button className={s.feedback__btn}>Download report</button> */}
               </div>
            </div>
         )}
      </>
   );
};

export default Feedback;
