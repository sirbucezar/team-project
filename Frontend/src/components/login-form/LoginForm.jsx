import React, { useEffect, useRef, useState } from 'react';
import s from './styles.module.scss';
import LoginButton from './login-button/LoginButton';

const LoginForm = ({ isAnimEnded }) => {
   const [activeTab, setActiveTab] = useState('Teacher');
   const indicatorRef = useRef(null);
   const sidebarRef = useRef(null);

   useEffect(() => {
      if (!sidebarRef.current) return;
      const activeItem = sidebarRef.current.querySelector(`.${s.loginFrom__item}.${s.active}`);
      if (activeItem && indicatorRef.current) {
         indicatorRef.current.style.width = `${activeItem.offsetWidth}px`;
         indicatorRef.current.style.left = `${activeItem.offsetLeft}px`;
      }
   }, [activeTab]);
   return (
      <div className={`${s.loginFrom} ${isAnimEnded ? s.active : ''}`.trim()}>
         <div className={s.loginFrom__main}>
            <div className={s.loginFrom__container}>
               <div className={s.loginFrom__body}>
                  <div className={s.loginFrom__title}>Login</div>
                  <div className={s.loginFrom__sidebar} ref={sidebarRef}>
                     <div
                        className={`${s.loginFrom__item} ${
                           activeTab === 'Teacher' ? s.active : ''
                        }`}
                        onClick={() => setActiveTab('Teacher')}>
                        Teacher
                     </div>
                     <div
                        className={`${s.loginFrom__item} ${
                           activeTab === 'Student' ? s.active : ''
                        }`}
                        onClick={() => setActiveTab('Student')}>
                        Student
                     </div>
                     <div ref={indicatorRef} className={s.loginFrom__indicator}></div>
                  </div>
                  <div className={s.loginFrom__buttons}>
                     <LoginButton type="microsoft" />
                     <LoginButton type="google" />
                  </div>
               </div>
            </div>
         </div>
      </div>
   );
};

export default LoginForm;
