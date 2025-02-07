import React from 'react';
import s from './styles.module.scss';

const LoginButton = ({ type }) => {
   return (
      <button className={s.btn}>
         <div className={s.btn__icon}>
            {type === 'microsoft' ? (
               <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M1.24512 1.19453H9.55879V9.50743H1.24512V1.19453Z" />
                  <path d="M10.4258 1.19453H18.7387V9.50743H10.4258V1.19453Z" />
                  <path d="M1.24512 10.3754H9.55879V18.6879H1.24512V10.3754Z" />
                  <path d="M10.4258 10.3754H18.7387V18.6879H10.4258V10.3754Z" />
               </svg>
            ) : (
               <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 488 512">
                  <path d="M488 261.8C488 403.3 391.1 504 248 504 110.8 504 0 393.2 0 256S110.8 8 248 8c66.8 0 123 24.5 166.3 64.9l-67.5 64.9C258.5 52.6 94.3 116.6 94.3 256c0 86.5 69.1 156.6 153.7 156.6 98.2 0 135-70.4 140.8-106.9H248v-85.3h236.1c2.3 12.7 3.9 24.9 3.9 41.4z" />
               </svg>
            )}
         </div>
         <div className={s.btn__main}>
            <div className={s.btn__text}>
               Continue with {type === 'microsoft' ? 'Microsoft' : 'Google'}
            </div>
         </div>
      </button>
   );
};

export default LoginButton;
